"""Verify released call data and recompute the paper's measures.

Run from any directory: python analysis/reproduce.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORLDS = ("regional_mutual_aid", "local_emergency_reserve")
CONTEXTS = (
    "minimal",
    "neutral_padded",
    "generic_background",
    "decision_grounded",
    "normative_grounded",
)
SURFACES = ("canonical_prose", "bullets", "key_value_table", "alternative_prose")
MODELS = ("Sol", "Terra")


def measures(cells: np.ndarray) -> tuple[float, float]:
    """Return surface sensitivity and regional-minus-local response in points."""

    surface = np.abs(cells[:, 1:] - cells[:, :1]).mean() * 100
    response = (cells[0] - cells[1]).mean() * 100
    return float(surface), float(response)


def load() -> tuple[dict[tuple[str, str, str, str], list[float]], dict]:
    paper = json.loads((ROOT / "data/paper_results.json").read_text())
    manifest = json.loads(
        (ROOT / "tests/fixtures/prompts/hospital/pilot_v4/manifest.json").read_text()
    )
    prompt_hashes = {entry["condition_id"]: entry["prompt_sha256"] for entry in manifest["cells"]}
    assert len(prompt_hashes) == 40
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    batches: Counter[tuple[str, str, str, str, str]] = Counter()
    with (ROOT / "data/paper_calls.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 800
    for row in rows:
        model, world, context, surface = (row[key] for key in ("model", "world", "context", "surface"))
        assert model in MODELS and world in WORLDS and context in CONTEXTS and surface in SURFACES
        assert row["batch"] in {"initial", "extension"}
        condition_id = f"hospital/{world}/{context}/{surface}"
        assert row["prompt_sha256"] == prompt_hashes[condition_id]
        t, r, u = (float(row[key]) for key in (
            "transfer_reported_mass", "retain_reported_mass", "unreturned_probability_mass"
        ))
        lower, upper, midpoint = (float(row[key]) for key in (
            "transfer_probability_lower", "transfer_probability_upper", "transfer_probability_midpoint"
        ))
        assert 0 <= lower <= midpoint <= upper <= 1 + 1e-12
        assert abs(midpoint - (lower + upper) / 2) < 1e-12
        if row["status"] == "complete":
            assert abs(lower - t / (t + r)) < 1e-12
            assert abs(upper - lower) < 1e-12
        else:
            assert row["status"] in {"censored_missing_retain_branch", "censored_missing_transfer_branch"}
            assert abs(lower - t / (t + r + u)) < 1e-12
            assert abs(upper - (t + u) / (t + r + u)) < 1e-12
        values[model, context, world, surface].append(midpoint)
        batches[model, row["batch"], context, world, surface] += 1
        counts[model, row["status"]] += 1
    assert len(values) == 80 and all(len(call_values) == 10 for call_values in values.values())
    assert len(batches) == 160 and set(batches.values()) == {5}
    assert sum(n for (_, status), n in counts.items() if status != "complete") == 339
    for model in MODELS:
        assert {status: count for (name, status), count in counts.items() if name == model} == paper["models"][model]["status_counts"]
    return values, paper


def analyze(values: dict, paper: dict, *, seed: int = 20260923) -> dict:
    rng = np.random.default_rng(seed)
    result = {"calls": 800, "cells": 80, "censored_calls": 339, "bootstrap_seed": seed, "models": {}}
    for model in MODELS:
        result["models"][model] = {}
        for context in CONTEXTS:
            calls = np.array([
                [values[model, context, world, surface] for surface in SURFACES]
                for world in WORLDS
            ])
            assert calls.shape == (2, 4, 10)
            surface, response = measures(calls.mean(axis=-1))
            published = paper["models"][model]["pooled_ten_call_estimates"][context]
            assert abs(surface - published["surface_sensitivity_pp"]) < 1e-10
            assert abs(response - published["world_response_pp"]) < 1e-10
            indices = rng.integers(0, 10, size=(10_000, 2, 4, 10))
            sampled = np.take_along_axis(np.broadcast_to(calls, indices.shape), indices, axis=-1).mean(axis=-1)
            boot_surface = np.abs(sampled[:, :, 1:] - sampled[:, :, :1]).mean(axis=(1, 2)) * 100
            boot_response = (sampled[:, 0] - sampled[:, 1]).mean(axis=1) * 100
            result["models"][model][context] = {
                "cell_transfer_probability_percent": {
                    world: {
                        surface_name: float(calls[w, s].mean() * 100)
                        for s, surface_name in enumerate(SURFACES)
                    }
                    for w, world in enumerate(WORLDS)
                },
                "surface_sensitivity_pp": surface,
                "world_response_pp": response,
                "bootstrap_surface_ci95_pp": np.quantile(boot_surface, [0.025, 0.975]).tolist(),
                "bootstrap_response_ci95_pp": np.quantile(boot_response, [0.025, 0.975]).tolist(),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduction")
    args = parser.parse_args()
    values, paper = load()
    result = analyze(values, paper)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recomputed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified 800 calls, 80 cells, 339 censored calls; wrote {args.output}")


if __name__ == "__main__":
    main()
