"""Recompute the paper's exact missing-branch sensitivity ranges from released data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from reproduce import CONTEXTS, MODELS, ROOT, SURFACES, WORLDS


def surface_bounds(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extrema of mean |alternative - shared canonical| over interval cells."""

    canonical_low, canonical_high = low[..., 0], high[..., 0]
    other_low, other_high = low[..., 1:], high[..., 1:]
    candidates = np.concatenate([low, high], axis=-1)
    candidates = np.clip(candidates, canonical_low[..., None], canonical_high[..., None])
    distance = np.maximum(other_low[..., None, :] - candidates[..., :, None], 0)
    distance += np.maximum(candidates[..., :, None] - other_high[..., None, :], 0)
    minimum = distance.mean(axis=-1).min(axis=-1)
    maximum = np.maximum(
        np.maximum(abs(other_low - canonical_low[..., None]), abs(other_high - canonical_low[..., None])).mean(axis=-1),
        np.maximum(abs(other_low - canonical_high[..., None]), abs(other_high - canonical_high[..., None])).mean(axis=-1),
    )
    return minimum, maximum


def measure_bounds(low: np.ndarray, high: np.ndarray) -> list[float]:
    surface_low, surface_high = surface_bounds(low, high)
    response_low = (low[0] - high[1]).mean()
    response_high = (high[0] - low[1]).mean()
    return [float(surface_low.mean() * 100), float(surface_high.mean() * 100),
            float(response_low * 100), float(response_high * 100)]


def main() -> None:
    with (ROOT / "data/paper_calls.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    paper = json.loads((ROOT / "data/paper_results.json").read_text())
    largest_change = 0.0
    ranges = {}
    for model in MODELS:
        ranges[model] = {}
        for context in CONTEXTS:
            low, high = np.empty((2, 4)), np.empty((2, 4))
            for w, world in enumerate(WORLDS):
                for s, surface in enumerate(SURFACES):
                    cell = [row for row in rows if (row["model"], row["context"], row["world"], row["surface"]) == (model, context, world, surface)]
                    assert len(cell) == 10
                    low[w, s] = np.mean([float(row["transfer_probability_lower"]) for row in cell])
                    high[w, s] = np.mean([float(row["transfer_probability_upper"]) for row in cell])
            exact = measure_bounds(low, high)
            saved = paper["censoring_sensitivity"]["models"][model][context]
            assert np.allclose(exact, saved, atol=1e-10, rtol=0)
            estimates = paper["models"][model]["pooled_ten_call_estimates"][context]
            midpoint = [estimates["surface_sensitivity_pp"], estimates["world_response_pp"]]
            largest_change = max(largest_change, *(abs(value - midpoint[i // 2]) for i, value in enumerate(exact)))
            ranges[model][context] = exact
        sensitivity_order = sorted(CONTEXTS, key=lambda name: paper["models"][model]["pooled_ten_call_estimates"][name]["surface_sensitivity_pp"])
        response_order = sorted(CONTEXTS, key=lambda name: paper["models"][model]["pooled_ten_call_estimates"][name]["world_response_pp"])
        for order, low_index, high_index in ((sensitivity_order, 0, 1), (response_order, 2, 3)):
            assert all(ranges[model][a][high_index] < ranges[model][b][low_index] for a, b in zip(order, order[1:]))
    assert abs(largest_change - paper["censoring_sensitivity"]["max_change_from_midpoint_pp"]) < 1e-10
    assert largest_change < 0.49
    print(f"Verified all 20 measure ranges; maximum midpoint change: {largest_change:.6f} pp")


if __name__ == "__main__":
    main()
