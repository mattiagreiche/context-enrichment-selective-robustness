"""Convert four validated probability run summaries to the released CSV layout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "model", "batch", "world", "context", "surface", "prompt_sha256", "status",
    "transfer_reported_mass", "retain_reported_mass", "unreturned_probability_mass",
    "transfer_probability_lower", "transfer_probability_upper",
    "transfer_probability_midpoint",
)
STATUSES = {
    "complete", "censored_missing_retain_branch", "censored_missing_transfer_branch"
}


def convert(sources: dict[tuple[str, str], Path]) -> list[dict]:
    manifest = json.loads(
        (ROOT / "tests/fixtures/prompts/hospital/pilot_v4/manifest.json").read_text()
    )
    hashes = {cell["condition_id"]: cell["prompt_sha256"] for cell in manifest["cells"]}
    if len(hashes) != 40:
        raise ValueError("expected 40 frozen prompt hashes")

    rows: list[dict] = []
    all_call_ids: set[str] = set()
    for model in ("Sol", "Terra"):
        for batch in ("initial", "extension"):
            path = sources[model, batch]
            payload = json.loads(path.read_text())
            if (
                payload.get("analysis") != "hospital_first_action_branch_probabilities"
                or payload.get("measurement_protocol") != "first_action_branch_v1"
                or payload.get("prompt_version") != "pilot_v4"
                or payload.get("requested_model_identifier") != f"gpt-5.6-{model.lower()}"
                or payload.get("scheduled_call_count") != 200
            ):
                raise ValueError(f"{path}: summary metadata does not match the paper protocol")
            observations = payload.get("observations")
            if not isinstance(observations, list) or len(observations) != 200:
                raise ValueError(f"{path}: expected 200 observations")
            counts: Counter[str] = Counter()
            call_ids: set[str] = set()
            for observation in observations:
                condition = observation["condition_id"]
                if condition not in hashes or observation["prompt_sha256"] != hashes[condition]:
                    raise ValueError(f"{path}: unrecognized or changed prompt")
                if observation["call_id"] in call_ids or observation["call_id"] in all_call_ids:
                    raise ValueError(f"{path}: repeated call ID")
                call_ids.add(observation["call_id"])
                all_call_ids.add(observation["call_id"])
                counts[condition] += 1
                status = observation["status"]
                if status not in STATUSES:
                    raise ValueError(f"{path}: unsupported observation status {status!r}")
                t = float(observation["transfer_reported_mass"])
                r = float(observation["retain_reported_mass"])
                u = float(observation["unreturned_probability_mass"])
                denominator = t + r if status == "complete" else t + r + u
                if min(t, r, u) < 0 or denominator <= 0:
                    raise ValueError(f"{path}: invalid branch masses")
                lower = t / denominator
                upper = lower if status == "complete" else (t + u) / denominator
                world, context, surface = condition.split("/")[1:]
                rows.append(dict(zip(FIELDS, (
                    model, batch, world, context, surface, hashes[condition], status,
                    t, r, u, lower, upper, (lower + upper) / 2,
                ))))
            if set(counts) != set(hashes) or set(counts.values()) != {5}:
                raise ValueError(f"{path}: expected five observations per frozen cell")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ("sol", "terra"):
        for batch in ("initial", "extension"):
            parser.add_argument(f"--{model}-{batch}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = {
        (model.title(), batch): getattr(args, f"{model}_{batch}")
        for model in ("sol", "terra") for batch in ("initial", "extension")
    }
    rows = convert(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validated call rows to {args.output}")


if __name__ == "__main__":
    main()
