"""Pure extraction and summaries for first action-branch probabilities.

The frozen hospital prompts require exactly ``TRANSFER`` or ``RETAIN``.  Those
labels begin with different bytes, so the first non-whitespace output token
selects an action branch even when a label spans multiple provider tokens.
This module deliberately estimates that branch probability rather than
claiming an unavailable full-string probability.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from statistics import mean, stdev
from typing import Any

from .domain import DomainAction


TRANSFER_LABEL = DomainAction.TRANSFER.value
RETAIN_LABEL = DomainAction.RETAIN.value


class BranchProbabilityError(ValueError):
    """Raised when provider logprobs violate the measurement contract."""


@dataclass(frozen=True, slots=True)
class FirstBranchProbability:
    status: str
    first_action_token_index: int
    sampled_token: str
    sampled_action: str | None
    final_output: str
    completion_matches_sampled_branch: bool
    transfer_reported_mass: float
    retain_reported_mass: float
    other_reported_mass: float
    returned_candidate_mass: float
    unreturned_probability_mass: float
    conditional_transfer_probability: float | None
    returned_candidate_count: int


def extract_first_branch_probability(
    raw_response: Mapping[str, Any],
    *,
    output_text: str | None = None,
) -> FirstBranchProbability:
    """Extract the first non-whitespace TRANSFER-versus-RETAIN branch.

    Candidate tokens are classified by their decoded bytes.  All returned
    tokenizations compatible with the beginning of one exact action label are
    summed before normalizing the two reported branch masses.  Missing branch
    candidates remain censored; they are never converted to probability zero.
    """

    if not isinstance(raw_response, Mapping):
        raise BranchProbabilityError("raw provider response must be an object")
    final_output = output_text
    if final_output is None:
        value = raw_response.get("output_text")
        final_output = value if isinstance(value, str) else _output_text(raw_response)
    if not isinstance(final_output, str):
        raise BranchProbabilityError("provider response has no output text")

    positions = _output_logprob_positions(raw_response, final_output)
    selected_index = None
    selected: Mapping[str, Any] | None = None
    for index, position in enumerate(positions):
        sampled = _candidate_bytes(position, f"logprobs[{index}]")
        if sampled.strip(b" \t\r\n\f\v"):
            selected_index = index
            selected = position
            break
    if selected_index is None or selected is None:
        raise BranchProbabilityError("output logprobs contain no action-bearing token")

    sampled_token = _candidate_text(selected, f"logprobs[{selected_index}]")
    candidates = _returned_candidates(selected, selected_index)
    transfer_mass = 0.0
    retain_mass = 0.0
    returned_mass = 0.0
    sampled_action = _compatible_action_bytes(
        _candidate_bytes(selected, f"logprobs[{selected_index}]")
    )
    for candidate in candidates:
        probability = _probability_from_logprob(
            candidate.get("logprob"), "candidate.logprob"
        )
        returned_mass += probability
        action = _compatible_action_bytes(_candidate_bytes(candidate, "candidate"))
        if action == TRANSFER_LABEL:
            transfer_mass += probability
        elif action == RETAIN_LABEL:
            retain_mass += probability

    if returned_mass > 1.0:
        scale = 1.0 / returned_mass
        transfer_mass *= scale
        retain_mass *= scale
        returned_mass = 1.0
    other_mass = max(0.0, returned_mass - transfer_mass - retain_mass)
    unreturned_mass = max(0.0, 1.0 - returned_mass)
    if transfer_mass <= 0 and retain_mass <= 0:
        status = "censored_missing_both_action_branches"
        conditional = None
    elif transfer_mass <= 0:
        status = "censored_missing_transfer_branch"
        conditional = None
    elif retain_mass <= 0:
        status = "censored_missing_retain_branch"
        conditional = None
    else:
        status = "complete"
        conditional = transfer_mass / (transfer_mass + retain_mass)

    normalized_output = final_output.strip()
    completion_matches = (
        sampled_action is not None and normalized_output == sampled_action
    )
    return FirstBranchProbability(
        status=status,
        first_action_token_index=selected_index,
        sampled_token=sampled_token,
        sampled_action=sampled_action,
        final_output=final_output,
        completion_matches_sampled_branch=completion_matches,
        transfer_reported_mass=transfer_mass,
        retain_reported_mass=retain_mass,
        other_reported_mass=other_mass,
        returned_candidate_mass=returned_mass,
        unreturned_probability_mass=unreturned_mass,
        conditional_transfer_probability=conditional,
        returned_candidate_count=len(candidates),
    )


def summarize_first_branch_probabilities(
    observations: Iterable[Mapping[str, Any]],
    *,
    scheduled_calls: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a complete schedule and summarize cell-level branch propensities."""

    rows = tuple(dict(row) for row in observations)
    schedule = tuple(dict(row) for row in scheduled_calls)
    if not rows or not schedule:
        raise BranchProbabilityError("observations and schedule must be nonempty")
    by_call: dict[str, dict[str, Any]] = {}
    for row in rows:
        call_id = _required_text(row.get("call_id"), "observation.call_id")
        if call_id in by_call:
            raise BranchProbabilityError(f"duplicate observation call_id {call_id!r}")
        by_call[call_id] = row
    scheduled_by_call: dict[str, dict[str, Any]] = {}
    for row in schedule:
        call_id = _required_text(row.get("call_id"), "schedule.call_id")
        if call_id in scheduled_by_call:
            raise BranchProbabilityError(f"duplicate scheduled call_id {call_id!r}")
        scheduled_by_call[call_id] = row
    if set(by_call) != set(scheduled_by_call):
        raise BranchProbabilityError("observations do not exactly cover the schedule")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call_id, row in by_call.items():
        scheduled = scheduled_by_call[call_id]
        for field in ("condition_id", "replicate_id"):
            if row.get(field) != scheduled.get(field):
                raise BranchProbabilityError(
                    f"observation {call_id!r} disagrees with scheduled {field}"
                )
        grouped[_required_text(row.get("condition_id"), "condition_id")].append(row)

    cells: list[dict[str, Any]] = []
    for condition_id in sorted(grouped):
        group = sorted(grouped[condition_id], key=lambda row: str(row["replicate_id"]))
        expected = sum(
            row.get("condition_id") == condition_id
            for row in scheduled_by_call.values()
        )
        if len(group) != expected:
            raise BranchProbabilityError(f"incomplete scheduled cell {condition_id}")
        complete_values = [
            float(row["conditional_transfer_probability"])
            for row in group
            if row.get("status") == "complete"
            and row.get("conditional_transfer_probability") is not None
        ]
        all_complete = len(complete_values) == len(group)
        exemplar = group[0]
        cells.append(
            {
                "condition_id": condition_id,
                "social_world": exemplar.get("social_world"),
                "context_condition": exemplar.get("context_condition"),
                "surface_version": exemplar.get("surface_version"),
                "prompt_sha256": exemplar.get("prompt_sha256"),
                "scheduled_count": expected,
                "complete_count": len(complete_values),
                "censored_count": len(group) - len(complete_values),
                "completion_mismatch_count": sum(
                    row.get("completion_matches_sampled_branch") is not True
                    for row in group
                ),
                "mean_conditional_transfer_probability": (
                    mean(complete_values) if all_complete else None
                ),
                "sample_sd_conditional_transfer_probability": (
                    stdev(complete_values)
                    if all_complete and len(complete_values) > 1
                    else (0.0 if all_complete else None)
                ),
                "minimum_conditional_transfer_probability": (
                    min(complete_values) if all_complete else None
                ),
                "maximum_conditional_transfer_probability": (
                    max(complete_values) if all_complete else None
                ),
                "mean_returned_action_branch_mass": mean(
                    float(row["transfer_reported_mass"])
                    + float(row["retain_reported_mass"])
                    for row in group
                ),
                "mean_unreturned_probability_mass": mean(
                    float(row["unreturned_probability_mass"]) for row in group
                ),
            }
        )
    surface_sensitivity = _surface_sensitivity(cells)
    substantive_response = _substantive_response(cells)
    return {
        "estimand": "conditional_first_action_branch_transfer_probability",
        "normalization": (
            "reported_TRANSFER_branch_mass/"
            "(reported_TRANSFER_branch_mass+reported_RETAIN_branch_mass)"
        ),
        "observation_count": len(rows),
        "cell_count": len(cells),
        "cells": cells,
        "surface_sensitivity": surface_sensitivity,
        "substantive_response": substantive_response,
    }


def observation_dict(
    *,
    call_id: str,
    replicate_id: str,
    condition_id: str,
    social_world: str,
    context_condition: str,
    surface_version: str,
    prompt_sha256: str,
    probability: FirstBranchProbability,
) -> dict[str, Any]:
    """Attach manifest and factor identity to one pure extraction result."""

    return {
        "call_id": call_id,
        "replicate_id": replicate_id,
        "condition_id": condition_id,
        "social_world": social_world,
        "context_condition": context_condition,
        "surface_version": surface_version,
        "prompt_sha256": prompt_sha256,
        **asdict(probability),
    }


def thaw_json(value: Any) -> Any:
    """Reverse the immutable JSON representation used by result validation."""

    if not isinstance(value, tuple):
        return value
    if value and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        for item in value
    ):
        return {key: thaw_json(item) for key, item in value}
    return [thaw_json(item) for item in value]


def _surface_sensitivity(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for cell in cells:
        key = (
            _required_text(cell.get("social_world"), "social_world"),
            _required_text(cell.get("context_condition"), "context_condition"),
        )
        surface = _required_text(cell.get("surface_version"), "surface_version")
        if surface in grouped[key]:
            raise BranchProbabilityError("duplicate surface probability cell")
        grouped[key][surface] = cell
    expected = {
        "canonical_prose",
        "bullets",
        "key_value_table",
        "alternative_prose",
    }
    rows: list[dict[str, Any]] = []
    for (world, context), by_surface in sorted(grouped.items()):
        complete_crossing = set(by_surface) == expected
        estimates = {
            surface: by_surface[surface].get(
                "mean_conditional_transfer_probability"
            )
            for surface in expected
            if surface in by_surface
        }
        estimable = complete_crossing and all(
            isinstance(value, (int, float)) for value in estimates.values()
        )
        differences = None
        average = None
        if estimable:
            canonical = float(estimates["canonical_prose"])
            differences = {
                surface: abs(float(estimates[surface]) - canonical)
                for surface in sorted(expected - {"canonical_prose"})
            }
            average = mean(differences.values())
        rows.append(
            {
                "social_world": world,
                "context_condition": context,
                "complete_surface_crossing": complete_crossing,
                "absolute_differences_from_canonical": differences,
                "mean_absolute_difference_from_canonical": average,
            }
        )
    return rows


def _substantive_response(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    for cell in cells:
        context = _required_text(cell.get("context_condition"), "context_condition")
        key = (
            _required_text(cell.get("social_world"), "social_world"),
            _required_text(cell.get("surface_version"), "surface_version"),
        )
        if key in grouped[context]:
            raise BranchProbabilityError("duplicate world/surface probability cell")
        grouped[context][key] = cell
    surfaces = {
        "canonical_prose",
        "bullets",
        "key_value_table",
        "alternative_prose",
    }
    worlds = {"regional_mutual_aid", "local_emergency_reserve"}
    expected = {(world, surface) for world in worlds for surface in surfaces}
    rows: list[dict[str, Any]] = []
    for context, by_world_surface in sorted(grouped.items()):
        complete_crossing = set(by_world_surface) == expected
        estimable = complete_crossing and all(
            isinstance(
                by_world_surface[key].get(
                    "mean_conditional_transfer_probability"
                ),
                (int, float),
            )
            for key in expected
        )
        differences = None
        average = None
        if estimable:
            differences = {
                surface: float(
                    by_world_surface[("regional_mutual_aid", surface)][
                        "mean_conditional_transfer_probability"
                    ]
                )
                - float(
                    by_world_surface[("local_emergency_reserve", surface)][
                        "mean_conditional_transfer_probability"
                    ]
                )
                for surface in sorted(surfaces)
            }
            average = mean(differences.values())
        rows.append(
            {
                "context_condition": context,
                "complete_world_surface_crossing": complete_crossing,
                "regional_minus_local_by_surface": differences,
                "mean_regional_minus_local_over_surfaces": average,
            }
        )
    return rows


def _output_logprob_positions(
    raw_response: Mapping[str, Any], output_text: str
) -> tuple[Mapping[str, Any], ...]:
    output = raw_response.get("output")
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise BranchProbabilityError("provider response output must be a list")
    matching: list[tuple[Mapping[str, Any], ...]] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "output_text":
                continue
            if part.get("text") != output_text:
                continue
            logprobs = part.get("logprobs")
            if not isinstance(logprobs, Sequence) or isinstance(
                logprobs, (str, bytes)
            ):
                raise BranchProbabilityError("matching output text has no logprobs")
            if any(not isinstance(position, Mapping) for position in logprobs):
                raise BranchProbabilityError("output logprob positions must be objects")
            matching.append(tuple(logprobs))
    if len(matching) != 1 or not matching[0]:
        raise BranchProbabilityError(
            "provider response must contain one matching output-text logprob sequence"
        )
    return matching[0]


def _output_text(raw_response: Mapping[str, Any]) -> str:
    texts: list[str] = []
    output = raw_response.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            for part in content:
                if (
                    isinstance(part, Mapping)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    texts.append(part["text"])
    if len(texts) != 1:
        raise BranchProbabilityError("provider response has ambiguous output text")
    return texts[0]


def _returned_candidates(
    selected: Mapping[str, Any], position_index: int
) -> tuple[Mapping[str, Any], ...]:
    top = selected.get("top_logprobs")
    if not isinstance(top, Sequence) or isinstance(top, (str, bytes)):
        raise BranchProbabilityError(
            f"logprobs[{position_index}].top_logprobs must be a list"
        )
    values = [candidate for candidate in top if isinstance(candidate, Mapping)]
    if len(values) != len(top):
        raise BranchProbabilityError("top-logprob candidates must be objects")
    chosen = {
        "token": selected.get("token"),
        "bytes": selected.get("bytes"),
        "logprob": selected.get("logprob"),
    }
    chosen_key = _candidate_key(chosen, "sampled candidate")
    by_key: dict[tuple[int, ...], Mapping[str, Any]] = {}
    for candidate in (*values, chosen):
        key = _candidate_key(candidate, "candidate")
        if key in by_key:
            previous = _probability_from_logprob(
                by_key[key].get("logprob"), "candidate.logprob"
            )
            current = _probability_from_logprob(
                candidate.get("logprob"), "candidate.logprob"
            )
            if current > previous:
                by_key[key] = candidate
        else:
            by_key[key] = candidate
    if chosen_key not in by_key:
        raise AssertionError("sampled candidate was not preserved")
    return tuple(by_key[key] for key in sorted(by_key))


def _candidate_key(candidate: Mapping[str, Any], label: str) -> tuple[int, ...]:
    raw = candidate.get("bytes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise BranchProbabilityError(f"{label}.bytes must be a byte list")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item <= 255
        for item in raw
    ):
        raise BranchProbabilityError(f"{label}.bytes contains an invalid byte")
    return tuple(raw)


def _candidate_text(candidate: Mapping[str, Any], label: str) -> str:
    raw = _candidate_bytes(candidate, label)
    return raw.decode("utf-8", errors="replace")


def _candidate_bytes(candidate: Mapping[str, Any], label: str) -> bytes:
    return bytes(_candidate_key(candidate, label))


def _compatible_action_bytes(token_bytes: bytes) -> str | None:
    candidate = token_bytes.lstrip(b" \t\r\n\f\v")
    if not candidate:
        return None
    matches = []
    for label in (TRANSFER_LABEL, RETAIN_LABEL):
        encoded = label.encode("ascii")
        if encoded.startswith(candidate) or (
            candidate.startswith(encoded)
            and candidate[len(encoded) :].strip(b" \t\r\n\f\v") == b""
        ):
            matches.append(label)
    return matches[0] if len(matches) == 1 else None


def _probability_from_logprob(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BranchProbabilityError(f"{label} must be numeric")
    result = float(value)
    if math.isnan(result) or result == -math.inf:
        return 0.0
    if result >= 0.0 or result == math.inf:
        return 1.0
    return math.exp(result)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BranchProbabilityError(f"{label} must be nonempty trimmed text")
    return value
