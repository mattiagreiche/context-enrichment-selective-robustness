from __future__ import annotations

import math

import pytest

from selective_robustness.probabilities import (
    BranchProbabilityError,
    extract_first_branch_probability,
    summarize_first_branch_probabilities,
)


def _response(
    *,
    text: str,
    positions: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "id": "resp-test",
        "output_text": text,
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "logprobs": positions,
                    }
                ],
            }
        ],
    }


def _candidate(text: str, probability: float) -> dict[str, object]:
    return {
        "token": text,
        "bytes": list(text.encode("utf-8")),
        "logprob": math.log(probability),
    }


def test_first_action_token_sums_compatible_tokenizations() -> None:
    chosen = _candidate("TRANSFER", 0.45)
    response = _response(
        text="TRANSFER",
        positions=[
            {
                **chosen,
                "top_logprobs": [
                    chosen,
                    _candidate("T", 0.15),
                    _candidate("RE", 0.30),
                    _candidate("I", 0.10),
                ],
            }
        ],
    )

    result = extract_first_branch_probability(response)

    assert result.status == "complete"
    assert result.sampled_action == "TRANSFER"
    assert result.completion_matches_sampled_branch is True
    assert result.transfer_reported_mass == pytest.approx(0.60)
    assert result.retain_reported_mass == pytest.approx(0.30)
    assert result.other_reported_mass == pytest.approx(0.10)
    assert result.conditional_transfer_probability == pytest.approx(2 / 3)
    assert result.unreturned_probability_mass == pytest.approx(0)


def test_leading_whitespace_is_skipped_and_probability_is_conditional_on_it() -> None:
    whitespace = _candidate("\n", 0.9)
    chosen = _candidate("RE", 0.75)
    response = _response(
        text="\nRETAIN",
        positions=[
            {**whitespace, "top_logprobs": [whitespace, _candidate(" ", 0.1)]},
            {
                **chosen,
                "top_logprobs": [chosen, _candidate("TRANSFER", 0.25)],
            },
        ],
    )

    result = extract_first_branch_probability(response)

    assert result.first_action_token_index == 1
    assert result.sampled_action == "RETAIN"
    assert result.completion_matches_sampled_branch is True
    assert result.conditional_transfer_probability == pytest.approx(0.25)


def test_missing_top_twenty_branch_is_censored_not_zero() -> None:
    chosen = _candidate("TRANSFER", 0.99)
    response = _response(
        text="TRANSFER",
        positions=[
            {**chosen, "top_logprobs": [chosen, _candidate("I", 0.01)]}
        ],
    )

    result = extract_first_branch_probability(response)

    assert result.status == "censored_missing_retain_branch"
    assert result.conditional_transfer_probability is None


def test_rounded_candidate_mass_above_one_is_rescaled_not_rejected() -> None:
    chosen = _candidate("TRANSFER", 0.60001)
    response = _response(
        text="TRANSFER",
        positions=[
            {
                **chosen,
                "top_logprobs": [chosen, _candidate("RE", 0.40001)],
            }
        ],
    )

    result = extract_first_branch_probability(response)

    assert result.status == "complete"
    assert result.returned_candidate_mass == 1.0
    assert result.unreturned_probability_mass == 0.0
    assert result.transfer_reported_mass + result.retain_reported_mass == (
        pytest.approx(1.0)
    )
    assert result.conditional_transfer_probability == pytest.approx(
        0.60001 / 1.00002
    )


def test_nonstandard_numeric_logprobs_are_clamped_not_rejected() -> None:
    chosen = {
        "token": "TRANSFER",
        "bytes": list(b"TRANSFER"),
        "logprob": 1e-6,
    }
    response = _response(
        text="TRANSFER",
        positions=[
            {
                **chosen,
                "top_logprobs": [
                    chosen,
                    {
                        "token": "RE",
                        "bytes": list(b"RE"),
                        "logprob": -math.inf,
                    },
                    {
                        "token": "I",
                        "bytes": list(b"I"),
                        "logprob": math.nan,
                    },
                ],
            }
        ],
    )

    result = extract_first_branch_probability(response)

    assert result.status == "censored_missing_retain_branch"
    assert result.transfer_reported_mass == 1.0
    assert result.retain_reported_mass == 0.0
    assert result.returned_candidate_mass == 1.0


def test_duplicate_candidate_rounding_uses_larger_probability() -> None:
    chosen = _candidate("TRANSFER", 0.6)
    duplicate = _candidate("TRANSFER", 0.61)
    response = _response(
        text="TRANSFER",
        positions=[
            {
                **chosen,
                "top_logprobs": [duplicate, _candidate("RE", 0.39)],
            }
        ],
    )

    result = extract_first_branch_probability(response)

    assert result.status == "complete"
    assert result.returned_candidate_count == 2
    assert result.transfer_reported_mass == pytest.approx(0.61)
    assert result.retain_reported_mass == pytest.approx(0.39)
    assert result.conditional_transfer_probability == pytest.approx(0.61)


def test_malformed_or_ambiguous_logprobs_fail_closed() -> None:
    with pytest.raises(BranchProbabilityError, match="no logprobs"):
        extract_first_branch_probability(
            {
                "output_text": "TRANSFER",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "TRANSFER"}
                        ],
                    }
                ],
            }
        )


def test_cell_summary_requires_exact_schedule_and_all_complete_observations() -> None:
    base = {
        "condition_id": "hospital/regional_mutual_aid/minimal/canonical_prose",
        "social_world": "regional_mutual_aid",
        "context_condition": "minimal",
        "surface_version": "canonical_prose",
        "prompt_sha256": "a" * 64,
        "status": "complete",
        "completion_matches_sampled_branch": True,
        "transfer_reported_mass": 0.5,
        "retain_reported_mass": 0.4,
        "unreturned_probability_mass": 0.1,
    }
    observations = [
        {
            **base,
            "call_id": "call-1",
            "replicate_id": "replicate-0001",
            "conditional_transfer_probability": 0.4,
        },
        {
            **base,
            "call_id": "call-2",
            "replicate_id": "replicate-0002",
            "conditional_transfer_probability": 0.6,
        },
    ]
    schedule = [
        {
            "call_id": row["call_id"],
            "replicate_id": row["replicate_id"],
            "condition_id": row["condition_id"],
        }
        for row in observations
    ]

    summary = summarize_first_branch_probabilities(
        observations, scheduled_calls=schedule
    )

    assert summary["observation_count"] == 2
    assert summary["cells"][0]["mean_conditional_transfer_probability"] == 0.5
    assert summary["cells"][0][
        "sample_sd_conditional_transfer_probability"
    ] == pytest.approx(math.sqrt(0.02))

    with pytest.raises(BranchProbabilityError, match="exactly cover"):
        summarize_first_branch_probabilities(
            observations[:1], scheduled_calls=schedule
        )
