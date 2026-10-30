"""Check the paper's SiliSocS event path without credentials."""

from selective_robustness.catalog import hospital_cells
from selective_robustness.results import load_focal_decisions
from selective_robustness.silisocs_runner import run_scripted_smoke
from selective_robustness.validation import text_sha256


def test_scripted_silisocs_event_round_trip(tmp_path) -> None:
    cell = hospital_cells()[0]
    run_scripted_smoke(cell, tmp_path)
    decisions = load_focal_decisions(
        tmp_path / "action_events.jsonl",
        study_stage="development",
        requested_model_identifier="scripted-hospital-pilot",
        prompt_version="pilot_v4",
    )
    assert len(decisions) == 1
    assert decisions[0].condition_id == cell.condition_id
    assert decisions[0].prompt_sha256 == text_sha256(decisions[0].raw_prompt)
    assert decisions[0].raw_response == "TRANSFER"
