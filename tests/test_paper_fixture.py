"""The released fixture must be the exact prompt instrument used by the runner."""

from selective_robustness.catalog import hospital_cells
from selective_robustness.live_runner import verify_live_governance
from selective_robustness.prompts import render_prompt
from selective_robustness.validation import text_sha256


def test_frozen_paper_prompts_match_the_renderer_and_run_authorization() -> None:
    fixture, authorization = verify_live_governance(
        "pilot_v4", "first_action_branch_v1"
    )
    cells = hospital_cells("pilot_v4")

    assert len(cells) == 40
    assert len(fixture.prompts) == 40
    assert len(authorization.condition_ids) == 40
    assert authorization.calls_per_template == 5
    assert authorization.maximum_total_calls == 200

    for cell in cells:
        frozen = fixture.prompt_for(cell.condition_id)
        rendered = render_prompt(cell).text
        assert frozen.text == rendered
        assert frozen.prompt_sha256 == text_sha256(rendered)
