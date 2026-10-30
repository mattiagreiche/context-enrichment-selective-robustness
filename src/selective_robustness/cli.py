"""Commands for the published hospital probability experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import sys
from pathlib import Path
from typing import Any

from .catalog import context_conditions_for, hospital_cells, validate_hospital_catalog
from .domain import SocialWorld, SurfaceVersion
from .sampling import ReasoningEffort
from .validation import (
    validate_and_audit_hospital_prompts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selective-robustness")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the frozen pilot_v4 prompts")
    probability_plan = commands.add_parser(
        "probability-plan",
        help="show the reviewed full-family first-branch probability protocol",
    )
    probability_plan.add_argument("--json", action="store_true", dest="as_json")
    smoke = commands.add_parser(
        "smoke", help="run one scripted native SiliSocS episode"
    )
    smoke.add_argument("--output", type=Path, default=Path("results/silisocs-smoke"))
    pilot_prepare = commands.add_parser(
        "pilot-prepare", help="write an immutable reviewed-pilot run manifest"
    )
    pilot_prepare.add_argument("--model", required=True)
    pilot_prepare.add_argument("--prompt-version", default="pilot_v4", choices=["pilot_v4"])
    pilot_prepare.add_argument("--temperature", required=True, type=float)
    pilot_prepare.add_argument(
        "--reasoning-effort",
        required=True,
        choices=[effort.value for effort in ReasoningEffort],
    )
    pilot_prepare.add_argument("--top-p", type=float)
    pilot_prepare.add_argument(
        "--measurement-protocol",
        default="first_action_branch_v1",
        choices=["first_action_branch_v1"],
    )
    pilot_prepare.add_argument("--top-logprobs", type=int)
    pilot_prepare.add_argument("--max-output-tokens", required=True, type=int)
    pilot_prepare.add_argument("--timeout-seconds", required=True, type=float)
    pilot_prepare.add_argument("--calls-per-template", required=True, type=int)
    pilot_prepare.add_argument("--maximum-total-calls", required=True, type=int)
    pilot_prepare.add_argument("--randomization-seed", required=True, type=int)
    pilot_prepare.add_argument("--output", required=True, type=Path)
    pilot_prepare.add_argument("--allow-paid", action="store_true")
    pilot_prepare.add_argument("--acknowledge-paid-pilot", default="")
    pilot_prepare.add_argument("--acknowledge-reviewed-prompt", action="store_true")
    pilot_run = commands.add_parser(
        "pilot-run", help="execute or safely resume an immutable pilot schedule"
    )
    pilot_run.add_argument("--manifest", required=True, type=Path)
    pilot_run.add_argument("--allow-paid", action="store_true")
    pilot_run.add_argument("--acknowledge-paid-pilot", default="")
    pilot_run.add_argument(
        "--through-replicate",
        type=int,
        help="stop after this complete replicate block; resume uses the same manifest",
    )
    probability_summarize = commands.add_parser(
        "probability-summarize",
        help="validate and summarize first action-branch probabilities",
    )
    probability_summarize.add_argument("--manifest", required=True, type=Path)
    probability_summarize.add_argument("--events", type=Path)
    probability_summarize.add_argument("--output", type=Path)
    return parser


def validate() -> None:
    from .live_runner import verify_live_governance

    cells = hospital_cells("pilot_v4")
    validate_hospital_catalog(cells)
    validate_and_audit_hospital_prompts(cells)
    verify_live_governance("pilot_v4", "first_action_branch_v1")


def _required_manifest_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"pilot manifest {name} must be non-empty trimmed text")
    return value


def _load_scheduled_decisions(
    *,
    manifest_path: Path,
    schedule: tuple[dict[str, Any], ...],
    aggregate_events: Path | None,
    study_stage: str,
    requested_model_identifier: str,
    prompt_version: str,
    manifest_sha256: str,
) -> tuple[Any, ...]:
    from .results import load_focal_decisions

    defaults = {
        "study_stage": study_stage,
        "requested_model_identifier": requested_model_identifier,
        "prompt_version": prompt_version,
    }
    if aggregate_events is not None:
        decisions = load_focal_decisions(aggregate_events, **defaults)
        _require_decision_manifest_binding(decisions, manifest_sha256)
        return decisions

    decisions: list[Any] = []
    calls_root = manifest_path.parent / "calls"
    for row in schedule:
        call_id = row["call_id"]
        event_path = calls_root / call_id / "action_events.jsonl"
        if not event_path.is_file():
            raise ValueError(
                f"scheduled call {call_id!r} is missing action_events.jsonl"
            )
        call_decisions = load_focal_decisions(event_path, **defaults)
        if len(call_decisions) != 1 or call_decisions[0].call_id != call_id:
            raise ValueError(
                f"scheduled call {call_id!r} must have exactly one matching focal event"
            )
        _require_decision_manifest_binding(call_decisions, manifest_sha256)
        decisions.extend(call_decisions)
    return tuple(decisions)


def _require_decision_manifest_binding(
    decisions: tuple[Any, ...], manifest_sha256: str
) -> None:
    for decision in decisions:
        metadata = dict(decision.provider_metadata)
        if metadata.get("run_manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"focal event {decision.call_id!r} is not bound to the run manifest"
            )


def _probability_observations(decisions: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    from .probabilities import (
        extract_first_branch_probability,
        observation_dict,
        thaw_json,
    )

    observations: list[dict[str, Any]] = []
    for decision in decisions:
        metadata = thaw_json(decision.provider_metadata)
        if not isinstance(metadata, dict):
            raise ValueError("focal provider metadata is not an object")
        raw_response = metadata.get("raw_provider_response")
        if not isinstance(raw_response, dict):
            raise ValueError("focal provider metadata has no raw provider response")
        extracted = extract_first_branch_probability(
            raw_response, output_text=decision.raw_response
        )
        stored = metadata.get("first_action_branch_probability")
        if stored != asdict(extracted):
            raise ValueError(
                f"stored first-branch probability disagrees for {decision.call_id!r}"
            )
        observations.append(
            observation_dict(
                call_id=decision.call_id,
                replicate_id=decision.replicate_id,
                condition_id=decision.condition_id,
                social_world=decision.social_world.value,
                context_condition=decision.context_condition.value,
                surface_version=decision.surface_version.value,
                prompt_sha256=decision.prompt_sha256,
                probability=extracted,
            )
        )
    return tuple(observations)


def _validate_completed_pilot_state(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    manifest_payload: dict[str, Any],
    schedule: tuple[dict[str, Any], ...],
    decisions: tuple[Any, ...],
) -> dict[str, str]:
    from .pilot_workflow import RUN_STATE_LEDGER_NAME, _call_state, _read_run_state

    ledger_path = manifest_path.parent / RUN_STATE_LEDGER_NAME
    ledger_bytes = ledger_path.read_bytes()
    ledger = _read_run_state(ledger_path, manifest_sha256)
    state = _call_state(ledger, list(schedule))
    if not state or any(status != "succeeded" for status in state.values()):
        raise ValueError(
            "pilot run-state does not contain one successful terminal event "
            "for every scheduled call"
        )
    completion_events = [
        event for event in ledger if event.get("event") == "pilot_run_completed"
    ]
    if len(completion_events) != 1 or ledger[-1] is not completion_events[0]:
        raise ValueError(
            "pilot run-state must end with exactly one pilot_run_completed record"
        )

    expected_by_call = {row["call_id"]: row for row in schedule}
    decisions_by_call = {decision.call_id: decision for decision in decisions}
    success_events = [
        event for event in ledger if event.get("event") == "pilot_call_succeeded"
    ]
    if len(success_events) != len(schedule):
        raise ValueError("pilot run-state success count differs from the schedule")
    returned_models: set[str] = set()
    requested_model = manifest_payload.get("requested_model_identifier")
    run_id = manifest_payload.get("run_id")
    for event in success_events:
        call_id = event.get("call_id")
        row = expected_by_call.get(call_id)
        decision = decisions_by_call.get(call_id)
        if row is None or decision is None:
            raise ValueError("pilot run-state contains a substituted call result")
        returned_model = event.get("returned_model_identifier")
        expected_artifact = str((manifest_path.parent / "calls" / call_id).resolve())
        if (
            event.get("status") != "success"
            or event.get("run_id") != run_id
            or event.get("condition_id") != row["condition_id"]
            or event.get("replicate_id") != row["replicate_id"]
            or event.get("requested_model_identifier") != requested_model
            or returned_model != decision.returned_model_identifier
            or event.get("prompt_sha256") != row["expected_prompt_sha256"]
            or event.get("call_artifact_path") != expected_artifact
        ):
            raise ValueError(
                f"pilot run-state provenance disagrees for scheduled call {call_id!r}"
            )
        if not isinstance(returned_model, str) or not returned_model.strip():
            raise ValueError("pilot run-state is missing a returned model identifier")
        returned_models.add(returned_model)
    completion = completion_events[0]
    if (
        completion.get("status") != "success"
        or completion.get("run_id") != run_id
        or completion.get("scheduled_calls") != len(schedule)
        or completion.get("returned_model_status") != "observed_consistent"
        or completion.get("returned_model_identifier") not in returned_models
        or len(returned_models) != 1
    ):
        raise ValueError("pilot_run_completed provenance is inconsistent")
    if ledger_path.read_bytes() != ledger_bytes:
        raise ValueError("pilot run-state changed during summary validation")
    return {
        "path": str(ledger_path.resolve()),
        "sha256": sha256(ledger_bytes).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            validate()
            print("valid: frozen 40-cell pilot_v4 probability experiment")
            return 0
        if arguments.command == "probability-plan":
            from .live_runner import verify_live_governance

            _, authorization = verify_live_governance(
                "pilot_v4", "first_action_branch_v1"
            )
            summary = {
                "scenario": "hospital",
                "prompt_version": "pilot_v4",
                "measurement_protocol": authorization.measurement_protocol,
                "social_worlds": len(SocialWorld),
                "context_conditions": len(context_conditions_for("pilot_v4")),
                "surface_versions": len(SurfaceVersion),
                "prompt_templates": len(authorization.condition_ids),
                "calls_per_template": authorization.calls_per_template,
                "maximum_total_calls_per_model": authorization.maximum_total_calls,
                "maximum_total_calls": (
                    authorization.maximum_total_calls
                    * len(authorization.requested_model_identifiers)
                ),
                "requested_model_identifiers": list(
                    authorization.requested_model_identifiers
                ),
                "sampling_spec": authorization.sampling.as_dict(),
                "condition_ids": list(authorization.condition_ids),
            }
            if arguments.as_json:
                print(json.dumps(summary, indent=2))
            else:
                print(
                    "40 frozen pilot_v4 templates; 5 independent calls per "
                    "template; 200 calls per model and 400 Sol-plus-Terra calls"
                )
            return 0
        if arguments.command == "smoke":
            from .silisocs_runner import run_scripted_smoke

            validate()
            run_scripted_smoke(hospital_cells("pilot_v4")[0], arguments.output)
            print(f"SiliSocS scripted smoke passed: {arguments.output}")
            return 0
        if arguments.command == "pilot-prepare":
            from .pilot_workflow import PilotRunRequest, prepare_pilot_run
            from .sampling import LogprobMode, SamplingSpec, TopPMode

            artifact = prepare_pilot_run(
                PilotRunRequest(
                    prompt_version=arguments.prompt_version,
                    requested_model_identifier=arguments.model,
                    sampling=SamplingSpec(
                        temperature=arguments.temperature,
                        reasoning_effort=ReasoningEffort(arguments.reasoning_effort),
                        top_p_mode=(
                            TopPMode.EXPLICIT
                            if arguments.top_p is not None
                            else TopPMode.OMITTED_PROVIDER_DEFAULT
                        ),
                        top_p=arguments.top_p,
                        max_output_tokens=arguments.max_output_tokens,
                        timeout_seconds=arguments.timeout_seconds,
                        logprob_mode=(
                            LogprobMode.FIRST_ACTION_BRANCH
                            if arguments.measurement_protocol
                            == "first_action_branch_v1"
                            else LogprobMode.OMITTED
                        ),
                        top_logprobs=arguments.top_logprobs,
                    ),
                    calls_per_template=arguments.calls_per_template,
                    maximum_total_calls=arguments.maximum_total_calls,
                    randomization_seed=arguments.randomization_seed,
                    output=arguments.output,
                    allow_paid=arguments.allow_paid,
                    paid_pilot_acknowledgement=arguments.acknowledge_paid_pilot,
                    reviewed_prompt_acknowledgement=(
                        arguments.acknowledge_reviewed_prompt
                    ),
                    measurement_protocol=arguments.measurement_protocol,
                )
            )
            print(f"wrote immutable pilot manifest: {artifact.path}")
            return 0
        if arguments.command == "pilot-run":
            from .pilot_workflow import execute_pilot_run

            completed = execute_pilot_run(
                arguments.manifest,
                allow_paid=arguments.allow_paid,
                paid_pilot_acknowledgement=arguments.acknowledge_paid_pilot,
                through_replicate=arguments.through_replicate,
            )
            print(f"completed {len(completed)} newly scheduled pilot calls")
            return 0
        if arguments.command == "probability-summarize":
            from .pilot_workflow import (
                load_pilot_manifest,
                scheduled_calls_from_manifest,
            )
            from .probabilities import summarize_first_branch_probabilities

            manifest = load_pilot_manifest(arguments.manifest)
            if (
                manifest.payload.get("measurement_protocol")
                != "first_action_branch_v1"
            ):
                raise ValueError(
                    "probability-summarize requires a first_action_branch_v1 manifest"
                )
            schedule = scheduled_calls_from_manifest(manifest.path)
            study_stage = _required_manifest_text(manifest.payload, "study_stage")
            requested_model = _required_manifest_text(
                manifest.payload, "requested_model_identifier"
            )
            prompt_version = _required_manifest_text(
                manifest.payload, "prompt_version"
            )
            decisions = _load_scheduled_decisions(
                manifest_path=manifest.path,
                schedule=schedule,
                aggregate_events=arguments.events,
                study_stage=study_stage,
                requested_model_identifier=requested_model,
                prompt_version=prompt_version,
                manifest_sha256=manifest.sha256,
            )
            run_state_source = _validate_completed_pilot_state(
                manifest_path=manifest.path,
                manifest_sha256=manifest.sha256,
                manifest_payload=manifest.payload,
                schedule=schedule,
                decisions=decisions,
            )
            observations = _probability_observations(decisions)
            summary = summarize_first_branch_probabilities(
                observations, scheduled_calls=schedule
            )
            payload = {
                "schema_version": 1,
                "analysis": "hospital_first_action_branch_probabilities",
                "study_stage": study_stage,
                "measurement_protocol": "first_action_branch_v1",
                "requested_model_identifier": requested_model,
                "returned_model_identifiers": sorted(
                    {decision.returned_model_identifier for decision in decisions}
                ),
                "prompt_version": prompt_version,
                "scheduled_call_count": len(schedule),
                "run_manifest": {
                    "path": str(manifest.path),
                    "sha256": manifest.sha256,
                },
                "run_state": run_state_source,
                "observations": list(observations),
                "summary": summary,
            }
            text = json.dumps(payload, indent=2) + "\n"
            if arguments.output is None:
                print(text, end="")
            else:
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                arguments.output.write_text(text, encoding="utf-8")
                print(f"wrote probability summary to {arguments.output}")
            return 0
        raise AssertionError("unknown command")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
