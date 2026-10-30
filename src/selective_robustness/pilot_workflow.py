"""Immutable, resume-safe outer workflow for an authorized hospital pilot subset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from functools import wraps
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import uuid4, uuid5, NAMESPACE_URL

from .domain import ContextCondition, PromptCell, SocialWorld, SurfaceVersion
from .live_runner import (
    PAID_RUN_ACKNOWLEDGEMENT,
    LiveRunRequest,
    LiveRunResult,
    _validate_live_output_path,
    file_sha256,
    repository_root,
    run_live_once,
    verify_live_governance,
)
from .sampling import SamplingSpec


PAID_PILOT_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_SCHEDULED_PAID_PILOT"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_MANIFEST_HASH_NAME = "run_manifest.sha256"
RUN_STATE_LEDGER_NAME = "run_state.jsonl"
PROVIDER_ATTEMPT_LEDGER_NAME = "prompts_and_responses.jsonl"
PINNED_SILISOCS_PATTERN = re.compile(r"silisocs[^\n]*@([0-9a-f]{40})")


@dataclass(frozen=True, slots=True)
class PilotRunRequest:
    prompt_version: str
    requested_model_identifier: str
    sampling: SamplingSpec
    calls_per_template: int
    maximum_total_calls: int
    randomization_seed: int
    output: Path
    allow_paid: bool
    paid_pilot_acknowledgement: str
    reviewed_prompt_acknowledgement: bool
    study_stage: str = "development"
    measurement_protocol: str = "first_action_branch_v1"

    def validate(self) -> None:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.prompt_version)
            or ".." in self.prompt_version
        ):
            raise ValueError("prompt_version contains unsafe characters")
        if self.prompt_version != "pilot_v4":
            raise ValueError("unsupported prompt version")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}",
            self.requested_model_identifier,
        ):
            raise ValueError("model identifier contains unsafe characters")
        if not isinstance(self.sampling, SamplingSpec):
            raise TypeError("sampling must be a canonical SamplingSpec")
        if isinstance(self.calls_per_template, bool) or not isinstance(
            self.calls_per_template, int
        ):
            raise ValueError("calls_per_template must be a positive integer")
        if self.calls_per_template <= 0:
            raise ValueError("calls_per_template must be a positive integer")
        if (
            isinstance(self.maximum_total_calls, bool)
            or not isinstance(self.maximum_total_calls, int)
            or self.maximum_total_calls <= 0
        ):
            raise ValueError(
                "maximum_total_calls must be a positive integer; the authorized "
                "condition subset determines the exact budget"
            )
        if isinstance(self.randomization_seed, bool) or not isinstance(
            self.randomization_seed, int
        ):
            raise ValueError("randomization_seed must be an integer")
        if self.study_stage != "development":
            raise ValueError("the pilot workflow is development-only")
        if self.measurement_protocol != "first_action_branch_v1":
            raise ValueError("measurement_protocol is invalid")
        if not self.reviewed_prompt_acknowledgement:
            raise ValueError(
                "pilot preparation requires reviewed-prompt acknowledgement"
            )
        if not self.allow_paid:
            raise ValueError("pilot preparation requires --allow-paid")
        if self.paid_pilot_acknowledgement != PAID_PILOT_ACKNOWLEDGEMENT:
            raise ValueError(
                "paid-pilot acknowledgement does not match the required phrase"
            )


@dataclass(frozen=True, slots=True)
class PilotManifestArtifact:
    path: Path
    sha256: str
    payload: dict[str, Any]


def _exclusive_run_lock(function: Any) -> Any:
    """Serialize pilot execution processes for one immutable run directory."""

    @wraps(function)
    def wrapped(manifest_path: Path, *args: Any, **kwargs: Any) -> Any:
        artifact = load_pilot_manifest(manifest_path)
        lock_path = artifact.path.parent / ".pilot-run.lock"
        with lock_path.open("a", encoding="utf-8") as lock_stream:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another pilot-run process holds the run lock"
                ) from exc
            try:
                return function(artifact.path, *args, **kwargs)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    return wrapped


def prepare_pilot_run(request: PilotRunRequest) -> PilotManifestArtifact:
    """Write, once, the complete pre-dispatch manifest and its hash sidecar."""

    request.validate()
    fixture, authorization = verify_live_governance(
        request.prompt_version, request.measurement_protocol
    )
    _verify_authorized_pilot_request(request, authorization)
    output = _validate_live_output_path(request.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("pilot output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    run_id = f"hospital-pilot-{uuid4()}"
    authorized_condition_ids = set(authorization.condition_ids)
    prompt_rows = sorted(
        (
            item
            for item in fixture.prompts
            if item.cell.condition_id in authorized_condition_ids
        ),
        key=lambda item: item.cell.condition_id,
    )
    if {item.cell.condition_id for item in prompt_rows} != authorized_condition_ids:
        raise ValueError("authorized condition subset is absent from the fixture")
    condition_rows = [
        {
            "condition_id": item.cell.condition_id,
            "social_world": item.cell.world.value,
            "context_condition": item.cell.context.value,
            "surface_version": item.cell.surface.value,
            "prompt_version": item.cell.prompt_version,
            "expected_prompt_sha256": item.prompt_sha256,
        }
        for item in prompt_rows
    ]
    schedule = _build_schedule(
        run_id=run_id,
        conditions=condition_rows,
        calls_per_template=request.calls_per_template,
        seed=request.randomization_seed,
    )
    project = _project_provenance(repository_root())
    if project["project_git_dirty"]:
        raise ValueError(
            "pilot preparation requires a clean worktree so the revision binds all inputs"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "hospital_pilot_run_manifest",
        "manifest_status": "immutable_pre_run",
        "run_id": run_id,
        "created_at_utc": _utc_timestamp(),
        "study_stage": request.study_stage,
        "measurement_protocol": request.measurement_protocol,
        "prompt_version": request.prompt_version,
        "requested_model_identifier": request.requested_model_identifier,
        "returned_model_identifier": None,
        "returned_model_status": "pending",
        "sampling_spec": request.sampling.as_dict(),
        "calls_per_template": request.calls_per_template,
        "maximum_total_calls": request.maximum_total_calls,
        "condition_count": len(condition_rows),
        "conditions": condition_rows,
        "randomization": {
            "seed": request.randomization_seed,
            "algorithm": "sha256_seeded_sort_v1",
            "blocking": (
                f"one_complete_{len(condition_rows)}_cell_shuffle_per_replicate"
            ),
        },
        "schedule": schedule,
        "provenance": {
            **project,
            "fixture_manifest_path": str(fixture.manifest_path.resolve()),
            "fixture_manifest_sha256": fixture.manifest_sha256,
            "study_authorization_path": str(authorization.path.resolve()),
            "study_authorization_sha256": authorization.artifact_sha256,
        },
        "governance": {
            "machine_record_only_not_proof_of_human_review": True,
            "review_status": "declared_complete",
            "authorization_status": "authorized_for_paid_development",
            "paid_calls_authorized": authorization.paid_calls_authorized,
            "reviewed_prompt_acknowledged": request.reviewed_prompt_acknowledgement,
            "operator_paid_pilot_acknowledged": True,
        },
        "retry_policy": {
            "transport_retries": 0,
            "content_retries": 0,
            "automatic_replacement_calls": False,
        },
    }
    manifest_path = output / RUN_MANIFEST_NAME
    manifest_bytes = (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    manifest_hash = sha256(manifest_bytes).hexdigest()
    _write_exclusive_durable(manifest_path, manifest_bytes)
    _write_exclusive_durable(
        output / RUN_MANIFEST_HASH_NAME,
        f"{manifest_hash}  {RUN_MANIFEST_NAME}\n".encode("ascii"),
    )
    return PilotManifestArtifact(
        path=manifest_path, sha256=manifest_hash, payload=payload
    )


@_exclusive_run_lock
def execute_pilot_run(
    manifest_path: Path,
    *,
    allow_paid: bool,
    paid_pilot_acknowledgement: str,
    through_replicate: int | None = None,
) -> tuple[LiveRunResult, ...]:
    """Execute never-started calls through a replicate checkpoint, without retrying."""

    if not allow_paid or paid_pilot_acknowledgement != PAID_PILOT_ACKNOWLEDGEMENT:
        raise ValueError(
            "pilot execution requires the exact paid-pilot acknowledgement"
        )
    artifact = load_pilot_manifest(manifest_path)
    payload = artifact.payload
    prompt_version = _required_text(payload.get("prompt_version"), "prompt_version")
    measurement_protocol = _required_text(
        payload.get("measurement_protocol", "first_action_branch_v1"),
        "measurement_protocol",
    )
    fixture, authorization = verify_live_governance(
        prompt_version, measurement_protocol
    )
    provenance = _required_mapping(payload.get("provenance"), "provenance")
    if (
        provenance.get("fixture_manifest_sha256") != fixture.manifest_sha256
        or provenance.get("study_authorization_sha256") != authorization.artifact_sha256
    ):
        raise ValueError(
            "current fixture/governance artifacts differ from the run manifest"
        )
    current_project = _project_provenance(repository_root())
    for field in (
        "project_git_revision",
        "project_git_dirty",
        "uv_lock_sha256",
        "pinned_silisocs_revision",
        "installed_silisocs_revision",
        "installed_silisocs_version",
    ):
        if provenance.get(field) != current_project.get(field):
            raise ValueError(f"execution-time {field} differs from the run manifest")
    if current_project["project_git_dirty"]:
        raise ValueError(
            "pilot execution requires the manifest's clean project revision"
        )
    sampling = SamplingSpec.from_mapping(
        _required_mapping(payload.get("sampling_spec"), "sampling_spec")
    )
    requested_model = _required_text(
        payload.get("requested_model_identifier"), "requested_model_identifier"
    )
    if (
        requested_model not in authorization.requested_model_identifiers
        or sampling != authorization.sampling
        or payload.get("calls_per_template") != authorization.calls_per_template
        or payload.get("maximum_total_calls") != authorization.maximum_total_calls
        or set(_manifest_condition_ids(payload)) != set(authorization.condition_ids)
        or measurement_protocol != authorization.measurement_protocol
    ):
        raise ValueError("run manifest differs from the authorized study design")
    schedule = _validate_manifest_schedule(payload, fixture)
    calls_per_template = int(payload["calls_per_template"])
    if through_replicate is None:
        replicate_limit = calls_per_template
    elif (
        isinstance(through_replicate, bool)
        or not isinstance(through_replicate, int)
        or not 1 <= through_replicate <= calls_per_template
    ):
        raise ValueError(
            "through_replicate must be between 1 and calls_per_template"
        )
    else:
        replicate_limit = through_replicate
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for an acknowledged pilot run")
    ledger_path = artifact.path.parent / RUN_STATE_LEDGER_NAME
    ledger = _read_run_state(ledger_path, artifact.sha256)
    state = _call_state(ledger, schedule)
    returned_models = {
        item["returned_model_identifier"]
        for item in ledger
        if item.get("event") == "pilot_call_succeeded"
        and isinstance(item.get("returned_model_identifier"), str)
    }
    if any(
        item.get("requested_model_identifier") != requested_model
        for item in ledger
        if item.get("event") == "pilot_call_succeeded"
    ):
        raise RuntimeError("run-state requested model differs from the run manifest")
    if len(returned_models) > 1:
        raise RuntimeError("returned model identifiers are inconsistent within the run")
    blocking_failures = [
        item
        for item in ledger
        if item.get("event") == "pilot_call_failed"
        and item.get("continuation_allowed") is not True
    ]
    if blocking_failures:
        raise RuntimeError(
            "run-state contains a fail-closed call failure; execution cannot continue"
        )

    results: list[LiveRunResult] = []
    for item in schedule:
        if item["replicate_number"] > replicate_limit:
            break
        call_id = item["call_id"]
        prior = state[call_id]
        if prior in {"succeeded", "failed"}:
            continue
        if prior == "started":
            raise RuntimeError(
                f"call {call_id} has an unfinished attempt; no retry or replacement is allowed"
            )
        started = {
            "schema_version": 1,
            "event": "pilot_call_started",
            "status": "started",
            "run_manifest_sha256": artifact.sha256,
            "run_id": payload["run_id"],
            "call_id": call_id,
            "replicate_id": item["replicate_id"],
            "condition_id": item["condition_id"],
            "started_at_utc": _utc_timestamp(),
        }
        _append_durable_jsonl(ledger_path, started)
        cell = PromptCell(
            world=SocialWorld(item["social_world"]),
            context=ContextCondition(item["context_condition"]),
            surface=SurfaceVersion(item["surface_version"]),
            prompt_version=prompt_version,
        )
        request = LiveRunRequest(
            cell=cell,
            model=requested_model,
            output=artifact.path.parent / "calls" / call_id,
            sampling=sampling,
            seed=int(item["call_seed"]),
            call_id=call_id,
            replicate_id=item["replicate_id"],
            allow_paid=True,
            paid_run_acknowledgement=PAID_RUN_ACKNOWLEDGEMENT,
            reviewed_prompt_acknowledgement=True,
            max_calls=1,
            run_manifest_sha256=artifact.sha256,
            measurement_protocol=measurement_protocol,
        )
        result: LiveRunResult | None = None
        try:
            result = run_live_once(request)
            _validate_live_result(
                result=result,
                request=request,
                scheduled=item,
                returned_models=returned_models,
            )
        except Exception as exc:
            secret = os.environ.get("OPENAI_API_KEY", "")
            message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
            provider_failure = _verified_provider_failure(
                output=request.output,
                call_id=call_id,
                manifest_hash=artifact.sha256,
                expected_prompt_sha256=item["expected_prompt_sha256"],
            )
            continuation_allowed = provider_failure is not None
            failure = {
                "schema_version": 1,
                "event": "pilot_call_failed",
                "status": "failure",
                "run_manifest_sha256": artifact.sha256,
                "run_id": payload["run_id"],
                "call_id": call_id,
                "replicate_id": item["replicate_id"],
                "condition_id": item["condition_id"],
                "completed_at_utc": _utc_timestamp(),
                "exception_type": type(exc).__name__,
                "exception_message": message,
                "automatic_retry": False,
                "automatic_replacement": False,
                "continuation_allowed": continuation_allowed,
            }
            if provider_failure is not None:
                failure["provider_failure_phase"] = provider_failure.get(
                    "failure_phase"
                )
                failure["provider_response_received"] = provider_failure.get(
                    "provider_response_received"
                )
                failure["possibly_paid_attempt"] = provider_failure.get(
                    "possibly_paid_attempt"
                )
            if result is not None:
                failure["observed_call_id"] = _redact_value(result.call_id, secret)
                failure["observed_requested_model_identifier"] = _redact_value(
                    result.requested_model_identifier, secret
                )
                failure["observed_returned_model_identifier"] = _redact_value(
                    result.returned_model_identifier, secret
                )
                failure["observed_prompt_sha256"] = _redact_value(
                    result.prompt_sha256, secret
                )
            _append_durable_jsonl(ledger_path, failure)
            if continuation_allowed:
                continue
            raise
        _append_durable_jsonl(
            ledger_path,
            {
                "schema_version": 1,
                "event": "pilot_call_succeeded",
                "status": "success",
                "run_manifest_sha256": artifact.sha256,
                "run_id": payload["run_id"],
                "call_id": call_id,
                "replicate_id": item["replicate_id"],
                "condition_id": item["condition_id"],
                "completed_at_utc": _utc_timestamp(),
                "requested_model_identifier": result.requested_model_identifier,
                "returned_model_identifier": result.returned_model_identifier,
                "prompt_sha256": result.prompt_sha256,
                "call_artifact_path": str(result.output.resolve()),
            },
        )
        returned_models.add(result.returned_model_identifier)
        results.append(result)

    latest_ledger = _read_run_state(ledger_path, artifact.sha256)
    final_state = _call_state(latest_ledger, schedule)
    if all(value in {"succeeded", "failed"} for value in final_state.values()):
        completion_events = [
            event
            for event in latest_ledger
            if event.get("event") == "pilot_run_completed"
        ]
        if not completion_events:
            observed = {
                event["returned_model_identifier"]
                for event in latest_ledger
                if event.get("event") == "pilot_call_succeeded"
            }
            if len(observed) > 1:
                raise RuntimeError(
                    "a completed run must have one consistent returned model identifier"
                )
            succeeded_calls = sum(
                value == "succeeded" for value in final_state.values()
            )
            failed_calls = sum(value == "failed" for value in final_state.values())
            _append_durable_jsonl(
                ledger_path,
                {
                    "schema_version": 1,
                    "event": "pilot_run_completed",
                    "status": (
                        "success" if failed_calls == 0 else "completed_with_failures"
                    ),
                    "run_manifest_sha256": artifact.sha256,
                    "run_id": payload["run_id"],
                    "completed_at_utc": _utc_timestamp(),
                    "scheduled_calls": len(schedule),
                    "succeeded_calls": succeeded_calls,
                    "failed_calls": failed_calls,
                    "returned_model_status": (
                        "observed_consistent" if observed else "unobserved"
                    ),
                    "returned_model_identifier": (
                        next(iter(observed)) if observed else None
                    ),
                },
            )
    return tuple(results)


def _verified_provider_failure(
    *,
    output: Path,
    call_id: str,
    manifest_hash: str,
    expected_prompt_sha256: str,
) -> dict[str, Any] | None:
    """Return a finalized provider failure that is safe to skip without retrying."""

    path = output / PROVIDER_ATTEMPT_LEDGER_NAME
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if len(records) != 2 or any(not isinstance(item, dict) for item in records):
        return None
    started, finalized = records
    shared = (
        started.get("call_id") == finalized.get("call_id") == call_id
        and started.get("run_manifest_sha256")
        == finalized.get("run_manifest_sha256")
        == manifest_hash
        and started.get("prompt_sha256")
        == finalized.get("prompt_sha256")
        == expected_prompt_sha256
    )
    if not shared:
        return None
    if (
        started.get("event") != "openai_responses_attempt_started"
        or started.get("status") != "started"
        or finalized.get("event") != "openai_responses_attempt_finalized"
        or finalized.get("status") != "failure"
        or finalized.get("possibly_paid_attempt") is not True
        or not isinstance(finalized.get("failure_phase"), str)
    ):
        return None
    return finalized


def _verify_authorized_pilot_request(
    request: PilotRunRequest, authorization: Any
) -> None:
    if (
        request.requested_model_identifier
        not in authorization.requested_model_identifiers
    ):
        raise ValueError("requested model differs from the authorized study design")
    if request.measurement_protocol != authorization.measurement_protocol:
        raise ValueError(
            "measurement protocol differs from the authorized study design"
        )
    if request.sampling != authorization.sampling:
        raise ValueError(
            "sampling specification differs from the authorized study design"
        )
    if request.calls_per_template != authorization.calls_per_template:
        raise ValueError("calls_per_template differs from the authorized study design")
    if request.maximum_total_calls != authorization.maximum_total_calls:
        raise ValueError("maximum_total_calls differs from the authorized study design")
    if request.maximum_total_calls != (
        len(authorization.condition_ids) * request.calls_per_template
    ):
        raise ValueError("call budget differs from the authorized condition subset")


def _validate_live_result(
    *,
    result: LiveRunResult,
    request: LiveRunRequest,
    scheduled: dict[str, Any],
    returned_models: set[str],
) -> None:
    if (
        result.call_id != request.call_id
        or result.requested_model_identifier != request.model
        or result.prompt_sha256 != scheduled["expected_prompt_sha256"]
        or result.output.resolve() != request.output.resolve()
    ):
        raise RuntimeError("completed call provenance differs from the run manifest")
    if (
        not isinstance(result.returned_model_identifier, str)
        or not result.returned_model_identifier.strip()
    ):
        raise RuntimeError("completed call has no returned model identifier")
    if returned_models and result.returned_model_identifier not in returned_models:
        raise RuntimeError("returned model identifiers are inconsistent within the run")


def load_pilot_manifest(path: Path) -> PilotManifestArtifact:
    """Load a manifest only if its immutable bytes match the sidecar digest."""

    resolved = path.resolve()
    if resolved.name != RUN_MANIFEST_NAME:
        raise ValueError(f"pilot manifest must be named {RUN_MANIFEST_NAME}")
    _validate_live_output_path(resolved.parent)
    try:
        data = resolved.read_bytes()
        sidecar = (resolved.parent / RUN_MANIFEST_HASH_NAME).read_text(encoding="ascii")
    except OSError as exc:
        raise ValueError("pilot manifest or hash sidecar is unavailable") from exc
    actual = sha256(data).hexdigest()
    expected_parts = sidecar.strip().split()
    if len(expected_parts) != 2 or expected_parts[1] != RUN_MANIFEST_NAME:
        raise ValueError("pilot manifest hash sidecar is invalid")
    if expected_parts[0] != actual:
        raise ValueError("pilot manifest changed after creation")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("pilot manifest is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("pilot manifest must be an object")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "hospital_pilot_run_manifest"
        or payload.get("manifest_status") != "immutable_pre_run"
        or payload.get("returned_model_identifier") is not None
        or payload.get("returned_model_status") != "pending"
    ):
        raise ValueError("pilot manifest pre-run schema is invalid")
    return PilotManifestArtifact(path=resolved, sha256=actual, payload=payload)


def scheduled_calls_from_manifest(path: Path) -> tuple[dict[str, Any], ...]:
    """Return a strict immutable schedule mapping for downstream result loading.

    This validates the manifest's own selected-cell hash table and replicate blocks
    without requiring current paid-call authorization, which may later be
    revoked without invalidating already collected development data.
    """

    artifact = load_pilot_manifest(path)
    conditions = artifact.payload.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("pilot manifest must contain at least one condition")
    expected_hashes: dict[str, str] = {}
    for item in conditions:
        if not isinstance(item, dict):
            raise ValueError("pilot manifest condition entries must be objects")
        condition_id = item.get("condition_id")
        prompt_hash = item.get("expected_prompt_sha256")
        if (
            not isinstance(condition_id, str)
            or condition_id in expected_hashes
            or not isinstance(prompt_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash)
        ):
            raise ValueError("pilot manifest condition identity/hash table is invalid")
        try:
            cell = PromptCell(
                world=SocialWorld(item.get("social_world")),
                context=ContextCondition(item.get("context_condition")),
                surface=SurfaceVersion(item.get("surface_version")),
                prompt_version=_required_text(
                    item.get("prompt_version"), "condition.prompt_version"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("pilot manifest condition factors are invalid") from exc
        if (
            cell.condition_id != condition_id
            or cell.prompt_version != artifact.payload.get("prompt_version")
        ):
            raise ValueError("pilot manifest condition factors disagree with identity")
        expected_hashes[condition_id] = prompt_hash
    schedule = artifact.payload.get("schedule")
    validated = _validate_schedule_against_hashes(
        artifact.payload, schedule, expected_hashes
    )
    return tuple(dict(item) for item in validated)


def _build_schedule(
    *,
    run_id: str,
    conditions: list[dict[str, Any]],
    calls_per_template: int,
    seed: int,
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for replicate_number in range(1, calls_per_template + 1):
        replicate_id = f"replicate-{replicate_number:04d}"
        ranked = sorted(
            conditions,
            key=lambda item: (
                sha256(
                    f"{seed}:{replicate_number}:{item['condition_id']}".encode("utf-8")
                ).hexdigest(),
                item["condition_id"],
            ),
        )
        for block_position, condition in enumerate(ranked, 1):
            call_uuid = uuid5(
                NAMESPACE_URL,
                f"{run_id}:{replicate_id}:{condition['condition_id']}",
            )
            schedule.append(
                {
                    "schedule_index": len(schedule) + 1,
                    "replicate_number": replicate_number,
                    "replicate_id": replicate_id,
                    "block_position": block_position,
                    "call_id": f"call-r{replicate_number:04d}-{call_uuid.hex}",
                    "call_seed": seed,
                    **condition,
                }
            )
    return schedule


def _validate_manifest_schedule(
    payload: dict[str, Any], fixture: Any
) -> list[dict[str, Any]]:
    all_fixture_hashes = {
        item.cell.condition_id: item.prompt_sha256 for item in fixture.prompts
    }
    selected_ids = _manifest_condition_ids(payload)
    if not set(selected_ids) <= set(all_fixture_hashes):
        raise ValueError("pilot manifest selects a condition absent from the fixture")
    fixture_hashes = {
        condition_id: all_fixture_hashes[condition_id]
        for condition_id in selected_ids
    }
    return _validate_schedule_against_hashes(
        payload, payload.get("schedule"), fixture_hashes
    )


def _validate_schedule_against_hashes(
    payload: dict[str, Any], schedule: Any, expected_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(schedule, list):
        raise ValueError("pilot manifest schedule must be a list")
    if any(not isinstance(item, dict) for item in schedule):
        raise ValueError("pilot manifest schedule entries must be objects")
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or any(
        not isinstance(item, dict) for item in conditions
    ):
        raise ValueError("pilot manifest conditions must be objects")
    calls_per_template = payload.get("calls_per_template")
    maximum = payload.get("maximum_total_calls")
    condition_count = len(conditions)
    if (
        isinstance(calls_per_template, bool)
        or not isinstance(calls_per_template, int)
        or calls_per_template <= 0
        or condition_count <= 0
        or maximum != condition_count * calls_per_template
        or len(schedule) != maximum
        or len(expected_hashes) != condition_count
        or payload.get("condition_count", condition_count) != condition_count
    ):
        raise ValueError("pilot manifest schedule size is inconsistent")
    condition_ids: set[str] = set()
    for condition in conditions:
        condition_id = condition.get("condition_id")
        try:
            cell = PromptCell(
                world=SocialWorld(condition.get("social_world")),
                context=ContextCondition(condition.get("context_condition")),
                surface=SurfaceVersion(condition.get("surface_version")),
                prompt_version=_required_text(
                    condition.get("prompt_version"), "condition.prompt_version"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("pilot manifest condition factors are invalid") from exc
        if (
            cell.condition_id != condition_id
            or cell.prompt_version != payload.get("prompt_version")
            or condition_id in condition_ids
            or condition.get("expected_prompt_sha256")
            != expected_hashes.get(condition_id)
        ):
            raise ValueError("pilot manifest condition factors disagree with identity")
        condition_ids.add(condition_id)
    if condition_ids != set(expected_hashes):
        raise ValueError("pilot manifest condition table is incomplete")
    call_ids: set[str] = set()
    for replicate in range(1, calls_per_template + 1):
        block = [item for item in schedule if item.get("replicate_number") == replicate]
        if len(block) != condition_count or {
            item.get("condition_id") for item in block
        } != set(expected_hashes):
            raise ValueError("each replicate block must contain every condition once")
    for index, item in enumerate(schedule, 1):
        if not isinstance(item, dict) or item.get("schedule_index") != index:
            raise ValueError("pilot schedule indexes are invalid")
        call_id = item.get("call_id")
        condition_id = item.get("condition_id")
        if not isinstance(call_id, str) or call_id in call_ids:
            raise ValueError("pilot schedule call_ids must be unique")
        call_ids.add(call_id)
        if item.get("expected_prompt_sha256") != expected_hashes.get(condition_id):
            raise ValueError(
                "pilot schedule prompt hash disagrees with reviewed fixture"
            )
    randomization = payload.get("randomization")
    if (
        not isinstance(randomization, dict)
        or randomization.get("algorithm") != "sha256_seeded_sort_v1"
        or randomization.get("blocking")
        != f"one_complete_{condition_count}_cell_shuffle_per_replicate"
        or isinstance(randomization.get("seed"), bool)
        or not isinstance(randomization.get("seed"), int)
    ):
        raise ValueError("pilot manifest randomization specification is invalid")
    expected_schedule = _build_schedule(
        run_id=_required_text(payload.get("run_id"), "run_id"),
        conditions=[dict(item) for item in conditions],
        calls_per_template=calls_per_template,
        seed=randomization["seed"],
    )
    if schedule != expected_schedule:
        raise ValueError(
            "pilot schedule does not match its seeded replicate-block design"
        )
    return schedule


def _manifest_condition_ids(payload: dict[str, Any]) -> list[str]:
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or any(
        not isinstance(item, dict) for item in conditions
    ):
        raise ValueError("pilot manifest conditions must be objects")
    condition_ids = [item.get("condition_id") for item in conditions]
    if any(not isinstance(item, str) for item in condition_ids):
        raise ValueError("pilot manifest condition identities are invalid")
    return sorted(condition_ids)


def _project_provenance(root: Path) -> dict[str, Any]:
    revision = _git_output(root, ["rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("project has no immutable Git revision")
    dirty_output = _git_output(root, ["status", "--porcelain", "--untracked-files=all"])
    lock_path = root / "uv.lock"
    pyproject_path = root / "pyproject.toml"
    try:
        pyproject = pyproject_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("pyproject.toml is unavailable for provenance") from exc
    match = PINNED_SILISOCS_PATTERN.search(pyproject)
    if match is None:
        raise ValueError("pinned SiliSocS revision is unavailable")
    pinned_revision = match.group(1)
    try:
        installed = distribution("silisocs")
        direct_url_text = installed.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "null")
        installed_revision = direct_url["vcs_info"]["commit_id"]
    except (
        PackageNotFoundError,
        AttributeError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("installed SiliSocS revision is unavailable") from exc
    if installed_revision != pinned_revision:
        raise ValueError("installed SiliSocS revision differs from the project pin")
    return {
        "project_git_revision": revision,
        "project_git_dirty": bool(dirty_output),
        "uv_lock_sha256": file_sha256(lock_path),
        "pinned_silisocs_revision": pinned_revision,
        "installed_silisocs_revision": installed_revision,
        "installed_silisocs_version": installed.version,
    }


def _git_output(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("project Git provenance is unavailable")
    return completed.stdout.strip()


def _read_run_state(path: Path, manifest_hash: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid run-state JSON at line {line_number}") from exc
        if (
            not isinstance(value, dict)
            or value.get("run_manifest_sha256") != manifest_hash
        ):
            raise RuntimeError("run-state record is not linked to this manifest")
        records.append(value)
    return records


def _call_state(
    ledger: list[dict[str, Any]], schedule: list[dict[str, Any]]
) -> dict[str, str]:
    states = {item["call_id"]: "not_started" for item in schedule}
    scheduled_by_call = {item["call_id"]: item for item in schedule}
    starts: set[str] = set()
    terminals: set[str] = set()
    completion_seen = False
    next_start_index = 0
    active_call_id: str | None = None
    for index, event in enumerate(ledger):
        kind = event.get("event")
        if kind == "pilot_run_completed":
            if completion_seen or index != len(ledger) - 1:
                raise RuntimeError(
                    "run-state ledger has a duplicate/out-of-order completion"
                )
            if not all(value in {"succeeded", "failed"} for value in states.values()):
                raise RuntimeError(
                    "run-state completion precedes scheduled call completion"
                )
            returned = {
                item.get("returned_model_identifier")
                for item in ledger[:index]
                if item.get("event") == "pilot_call_succeeded"
            }
            succeeded_calls = sum(value == "succeeded" for value in states.values())
            failed_calls = sum(value == "failed" for value in states.values())
            expected_status = (
                "success" if failed_calls == 0 else "completed_with_failures"
            )
            expected_model_status = (
                "observed_consistent" if returned else "unobserved"
            )
            expected_model = next(iter(returned)) if returned else None
            if (
                event.get("status") != expected_status
                or event.get("scheduled_calls") != len(schedule)
                or event.get("succeeded_calls") != succeeded_calls
                or event.get("failed_calls") != failed_calls
                or event.get("returned_model_status") != expected_model_status
                or len(returned) > 1
                or event.get("returned_model_identifier") != expected_model
            ):
                raise RuntimeError("run-state completion summary is inconsistent")
            completion_seen = True
            continue
        if completion_seen:
            raise RuntimeError("run-state events cannot follow run completion")
        call_id = event.get("call_id")
        if call_id not in states:
            raise RuntimeError("run-state ledger contains an unscheduled call_id")
        scheduled = scheduled_by_call[call_id]
        if kind == "pilot_call_started":
            if call_id in starts:
                raise RuntimeError("run-state ledger contains a duplicate call start")
            if active_call_id is not None:
                raise RuntimeError(
                    "run-state ledger starts a call before the prior call is terminal"
                )
            if (
                next_start_index >= len(schedule)
                or call_id != schedule[next_start_index]["call_id"]
            ):
                raise RuntimeError(
                    "run-state call starts are not an exact manifest schedule prefix"
                )
            if (
                event.get("status") != "started"
                or event.get("replicate_id") != scheduled["replicate_id"]
                or event.get("condition_id") != scheduled["condition_id"]
            ):
                raise RuntimeError("run-state call start disagrees with the schedule")
            starts.add(call_id)
            next_start_index += 1
            active_call_id = call_id
            states[call_id] = "started"
        elif kind in {"pilot_call_succeeded", "pilot_call_failed"}:
            if (
                call_id not in starts
                or call_id in terminals
                or call_id != active_call_id
            ):
                raise RuntimeError(
                    "run-state ledger has an invalid terminal call event"
                )
            if kind == "pilot_call_succeeded":
                if (
                    event.get("status") != "success"
                    or event.get("replicate_id") != scheduled["replicate_id"]
                    or event.get("condition_id") != scheduled["condition_id"]
                    or event.get("prompt_sha256") != scheduled["expected_prompt_sha256"]
                    or not isinstance(event.get("requested_model_identifier"), str)
                    or not isinstance(event.get("returned_model_identifier"), str)
                    or not event["returned_model_identifier"].strip()
                ):
                    raise RuntimeError("run-state call success provenance is invalid")
            elif (
                event.get("status") != "failure"
                or event.get("automatic_retry") is not False
                or event.get("automatic_replacement") is not False
                or type(event.get("continuation_allowed")) is not bool
                or event.get("replicate_id") != scheduled["replicate_id"]
                or event.get("condition_id") != scheduled["condition_id"]
            ):
                raise RuntimeError("run-state call failure provenance is invalid")
            terminals.add(call_id)
            active_call_id = None
            states[call_id] = "succeeded" if kind.endswith("succeeded") else "failed"
        else:
            raise RuntimeError("run-state ledger contains an unknown event")
    return states


def _write_exclusive_durable(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _append_durable_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pilot manifest {name} must be non-empty text")
    return value


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"pilot manifest {name} must be an object")
    return value


def _redact_value(value: Any, secret: str) -> str:
    rendered = str(value)
    return rendered.replace(secret, "[REDACTED]") if secret else rendered


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
