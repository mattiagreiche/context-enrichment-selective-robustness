"""Safety-gated one-call SiliSocS boundary for a reviewed development pilot."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from .catalog import hospital_cells
from .domain import PromptCell
from .openai_provider import OFFICIAL_OPENAI_API_BASE
from .parsing import parse_model_response
from .prompts import render_prompt
from .sampling import LogprobMode, SamplingSpec
from .silisocs_runner import repository_root
from .validation import text_sha256


PAID_RUN_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_ONE_PAID_CALL"
FIXTURE_REVIEW_STATUS = "complete"
FIXTURE_AUTHORIZATION_STATUS = "authorized_for_paid_development"
_MODEL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RUN_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ReviewedPrompt:
    cell: PromptCell
    file: Path
    text: str
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class ReviewedFixture:
    prompt_version: str
    manifest_path: Path
    manifest_sha256: str
    prompts: tuple[ReviewedPrompt, ...]

    def prompt_for(self, condition_id: str) -> ReviewedPrompt:
        matches = [
            item for item in self.prompts if item.cell.condition_id == condition_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "reviewed fixture does not resolve the condition exactly once"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class StudyAuthorization:
    path: Path
    artifact_sha256: str
    prompt_version: str
    requested_model_identifiers: tuple[str, ...]
    condition_ids: tuple[str, ...]
    sampling: SamplingSpec
    calls_per_template: int
    maximum_total_calls: int
    declared_human_review_complete: bool
    paid_calls_authorized: bool
    measurement_protocol: str = "action_frequency_v1"


@dataclass(frozen=True, slots=True)
class LiveRunRequest:
    cell: PromptCell
    model: str
    output: Path
    sampling: SamplingSpec
    seed: int
    call_id: str
    replicate_id: str
    allow_paid: bool
    paid_run_acknowledgement: str
    reviewed_prompt_acknowledgement: bool
    max_calls: int = 1
    run_manifest_sha256: str | None = None
    measurement_protocol: str = "action_frequency_v1"

    def validate(self) -> None:
        if not isinstance(self.cell, PromptCell):
            raise TypeError("cell must be a PromptCell")
        if not _MODEL_IDENTIFIER_PATTERN.fullmatch(self.model):
            raise ValueError("model identifier contains unsafe characters")
        if not isinstance(self.sampling, SamplingSpec):
            raise TypeError("sampling must be a canonical SamplingSpec")
        if self.cell.prompt_version.startswith("draft"):
            raise ValueError("live execution refuses unreviewed draft prompts")
        if (
            not _VERSION_IDENTIFIER_PATTERN.fullmatch(self.cell.prompt_version)
            or ".." in self.cell.prompt_version
        ):
            raise ValueError("prompt version contains unsafe characters")
        if not self.reviewed_prompt_acknowledgement:
            raise ValueError("live execution requires reviewed-prompt acknowledgement")
        if not self.allow_paid:
            raise ValueError("live execution requires --allow-paid")
        if self.paid_run_acknowledgement != PAID_RUN_ACKNOWLEDGEMENT:
            raise ValueError(
                "paid-run acknowledgement does not match the required phrase"
            )
        if isinstance(self.max_calls, bool) or self.max_calls != 1:
            raise ValueError("the live boundary permits exactly one focal model call")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not _RUN_IDENTIFIER_PATTERN.fullmatch(self.call_id):
            raise ValueError("call_id contains unsafe characters")
        if not _RUN_IDENTIFIER_PATTERN.fullmatch(self.replicate_id):
            raise ValueError("replicate_id contains unsafe characters")
        if self.run_manifest_sha256 is not None and not _is_sha256(
            self.run_manifest_sha256
        ):
            raise ValueError("run_manifest_sha256 must be a lowercase SHA-256 digest")
        if self.run_manifest_sha256 is None:
            raise ValueError(
                "the one-call primitive requires an immutable pilot manifest binding"
            )
        if self.measurement_protocol not in {
            "action_frequency_v1",
            "first_action_branch_v1",
        }:
            raise ValueError("measurement_protocol is invalid")


@dataclass(frozen=True, slots=True)
class LiveRunResult:
    call_id: str
    requested_model_identifier: str
    returned_model_identifier: str
    prompt_sha256: str
    output: Path


def build_live_command(
    request: LiveRunRequest,
    *,
    reviewed_prompt: ReviewedPrompt | None = None,
) -> list[str]:
    """Compose one native SiliSocS command without placing a key in argv."""

    request.validate()
    if reviewed_prompt is None:
        fixture, _ = verify_live_governance(
            request.cell.prompt_version, request.measurement_protocol
        )
        reviewed_prompt = fixture.prompt_for(request.cell.condition_id)
    root = repository_root()
    config_root = root / "scenarios" / "selective_robustness" / "conf"
    output = request.output.resolve()
    sampling = request.sampling
    top_p_value = "null" if sampling.top_p is None else str(float(sampling.top_p))
    command = [
        sys.executable,
        "-m",
        "silisocs.runtime.runner",
        "--config-path",
        str(config_root),
        "agents=default",
        "output_rootname=" + json.dumps(str(output)),
        "hydra.run.dir=" + json.dumps(str(output / ".hydra-run")),
        f"seed={request.seed}",
        "num_agents=2",
        "num_steps=2",
        f"social_world={request.cell.world.value}",
        f"context_condition={request.cell.context.value}",
        f"surface_version={request.cell.surface.value}",
        f"prompt_version={request.cell.prompt_version}",
        f"condition_id={request.cell.condition_id}",
        f"call_id={request.call_id}",
        f"replicate_id={request.replicate_id}",
        f"episode_id={request.call_id}",
        "selective.study_stage=development",
        f"selective.condition_id={request.cell.condition_id}",
        f"selective.call_id={request.call_id}",
        f"selective.replicate_id={request.replicate_id}",
        f"selective.episode_id={request.call_id}",
        f"sampling_spec.temperature={float(sampling.temperature)}",
        f"sampling_spec.reasoning_effort={sampling.reasoning_effort.value}",
        f"sampling_spec.top_p.mode={sampling.top_p_mode.value}",
        f"sampling_spec.top_p.value={top_p_value}",
        f"sampling_spec.max_output_tokens={sampling.max_output_tokens}",
        f"sampling_spec.timeout_seconds={float(sampling.timeout_seconds)}",
        "sampling_spec.transport_retries=0",
        "sampling_spec.content_retries=0",
        "sampling_spec.response_api=responses",
        "sampling_spec.store=false",
        "sampling_spec.provider_seed=unsupported_omitted",
        "sim.llm.provider=selective_robustness.openai_provider.OpenAIResponsesModel",
        f"sim.llm.name={request.model}",
        f"sim.llm.api_base={OFFICIAL_OPENAI_API_BASE}",
        "sim.llm.api_key=null",
        "provider_extra_kwargs.text_response=null",
        "++provider_extra_kwargs.max_retries=${sampling_spec.transport_retries}",
        "++provider_extra_kwargs.sampling_spec=${sampling_spec}",
        f"++provider_extra_kwargs.call_id={request.call_id}",
        "++provider_extra_kwargs.expected_prompt_file="
        + json.dumps(str(reviewed_prompt.file.resolve())),
        f"++provider_extra_kwargs.expected_prompt_sha256={reviewed_prompt.prompt_sha256}",
    ]
    if sampling.logprob_mode is LogprobMode.FIRST_ACTION_BRANCH:
        command.extend(
            [
                "++sampling_spec.logprobs.mode=first_action_branch",
                f"++sampling_spec.logprobs.top_logprobs={sampling.top_logprobs}",
            ]
        )
    if request.run_manifest_sha256 is not None:
        command.append(
            f"++provider_extra_kwargs.run_manifest_sha256={request.run_manifest_sha256}"
        )
    return command


def run_live_once(request: LiveRunRequest) -> LiveRunResult:
    """Run one authorized prompt, enforcing fixture and provenance contracts."""

    request.validate()
    fixture, authorization = verify_live_governance(
        request.cell.prompt_version, request.measurement_protocol
    )
    _verify_authorized_call_parameters(request, authorization)
    reviewed_prompt = fixture.prompt_for(request.cell.condition_id)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for an acknowledged live run")
    output = _validate_live_output_path(request.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("live output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    root = repository_root()
    environment = _sanitized_live_environment(root, api_key)
    try:
        with tempfile.TemporaryDirectory(prefix="selective-robustness-live-") as cwd:
            completed = subprocess.run(
                build_live_command(request, reviewed_prompt=reviewed_prompt),
                cwd=cwd,
                env=environment,
                check=False,
                text=True,
                capture_output=True,
                timeout=max(300.0, request.sampling.timeout_seconds + 120.0),
            )
    except (OSError, subprocess.SubprocessError) as exc:
        _assert_secret_absent(output, api_key)
        exception_parts = [str(exc)]
        for attribute in ("stdout", "stderr", "output"):
            value = getattr(exc, attribute, None)
            if value:
                exception_parts.append(
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else str(value)
                )
        detail = " | ".join(exception_parts).replace(api_key, "[REDACTED]")
        raise RuntimeError(f"SiliSocS live subprocess failed: {detail}") from exc
    if completed.returncode != 0:
        _assert_secret_absent(output, api_key)
        detail = " | ".join(completed.stderr.strip().splitlines()[-12:]).replace(
            api_key, "[REDACTED]"
        )
        raise RuntimeError(f"SiliSocS live boundary failed: {detail or 'no detail'}")
    return _verify_live_artifacts(request, reviewed_prompt, api_key)


def verify_live_governance(
    prompt_version: str,
    measurement_protocol: str = "action_frequency_v1",
) -> tuple[ReviewedFixture, StudyAuthorization]:
    """Verify complete fixture review plus declared study-level authorization."""

    if str(prompt_version).startswith("draft"):
        raise ValueError("live execution refuses unreviewed draft prompts")
    return (
        _load_complete_reviewed_fixture(prompt_version),
        _load_study_authorization(prompt_version, measurement_protocol),
    )


def _load_complete_reviewed_fixture(prompt_version: str) -> ReviewedFixture:
    root = (
        repository_root()
        / "tests"
        / "fixtures"
        / "prompts"
        / "hospital"
        / prompt_version
    )
    manifest_path = root / "manifest.json"
    manifest_text = _read_text(manifest_path, "reviewed prompt fixture manifest")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ValueError("reviewed prompt fixture manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("reviewed prompt fixture manifest must be an object")
    required_governance = {
        "status": "reviewed",
        "review_status": FIXTURE_REVIEW_STATUS,
        "authorization_status": FIXTURE_AUTHORIZATION_STATUS,
        "frozen": True,
        "prompt_version": prompt_version,
    }
    if any(manifest.get(key) != value for key, value in required_governance.items()):
        raise ValueError(
            "live execution requires a complete reviewed, authorized, frozen fixture"
        )
    entries = manifest.get("cells")
    expected_cells = hospital_cells(prompt_version)
    expected_count = len(expected_cells)
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(
            f"reviewed prompt fixture must contain all {expected_count} cells"
        )
    expected_by_id = {cell.condition_id: cell for cell in expected_cells}
    if len(expected_by_id) != expected_count:
        raise AssertionError("hospital catalog condition ids are not unique")
    seen: set[str] = set()
    reviewed: list[ReviewedPrompt] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("reviewed prompt fixture cell entries must be objects")
        condition_id = entry.get("condition_id")
        if not isinstance(condition_id, str) or condition_id not in expected_by_id:
            raise ValueError("reviewed prompt fixture contains an unknown condition")
        if condition_id in seen:
            raise ValueError("reviewed prompt fixture contains a duplicate condition")
        seen.add(condition_id)
        cell = expected_by_id[condition_id]
        expected_factors = {
            "prompt_version": prompt_version,
            "social_world": cell.world.value,
            "context_condition": cell.context.value,
            "surface_version": cell.surface.value,
        }
        if any(entry.get(key) != value for key, value in expected_factors.items()):
            raise ValueError("reviewed prompt fixture factors do not match the catalog")
        file_name = entry.get("file")
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError("reviewed prompt fixture has an invalid file name")
        fixture_path = root / file_name
        fixture_text = _read_text(fixture_path, "reviewed prompt fixture text")
        prompt_hash = text_sha256(fixture_text)
        if entry.get("prompt_sha256") != prompt_hash:
            raise ValueError("reviewed prompt fixture hash is invalid")
        if fixture_text != render_prompt(cell).text:
            raise ValueError(
                "reviewed prompt fixture does not match the current renderer"
            )
        reviewed.append(
            ReviewedPrompt(
                cell=cell,
                file=fixture_path,
                text=fixture_text,
                prompt_sha256=prompt_hash,
            )
        )
    if seen != set(expected_by_id):
        raise ValueError(
            "reviewed prompt fixture does not contain the complete crossing"
        )
    return ReviewedFixture(
        prompt_version=prompt_version,
        manifest_path=manifest_path,
        manifest_sha256=text_sha256(manifest_text),
        prompts=tuple(reviewed),
    )


def _load_study_authorization(
    prompt_version: str, measurement_protocol: str
) -> StudyAuthorization:
    names = {
        "action_frequency_v1": "design.yaml",
        "first_action_branch_v1": "probability_design.yaml",
    }
    try:
        name = names[measurement_protocol]
    except KeyError as exc:
        raise ValueError("unknown measurement protocol") from exc
    path = repository_root() / "experiments" / "studies" / "hospital_pilot" / name
    text = _read_text(path, "study authorization artifact")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("study authorization artifact is invalid YAML") from exc
    if not isinstance(value, dict):
        raise ValueError("study authorization artifact must be a mapping")
    study = value.get("study")
    status = value.get("status")
    stimulus = value.get("stimulus_contract")
    pilot_workflow = value.get("pilot_workflow")
    if not all(
        isinstance(item, dict) for item in (study, status, stimulus, pilot_workflow)
    ):
        raise ValueError("study authorization artifact is incomplete")
    assert (
        isinstance(study, dict)
        and isinstance(status, dict)
        and isinstance(stimulus, dict)
        and isinstance(pilot_workflow, dict)
    )
    review_complete = (
        status.get("prompt_review") == "complete"
        and status.get("human_manipulation_check") == "complete"
        and status.get("ethics_determination") == "complete"
        and stimulus.get("status") == "reviewed"
    )
    paid_authorized = status.get("paid_calls_authorized") is True
    if (
        study.get("stage") != "development"
        or study.get("prompt_version") != prompt_version
        or study.get("measurement_protocol", "action_frequency_v1")
        != measurement_protocol
        or not review_complete
        or not paid_authorized
    ):
        raise ValueError(
            "study-level human review and paid-call authorization are not declared complete"
        )
    model_snapshots = value.get("model_snapshots")
    if (
        not isinstance(model_snapshots, list)
        or not model_snapshots
        or any(
            not isinstance(model, str)
            or not _MODEL_IDENTIFIER_PATTERN.fullmatch(model)
            for model in model_snapshots
        )
        or len(set(model_snapshots)) != len(model_snapshots)
    ):
        raise ValueError(
            "an authorized study must declare one or more unique pinned model identifiers"
        )
    all_condition_ids = {
        cell.condition_id for cell in hospital_cells(prompt_version)
    }
    configured_condition_ids = pilot_workflow.get("condition_ids")
    if configured_condition_ids is None:
        condition_ids = tuple(sorted(all_condition_ids))
    elif (
        not isinstance(configured_condition_ids, list)
        or not configured_condition_ids
        or any(not isinstance(item, str) for item in configured_condition_ids)
        or len(set(configured_condition_ids)) != len(configured_condition_ids)
        or not set(configured_condition_ids) <= all_condition_ids
    ):
        raise ValueError(
            "an authorized study must declare a nonempty unique condition subset"
        )
    else:
        condition_ids = tuple(configured_condition_ids)
    calls_per_template = value.get("calls_per_template")
    maximum_total_calls = pilot_workflow.get("maximum_total_calls")
    if (
        isinstance(calls_per_template, bool)
        or not isinstance(calls_per_template, int)
        or calls_per_template <= 0
        or isinstance(maximum_total_calls, bool)
        or not isinstance(maximum_total_calls, int)
        or maximum_total_calls != len(condition_ids) * calls_per_template
    ):
        raise ValueError(
            "an authorized study must declare a positive call budget with "
            "maximum_total_calls equal to selected conditions * calls_per_template"
        )
    try:
        sampling = SamplingSpec.from_mapping(pilot_workflow.get("sampling_spec"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "an authorized study must declare one complete canonical sampling spec"
        ) from exc
    return StudyAuthorization(
        path=path,
        artifact_sha256=text_sha256(text),
        prompt_version=prompt_version,
        requested_model_identifiers=tuple(model_snapshots),
        condition_ids=condition_ids,
        sampling=sampling,
        calls_per_template=calls_per_template,
        maximum_total_calls=maximum_total_calls,
        declared_human_review_complete=True,
        paid_calls_authorized=True,
        measurement_protocol=measurement_protocol,
    )


def _verify_authorized_call_parameters(
    request: LiveRunRequest, authorization: StudyAuthorization
) -> None:
    if request.model not in authorization.requested_model_identifiers:
        raise ValueError("requested model differs from the authorized study design")
    if request.cell.condition_id not in authorization.condition_ids:
        raise ValueError("condition differs from the authorized study subset")
    if request.sampling != authorization.sampling:
        raise ValueError(
            "sampling specification differs from the authorized study design"
        )
    if request.measurement_protocol != authorization.measurement_protocol:
        raise ValueError(
            "measurement protocol differs from the authorized study design"
        )


def _sanitized_live_environment(root: Path, api_key: str) -> dict[str, str]:
    """Build a minimal child environment with no ambient provider routing."""

    environment = {
        "OPENAI_API_KEY": api_key,
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": str(root / "src"),
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONUTF8": "1",
    }
    for name in ("LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _validate_live_output_path(output: Path) -> Path:
    resolved = output.resolve()
    configured_root = repository_root() / "results"
    if configured_root.is_symlink():
        raise ValueError("the approved results directory cannot be a symlink")
    approved_root = configured_root.resolve()
    if resolved == approved_root or not resolved.is_relative_to(approved_root):
        raise ValueError(
            "live output must be a child of the approved results directory"
        )
    return resolved


def _verify_live_artifacts(
    request: LiveRunRequest,
    reviewed_prompt: ReviewedPrompt,
    api_key: str,
) -> LiveRunResult:
    output = request.output.resolve()
    _assert_secret_absent(output, api_key)
    event_path = output / "action_events.jsonl"
    if not event_path.is_file():
        raise RuntimeError("live run did not produce action_events.jsonl")
    events = _read_json_lines(event_path)
    focal = [
        event
        for event in events
        if event.get("action_type") == "choose_hospital_action"
        and isinstance(event.get("data"), dict)
        and event["data"].get("source") == "model"
    ]
    if len(focal) != 1:
        raise RuntimeError(
            f"expected one focal model decision event, found {len(focal)}"
        )
    data = focal[0]["data"]
    if data.get("condition_id") != request.cell.condition_id:
        raise RuntimeError("live result condition does not match the request")
    if (
        data.get("call_id") != request.call_id
        or data.get("episode_id") != request.call_id
    ):
        raise RuntimeError("live result call identity does not match the schedule")
    if data.get("replicate_id") != request.replicate_id:
        raise RuntimeError("live result replicate identity does not match the schedule")
    if data.get("study_stage") != "development":
        raise RuntimeError("live result is not marked as development data")
    if data.get("attempts") != 1 or data.get("provider_attempts") != 1:
        raise RuntimeError("live result violated the one-call attempt policy")
    attempts = data.get("content_attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise RuntimeError("live result did not preserve exactly one content attempt")
    if (
        data.get("prompt") != reviewed_prompt.text
        or data.get("prompt_sha256") != reviewed_prompt.prompt_sha256
        or attempts[0].get("request_prompt") != reviewed_prompt.text
        or attempts[0].get("request_prompt_sha256") != reviewed_prompt.prompt_sha256
    ):
        raise RuntimeError("actual provider prompt differs from the reviewed fixture")
    metadata = data.get("provider_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("live result is missing provider metadata")
    _verify_provider_metadata(request, metadata, reviewed_prompt)
    if not isinstance(metadata.get("raw_provider_response"), dict):
        raise RuntimeError("live result did not preserve the raw provider response")
    if data.get("raw_response") is None:
        raise RuntimeError("live result did not preserve the final raw response")
    raw_response = data["raw_response"]
    if not isinstance(raw_response, str):
        raise RuntimeError("live result raw response must be text")
    parsed = parse_model_response(raw_response)
    expected_action = parsed.action.value if parsed.action is not None else None
    if (
        data.get("action") != expected_action
        or data.get("valid_parse") is not parsed.valid
        or data.get("parse_method") != parsed.method.value
        or metadata.get("canonical_parse")
        != {
            "valid": parsed.valid,
            "method": parsed.method.value,
            "action": expected_action,
        }
    ):
        raise RuntimeError("live event parse state disagrees with the canonical parser")

    call_path = output / "prompts_and_responses.jsonl"
    if not call_path.is_file():
        raise RuntimeError("live run did not produce prompts_and_responses.jsonl")
    calls = _read_json_lines(call_path)
    starts = [
        item
        for item in calls
        if item.get("event") == "openai_responses_attempt_started"
        and item.get("call_id") == request.call_id
    ]
    finals = [
        item
        for item in calls
        if item.get("event") == "openai_responses_attempt_finalized"
        and item.get("call_id") == request.call_id
    ]
    if len(starts) != 1 or len(finals) != 1:
        raise RuntimeError(
            "provider ledger must contain one start and one final record"
        )
    start, final = starts[0], finals[0]
    if final.get("status") != "success":
        raise RuntimeError("live provider attempt did not finalize successfully")
    for record in (start, final):
        if (
            record.get("prompt") != reviewed_prompt.text
            or record.get("prompt_sha256") != reviewed_prompt.prompt_sha256
            or record.get("run_manifest_sha256") != request.run_manifest_sha256
        ):
            raise RuntimeError("provider ledger provenance disagrees with the request")
    attempt = attempts[0]
    if (
        final.get("output_text") != attempt.get("raw_response")
        or attempt.get("raw_response") != raw_response
    ):
        raise RuntimeError("provider output does not match the decision attempt")
    if final.get("provider_metadata") != attempt.get("provider_metadata"):
        raise RuntimeError("provider metadata does not match the decision attempt")
    if final.get("provider_metadata") != metadata:
        raise RuntimeError("provider metadata does not match the focal event")
    returned = metadata.get("returned_model_identifier")
    if not isinstance(returned, str) or not returned.strip():
        raise RuntimeError("provider did not report a returned model identifier")
    return LiveRunResult(
        call_id=request.call_id,
        requested_model_identifier=request.model,
        returned_model_identifier=returned,
        prompt_sha256=reviewed_prompt.prompt_sha256,
        output=output,
    )


def _verify_provider_metadata(
    request: LiveRunRequest,
    metadata: dict[str, Any],
    reviewed_prompt: ReviewedPrompt,
) -> None:
    if metadata.get("transport_max_retries") != 0:
        raise RuntimeError("live provider did not use the zero-retry policy")
    if metadata.get("requested_model_identifier") != request.model:
        raise RuntimeError(
            "live result did not preserve the requested model identifier"
        )
    if metadata.get("sampling_spec") != request.sampling.as_dict():
        raise RuntimeError("executed sampling parameters disagree with the run request")
    if metadata.get("reasoning_effort") != request.sampling.reasoning_effort.value:
        raise RuntimeError("executed reasoning effort disagrees with the run request")
    if metadata.get("call_id") != request.call_id:
        raise RuntimeError("provider metadata call_id disagrees with the run request")
    if metadata.get("run_manifest_sha256") != request.run_manifest_sha256:
        raise RuntimeError(
            "provider metadata manifest hash disagrees with the run request"
        )
    routing = metadata.get("routing")
    if not isinstance(routing, dict) or routing != {
        "endpoint": OFFICIAL_OPENAI_API_BASE,
        "organization": "not_configured",
        "project": "not_configured",
        "ambient_proxy_routing": "disabled",
    }:
        raise RuntimeError(
            "provider routing metadata is not the approved configuration"
        )
    parameters = metadata.get("request_parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("provider request parameters are missing")
    expected: dict[str, Any] = {
        "model": request.model,
        "input": reviewed_prompt.text,
        "store": False,
        "temperature": float(request.sampling.temperature),
        "reasoning": {"effort": request.sampling.reasoning_effort.value},
        "max_output_tokens": request.sampling.max_output_tokens,
        "timeout": float(request.sampling.timeout_seconds),
    }
    if request.sampling.top_p is not None:
        expected["top_p"] = float(request.sampling.top_p)
    expected.update(
        {
            key: value
            for key, value in request.sampling.provider_request_options().items()
            if key in {"include", "top_logprobs"}
        }
    )
    if parameters != expected:
        raise RuntimeError(
            "executed provider request parameters disagree with the canonical run request"
        )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"event at {path}:{line_number} is not an object")
        events.append(value)
    return events


def _assert_secret_absent(output: Path, secret: str) -> None:
    secret_bytes = secret.encode("utf-8")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"live artifact directory contains a symlink: {path}")
        if path.is_file() and secret_bytes in path.read_bytes():
            raise RuntimeError(f"API credential leaked into live artifact {path}")


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def file_sha256(path: Path) -> str:
    """Hash a file as bytes for run provenance."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
