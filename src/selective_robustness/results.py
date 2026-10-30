"""Validate and read focal SiliSocS decision events for probability analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeAlias

from .domain import ContextCondition, DomainAction, SocialWorld, SurfaceVersion
from .parsing import ParseMethod, parse_model_response


class ResultValidationError(ValueError):
    """Raised when a result artifact violates the focal-event schema."""


class StudyStage(StrEnum):
    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"


class ParseState(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


FrozenJSON: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["FrozenJSON", ...]
    | tuple[tuple[str, "FrozenJSON"], ...]
)


@dataclass(frozen=True, slots=True)
class ContentAttempt:
    """One immutable, provider-agnostic model content attempt."""

    request_prompt: str
    raw_response: str
    requested_at_utc: str
    completed_at_utc: str
    provider_metadata: tuple[tuple[str, FrozenJSON], ...]
    attempt: int | None = None
    extra_fields: tuple[tuple[str, FrozenJSON], ...] = ()


@dataclass(frozen=True, slots=True)
class FocalDecision:
    """One schema-validated focal-model decision event."""

    study_stage: StudyStage
    condition_id: str
    scenario: str
    social_world: SocialWorld
    context_condition: ContextCondition
    surface_version: SurfaceVersion
    requested_model_identifier: str
    returned_model_identifier: str
    prompt_version: str
    raw_prompt: str
    prompt_sha256: str
    raw_response: str
    parse_state: ParseState
    parse_method: ParseMethod
    action: DomainAction | None
    parse_note: str
    call_id: str
    replicate_id: str
    episode_id: str
    seed: int
    attempts: int
    provider_attempts: int
    provider_metadata: tuple[tuple[str, FrozenJSON], ...]
    content_attempts: tuple[ContentAttempt, ...]
    artifact_source: str
    source_artifact_sha256: str
    timestamp: str | None = None

    @property
    def valid_parse(self) -> bool:
        return self.parse_state is ParseState.VALID


@dataclass(frozen=True, slots=True)
class _ArtifactDefaults:
    study_stage: StudyStage | None = None
    requested_model_identifier: str | None = None
    prompt_version: str | None = None
    artifact_source: str = "<iterable>"
    source_artifact_sha256: str | None = None


def load_focal_decisions(
    source: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    study_stage: StudyStage | str | None = None,
    requested_model_identifier: str | None = None,
    prompt_version: str | None = None,
) -> tuple[FocalDecision, ...]:
    """Load focal decisions from JSON, JSONL, or in-memory event mappings.

    A JSON object may be one event or an artifact wrapper with an ``events``
    list.  Wrapper defaults can be supplied at top level or under ``study``.
    Explicit function arguments are defaults only: any conflicting event or
    artifact value is rejected.
    """

    supplied_requested_model = _optional_text(
        requested_model_identifier, "requested_model_identifier argument"
    )
    supplied = _ArtifactDefaults(
        study_stage=_optional_stage(study_stage, "study_stage argument"),
        requested_model_identifier=supplied_requested_model,
        prompt_version=_optional_text(prompt_version, "prompt_version argument"),
    )
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload, artifact_sha256 = _read_artifact(path)
        supplied = _ArtifactDefaults(
            study_stage=supplied.study_stage,
            requested_model_identifier=supplied.requested_model_identifier,
            prompt_version=supplied.prompt_version,
            artifact_source=str(path.resolve()),
            source_artifact_sha256=artifact_sha256,
        )
    else:
        payload = source
    return parse_focal_decisions(payload, defaults=supplied)


def parse_focal_decisions(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    defaults: _ArtifactDefaults | None = None,
) -> tuple[FocalDecision, ...]:
    """Pure counterpart to :func:`load_focal_decisions` for decoded events."""

    inherited = defaults or _ArtifactDefaults()
    prepared_payload: Mapping[str, Any] | tuple[Mapping[str, Any], ...]
    if isinstance(payload, Mapping):
        prepared_payload = payload
    elif isinstance(payload, (str, bytes)) or not isinstance(payload, Iterable):
        raise ResultValidationError("result payload must be an event object or iterable")
    else:
        prepared_payload = tuple(payload)
    if inherited.source_artifact_sha256 is None:
        inherited = _ArtifactDefaults(
            study_stage=inherited.study_stage,
            requested_model_identifier=inherited.requested_model_identifier,
            prompt_version=inherited.prompt_version,
            artifact_source=inherited.artifact_source,
            source_artifact_sha256=_canonical_payload_sha256(prepared_payload),
        )
    events, artifact_defaults = _events_and_defaults(prepared_payload, inherited)
    decisions: list[FocalDecision] = []
    call_ids: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ResultValidationError(f"event {index}: expected an object")
        if event.get("action_type") != "choose_hospital_action":
            continue
        data = _mapping(event.get("data"), f"event {index}.data")
        if data.get("source") != "model":
            continue
        decision = _parse_focal_event(
            event,
            data,
            index=index,
            defaults=artifact_defaults,
        )
        if decision.call_id in call_ids:
            raise ResultValidationError(
                f"event {index}: duplicate focal event call_id {decision.call_id!r}"
            )
        call_ids.add(decision.call_id)
        decisions.append(decision)
    records = tuple(decisions)
    _require_one_prompt_hash_per_condition_version(records)
    _require_consistent_returned_models(records)
    return records


def _read_artifact(
    path: Path,
) -> tuple[Mapping[str, Any] | list[Mapping[str, Any]], str]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise ResultValidationError(f"cannot read result artifact {path}: {error}") from error
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ResultValidationError(f"result artifact {path} is not UTF-8") from error
    artifact_sha256 = sha256(raw_bytes).hexdigest()
    if path.suffix.lower() == ".jsonl":
        events: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as error:
                raise ResultValidationError(
                    f"invalid JSONL in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(decoded, Mapping):
                raise ResultValidationError(
                    f"invalid JSONL in {path} at line {line_number}: expected an object"
                )
            events.append(decoded)
        return events, artifact_sha256
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ResultValidationError(f"invalid JSON in {path}: {error.msg}") from error
    if isinstance(decoded, Mapping):
        return decoded, artifact_sha256
    if isinstance(decoded, list):
        return decoded, artifact_sha256
    raise ResultValidationError(f"invalid JSON in {path}: expected an object or array")


def _events_and_defaults(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    supplied: _ArtifactDefaults,
) -> tuple[tuple[Mapping[str, Any], ...], _ArtifactDefaults]:
    if isinstance(payload, Mapping):
        if "events" not in payload:
            return (payload,), supplied
        raw_events = payload["events"]
        if isinstance(raw_events, (str, bytes, Mapping)) or not isinstance(raw_events, Iterable):
            raise ResultValidationError("artifact.events must be an array of event objects")
        study = payload.get("study", {})
        if study is None:
            study = {}
        study_map = _mapping(study, "artifact.study")
        artifact_stage = _first_present(payload, study_map, names=("study_stage", "stage"))
        artifact_model = _first_present(
            payload,
            study_map,
            names=("requested_model_identifier", "requested_model", "model_id", "model"),
        )
        artifact_prompt = _first_present(payload, study_map, names=("prompt_version",))
        merged = _ArtifactDefaults(
            study_stage=_merge_default(
                supplied.study_stage,
                _optional_stage(artifact_stage, "artifact study stage"),
                "study stage",
            ),
            requested_model_identifier=_merge_default(
                supplied.requested_model_identifier,
                _optional_text(
                    artifact_model, "artifact requested model identifier"
                ),
                "requested model identifier",
            ),
            prompt_version=_merge_default(
                supplied.prompt_version,
                _optional_text(artifact_prompt, "artifact prompt_version"),
                "prompt_version",
            ),
            artifact_source=supplied.artifact_source,
            source_artifact_sha256=supplied.source_artifact_sha256,
        )
        return tuple(raw_events), merged
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Iterable):
        raise ResultValidationError("result payload must be an event object or iterable")
    return tuple(payload), supplied


def _parse_focal_event(
    event: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    index: int,
    defaults: _ArtifactDefaults,
) -> FocalDecision:
    prefix = f"event {index}"
    condition_id = _required_text(data.get("condition_id"), f"{prefix}.data.condition_id")
    scenario, world, context, surface = _condition_factors(condition_id, prefix)
    _cross_check_factor(data, "scenario", scenario, prefix)
    _cross_check_factor(data, "social_world", world.value, prefix)
    _cross_check_factor(data, "context_condition", context.value, prefix)
    _cross_check_factor(data, "surface_version", surface.value, prefix)

    stage_value = _coalesced_value(event, data, ("study_stage", "stage"))
    stage = _required_defaulted(
        _optional_stage(stage_value, f"{prefix} study stage"),
        defaults.study_stage,
        f"{prefix}: study_stage is required",
    )
    prompt_version = _required_defaulted(
        _optional_text(
            _coalesced_value(event, data, ("prompt_version",)),
            f"{prefix} prompt_version",
        ),
        defaults.prompt_version,
        f"{prefix}: prompt_version is required",
    )
    provider_metadata = _mapping(
        data.get("provider_metadata"), f"{prefix}.data.provider_metadata"
    )
    explicit_requested_model = _optional_text(
        _coalesced_value(
            event,
            data,
            ("requested_model_identifier", "requested_model", "model_id", "model"),
        ),
        f"{prefix} requested model identifier",
    )
    provider_requested_model = _optional_text(
        _first_present(
            provider_metadata,
            names=("requested_model_identifier", "requested_model"),
        ),
        f"{prefix} provider requested model identifier",
    )
    try:
        requested_model_value = _merge_default(
            explicit_requested_model,
            provider_requested_model,
            "requested-model aliases",
        )
    except ResultValidationError as error:
        raise ResultValidationError(f"{prefix}: {error}") from error
    requested_model = _required_defaulted(
        requested_model_value,
        defaults.requested_model_identifier,
        f"{prefix}: requested_model_identifier is required",
    )
    explicit_returned_model = _optional_text(
        _coalesced_value(
            event,
            data,
            ("returned_model_identifier", "returned_model"),
        ),
        f"{prefix} returned model identifier",
    )
    provider_returned_model = _optional_text(
        _first_present(
            provider_metadata,
            names=(
                "returned_model_identifier",
                "returned_model",
                "model_id",
                "model",
                "model_name",
            ),
        ),
        f"{prefix} provider returned model identifier",
    )
    try:
        returned_model = _required_defaulted(
            explicit_returned_model,
            provider_returned_model,
            f"{prefix}: returned_model_identifier is required",
        )
    except ResultValidationError as error:
        raise ResultValidationError(f"{prefix}: {error}") from error

    raw_prompt = _aliased_text(data, "raw_prompt", "prompt", prefix)
    prompt_sha256 = _required_sha256(
        data.get("prompt_sha256"), f"{prefix}.data.prompt_sha256"
    )
    computed_prompt_sha256 = sha256(raw_prompt.encode("utf-8")).hexdigest()
    if prompt_sha256 != computed_prompt_sha256:
        raise ResultValidationError(
            f"{prefix}.data.prompt_sha256 does not match raw_prompt"
        )
    raw_response = _required_string(data.get("raw_response"), f"{prefix}.data.raw_response")
    parse_state = _parse_state(data, prefix)
    raw_parse_method = _required_text(
        data.get("parse_method"), f"{prefix}.data.parse_method"
    )
    try:
        parse_method = ParseMethod(raw_parse_method)
    except ValueError as error:
        raise ResultValidationError(
            f"{prefix}.data.parse_method must be exact_action or failed"
        ) from error
    canonical_parse = parse_model_response(raw_response)
    if parse_state is not (
        ParseState.VALID if canonical_parse.valid else ParseState.INVALID
    ):
        raise ResultValidationError(
            f"{prefix}.data stored parse state disagrees with the canonical strict parser"
        )
    if parse_method is not canonical_parse.method:
        raise ResultValidationError(
            f"{prefix}.data.parse_method disagrees with the canonical strict parser"
        )
    raw_action = data.get("action")
    if raw_action is None:
        action = None
    else:
        try:
            action = DomainAction(raw_action)
        except (TypeError, ValueError) as error:
            raise ResultValidationError(
                f"{prefix}.data.action must be null, TRANSFER, or RETAIN"
            ) from error
    if action is not canonical_parse.action:
        raise ResultValidationError(
            f"{prefix}.data normalized action disagrees with the canonical strict parser"
        )

    call_id = _required_text(data.get("call_id"), f"{prefix}.data.call_id")
    replicate_id = _required_text(
        data.get("replicate_id"), f"{prefix}.data.replicate_id"
    )
    episode_id = _required_text(data.get("episode_id"), f"{prefix}.data.episode_id")
    seed = _required_int(data.get("seed"), f"{prefix}.data.seed")
    if episode_id != call_id:
        raise ResultValidationError(
            f"{prefix}.data.episode_id must equal call_id {call_id!r}"
        )
    attempts = _required_int(data.get("attempts"), f"{prefix}.data.attempts", minimum=1)
    provider_attempts = _required_int(
        data.get("provider_attempts"),
        f"{prefix}.data.provider_attempts",
        minimum=1,
    )
    parse_note = _required_text(data.get("parse_note"), f"{prefix}.data.parse_note")
    content_attempts = _content_attempts(
        data.get("content_attempts"),
        attempts=attempts,
        raw_prompt=raw_prompt,
        raw_response=raw_response,
        prefix=prefix,
    )
    timestamp_value = _coalesced_value(event, data, ("timestamp", "created_at"))
    timestamp = (
        _optional_text(timestamp_value, f"{prefix} timestamp")
        or content_attempts[-1].completed_at_utc
    )
    frozen_metadata = _frozen_mapping(
        provider_metadata, f"{prefix}.data.provider_metadata"
    )
    if content_attempts[-1].provider_metadata != frozen_metadata:
        raise ResultValidationError(
            f"{prefix}.data.provider_metadata must match the final content attempt"
        )
    counted_provider_attempts = sum(
        _attempt_provider_count(attempt.provider_metadata, prefix)
        for attempt in content_attempts
    )
    if provider_attempts != counted_provider_attempts:
        raise ResultValidationError(
            f"{prefix}.data.provider_attempts must equal the content-attempt total "
            f"({counted_provider_attempts})"
        )
    return FocalDecision(
        study_stage=stage,
        condition_id=condition_id,
        scenario=scenario,
        social_world=world,
        context_condition=context,
        surface_version=surface,
        requested_model_identifier=requested_model,
        returned_model_identifier=returned_model,
        prompt_version=prompt_version,
        raw_prompt=raw_prompt,
        prompt_sha256=prompt_sha256,
        raw_response=raw_response,
        parse_state=parse_state,
        parse_method=parse_method,
        action=action,
        parse_note=parse_note,
        call_id=call_id,
        replicate_id=replicate_id,
        episode_id=episode_id,
        seed=seed,
        attempts=attempts,
        provider_attempts=provider_attempts,
        provider_metadata=frozen_metadata,
        content_attempts=content_attempts,
        artifact_source=defaults.artifact_source,
        source_artifact_sha256=_required_sha256(
            defaults.source_artifact_sha256, "source artifact SHA-256"
        ),
        timestamp=timestamp,
    )


def _content_attempts(
    value: Any,
    *,
    attempts: int,
    raw_prompt: str,
    raw_response: str,
    prefix: str,
) -> tuple[ContentAttempt, ...]:
    label = f"{prefix}.data.content_attempts"
    if not isinstance(value, list):
        raise ResultValidationError(f"{label} must be a list")
    if len(value) != attempts:
        raise ResultValidationError(
            f"{label} length must equal data.attempts ({attempts})"
        )
    parsed: list[ContentAttempt] = []
    required = {
        "request_prompt",
        "raw_response",
        "requested_at_utc",
        "completed_at_utc",
        "provider_metadata",
    }
    for attempt_index, raw_attempt in enumerate(value):
        attempt_label = f"{label}[{attempt_index}]"
        attempt_map = _mapping(raw_attempt, attempt_label)
        request_prompt = _required_nonempty_string(
            attempt_map.get("request_prompt"), f"{attempt_label}.request_prompt"
        )
        attempt_response = _required_string(
            attempt_map.get("raw_response"), f"{attempt_label}.raw_response"
        )
        requested_at = _required_utc_timestamp(
            attempt_map.get("requested_at_utc"), f"{attempt_label}.requested_at_utc"
        )
        completed_at = _required_utc_timestamp(
            attempt_map.get("completed_at_utc"), f"{attempt_label}.completed_at_utc"
        )
        if _parse_utc_timestamp(requested_at) > _parse_utc_timestamp(completed_at):
            raise ResultValidationError(
                f"{attempt_label} completes before it was requested"
            )
        attempt_provider = _frozen_mapping(
            attempt_map.get("provider_metadata"), f"{attempt_label}.provider_metadata"
        )
        attempt_number_value = attempt_map.get("attempt")
        attempt_number = (
            None
            if attempt_number_value is None
            else _required_int(attempt_number_value, f"{attempt_label}.attempt", minimum=1)
        )
        if attempt_number is not None and attempt_number != attempt_index + 1:
            raise ResultValidationError(
                f"{attempt_label}.attempt must equal {attempt_index + 1}"
            )
        extras = _frozen_mapping(
            {key: item for key, item in attempt_map.items() if key not in required | {"attempt"}},
            f"{attempt_label}.extra_fields",
        )
        parsed.append(
            ContentAttempt(
                request_prompt=request_prompt,
                raw_response=attempt_response,
                requested_at_utc=requested_at,
                completed_at_utc=completed_at,
                provider_metadata=attempt_provider,
                attempt=attempt_number,
                extra_fields=extras,
            )
        )
    if parsed[0].request_prompt != raw_prompt:
        raise ResultValidationError(
            f"{label}[0].request_prompt must equal the event raw_prompt"
        )
    if parsed[-1].raw_response != raw_response:
        raise ResultValidationError(
            f"{label} final raw_response must equal the event raw_response"
        )
    return tuple(parsed)


def _attempt_provider_count(
    metadata: tuple[tuple[str, FrozenJSON], ...], prefix: str
) -> int:
    by_name = dict(metadata)
    value = by_name.get("provider_attempts", 1)
    if type(value) is not int or value < 1:
        raise ResultValidationError(
            f"{prefix}.data.content_attempts provider_attempts must be a positive integer"
        )
    return value


def _condition_factors(
    condition_id: str, prefix: str
) -> tuple[str, SocialWorld, ContextCondition, SurfaceVersion]:
    parts = condition_id.split("/")
    if len(parts) != 4 or any(not part or part != part.strip() for part in parts):
        raise ResultValidationError(
            f"{prefix}.data.condition_id must be scenario/world/context/surface"
        )
    scenario, raw_world, raw_context, raw_surface = parts
    try:
        return (
            scenario,
            SocialWorld(raw_world),
            ContextCondition(raw_context),
            SurfaceVersion(raw_surface),
        )
    except ValueError as error:
        raise ResultValidationError(
            f"{prefix}.data.condition_id contains an unknown hospital factor level"
        ) from error


def _parse_state(data: Mapping[str, Any], prefix: str) -> ParseState:
    state_value = data.get("parse_state")
    valid_value = data.get("valid_parse")
    if state_value is None and type(valid_value) is not bool:
        raise ResultValidationError(
            f"{prefix}.data requires parse_state or boolean valid_parse"
        )
    if state_value is not None:
        try:
            state = ParseState(state_value)
        except (TypeError, ValueError) as error:
            raise ResultValidationError(
                f"{prefix}.data.parse_state must be valid or invalid"
            ) from error
        if valid_value is not None:
            if type(valid_value) is not bool:
                raise ResultValidationError(f"{prefix}.data.valid_parse must be boolean")
            if valid_value != (state is ParseState.VALID):
                raise ResultValidationError(f"{prefix}.data parse-state fields disagree")
        return state
    return ParseState.VALID if valid_value else ParseState.INVALID


def _require_one_prompt_hash_per_condition_version(
    records: Iterable[FocalDecision],
) -> None:
    grouped: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        grouped[(record.condition_id, record.prompt_version)].add(
            (record.prompt_sha256, record.raw_prompt)
        )
    for (condition_id, prompt_version), prompt_values in grouped.items():
        if len(prompt_values) != 1:
            raise ResultValidationError(
                "multiple prompt hashes or texts exist for "
                f"{condition_id}/{prompt_version}"
            )


def _require_consistent_returned_models(records: Iterable[FocalDecision]) -> None:
    by_requested: dict[tuple[StudyStage, str], set[str]] = defaultdict(set)
    for record in records:
        by_requested[
            (record.study_stage, record.requested_model_identifier)
        ].add(record.returned_model_identifier)
    for (stage, requested), returned in by_requested.items():
        if len(returned) != 1:
            raise ResultValidationError(
                "returned model identifiers are inconsistent for "
                f"{stage.value}/{requested}: {sorted(returned)}"
            )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ResultValidationError(f"{label} must be a string")
    return value


def _required_nonempty_string(value: Any, label: str) -> str:
    value = _required_string(value, label)
    if not value.strip():
        raise ResultValidationError(f"{label} must not be empty")
    return value


def _required_text(value: Any, label: str) -> str:
    value = _required_string(value, label)
    if not value or value != value.strip():
        raise ResultValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _required_sha256(value: Any, label: str) -> str:
    digest = _required_text(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ResultValidationError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest"
        )
    return digest


def _canonical_payload_sha256(value: Any) -> str:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ResultValidationError(
            "in-memory result payload must contain only JSON-compatible values"
        ) from error
    return sha256(serialized.encode("utf-8")).hexdigest()


def _required_utc_timestamp(value: Any, label: str) -> str:
    timestamp = _required_text(value, label)
    try:
        parsed = _parse_utc_timestamp(timestamp)
    except ValueError as error:
        raise ResultValidationError(
            f"{label} must be an ISO 8601 UTC timestamp"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResultValidationError(f"{label} must use UTC")
    return timestamp


def _parse_utc_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label)


def _optional_stage(value: Any, label: str) -> StudyStage | None:
    if value is None:
        return None
    try:
        return StudyStage(value)
    except (TypeError, ValueError) as error:
        raise ResultValidationError(
            f"{label} must be development or confirmatory"
        ) from error


def _required_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ResultValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ResultValidationError(f"{label} must be at least {minimum}")
    return value


def _required_defaulted(value: Any, default: Any, missing_message: str) -> Any:
    if value is not None and default is not None and value != default:
        label = missing_message.split(":", 1)[0]
        raise ResultValidationError(f"{label} conflicts with artifact defaults")
    resolved = value if value is not None else default
    if resolved is None:
        raise ResultValidationError(missing_message)
    return resolved


def _merge_default(first: Any, second: Any, label: str) -> Any:
    if first is not None and second is not None and first != second:
        raise ResultValidationError(f"conflicting {label} defaults")
    return first if first is not None else second


def _first_present(*mappings: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    found: list[Any] = []
    for mapping in mappings:
        for name in names:
            if name in mapping and mapping[name] is not None:
                found.append(mapping[name])
    if not found:
        return None
    if any(value != found[0] for value in found[1:]):
        raise ResultValidationError(f"conflicting aliases for {names[0]}")
    return found[0]


def _coalesced_value(
    event: Mapping[str, Any], data: Mapping[str, Any], names: tuple[str, ...]
) -> Any:
    return _first_present(data, event, names=names)


def _aliased_text(data: Mapping[str, Any], primary: str, alternate: str, prefix: str) -> str:
    value = _first_present(data, names=(primary, alternate))
    return _required_nonempty_string(value, f"{prefix}.data.{primary}")


def _cross_check_factor(
    data: Mapping[str, Any], name: str, expected: str, prefix: str
) -> None:
    if name not in data:
        return
    actual = _required_text(data[name], f"{prefix}.data.{name}")
    if actual != expected:
        raise ResultValidationError(
            f"{prefix}.data.{name} conflicts with condition_id"
        )


def _freeze_json(value: Any, label: str) -> FrozenJSON:
    if value is None or isinstance(value, (str, bool)) or type(value) in (int, float):
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item, f"{label}[]") for item in value)
    if isinstance(value, Mapping):
        items: list[tuple[str, FrozenJSON]] = []
        if any(not isinstance(key, str) for key in value):
            raise ResultValidationError(f"{label} keys must be strings")
        for key, item in sorted(value.items()):
            items.append((key, _freeze_json(item, f"{label}.{key}")))
        return tuple(items)
    raise ResultValidationError(f"{label} must contain only JSON-compatible values")


def _frozen_mapping(value: Any, label: str) -> tuple[tuple[str, FrozenJSON], ...]:
    mapping = _mapping(value, label)
    if any(not isinstance(key, str) for key in mapping):
        raise ResultValidationError(f"{label} keys must be strings")
    return tuple(
        (key, _freeze_json(item, f"{label}.{key}"))
        for key, item in sorted(mapping.items())
    )
