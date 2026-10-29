"""Credential-free loading and descriptive summaries for focal decisions.

The native SiliSocS ``choose_hospital_action`` event stores the study stage,
factor levels, model-call provenance, and response parse state inside
``event["data"]``.  Its condition identifier is the authoritative source for
the four factorial levels, and the explicit factor fields must agree with that
identifier.  Artifact-level defaults can supply declared study metadata, but
stage and both requested and returned model identities remain explicit in each
validated output record.

All analysis functions are deterministic and operate only on validated,
immutable records.  The canonical strict parser is rerun over every final raw
response; any disagreement with stored state, method, or action is rejected.
Invalid responses remain explicit in every denominator and summary.
"""

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
class PromptHashProvenance:
    """The one accepted prompt hash for a condition and prompt version."""

    condition_id: str
    prompt_version: str
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class CellTransferRate:
    """Explicit valid-only and all-scheduled-call estimands for one cell."""

    study_stage: StudyStage
    scenario: str
    social_world: SocialWorld
    context_condition: ContextCondition
    surface_version: SurfaceVersion
    requested_model_identifier: str
    returned_model_identifier: str
    prompt_version: str
    prompt_sha256: str
    source_artifact_sha256s: tuple[str, ...]
    scheduled_count: int
    observed_count: int
    valid_count: int
    invalid_count: int
    transfer_count: int
    retain_count: int
    conditional_transfer_rate_among_valid_pp: float | None
    invalid_response_rate_over_all_scheduled_pp: float
    transfer_lower_bound_over_all_scheduled_pp: float
    transfer_upper_bound_over_all_scheduled_pp: float


@dataclass(frozen=True, slots=True)
class SurfaceDifference:
    """Alternative-minus-canonical contrasts for one surface."""

    surface_version: SurfaceVersion
    canonical_prompt_sha256: str
    comparison_prompt_sha256: str
    source_artifact_sha256s: tuple[str, ...]
    absolute_conditional_transfer_rate_difference_pp: float | None
    signed_invalid_response_rate_difference_pp: float
    absolute_invalid_response_rate_difference_pp: float


@dataclass(frozen=True, slots=True)
class SurfaceSensitivity:
    """Canonical-versus-other surface sensitivity for one world/context."""

    study_stage: StudyStage
    scenario: str
    social_world: SocialWorld
    context_condition: ContextCondition
    requested_model_identifier: str
    returned_model_identifier: str
    prompt_version: str
    prompt_provenance: tuple[PromptHashProvenance, ...]
    source_artifact_sha256s: tuple[str, ...]
    canonical_conditional_transfer_rate_among_valid_pp: float | None
    canonical_invalid_response_rate_pp: float
    differences: tuple[SurfaceDifference, ...]
    average_absolute_conditional_transfer_rate_difference_pp: float | None
    average_absolute_invalid_response_rate_difference_pp: float


@dataclass(frozen=True, slots=True)
class WorldDifference:
    """Regional-minus-local contrasts for one surface."""

    surface_version: SurfaceVersion
    regional_prompt_sha256: str
    local_prompt_sha256: str
    source_artifact_sha256s: tuple[str, ...]
    regional_conditional_transfer_rate_among_valid_pp: float | None
    local_conditional_transfer_rate_among_valid_pp: float | None
    signed_conditional_transfer_rate_difference_pp: float | None
    regional_invalid_response_rate_pp: float
    local_invalid_response_rate_pp: float
    signed_invalid_response_rate_difference_pp: float


@dataclass(frozen=True, slots=True)
class SubstantiveResponse:
    """Regional-minus-local response averaged over all four surfaces."""

    study_stage: StudyStage
    scenario: str
    context_condition: ContextCondition
    requested_model_identifier: str
    returned_model_identifier: str
    prompt_version: str
    prompt_provenance: tuple[PromptHashProvenance, ...]
    source_artifact_sha256s: tuple[str, ...]
    differences: tuple[WorldDifference, ...]
    average_signed_conditional_transfer_rate_difference_pp: float | None
    average_signed_invalid_response_rate_difference_pp: float


@dataclass(frozen=True, slots=True)
class DescriptiveSummary:
    prompt_provenance: tuple[PromptHashProvenance, ...]
    source_artifact_sha256s: tuple[str, ...]
    cells: tuple[CellTransferRate, ...]
    surface_sensitivity: tuple[SurfaceSensitivity, ...]
    substantive_response: tuple[SubstantiveResponse, ...]


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
    model_id: str | None = None,
    prompt_version: str | None = None,
) -> tuple[FocalDecision, ...]:
    """Load focal decisions from JSON, JSONL, or in-memory event mappings.

    A JSON object may be one event or an artifact wrapper with an ``events``
    list.  Wrapper defaults can be supplied at top level or under ``study``.
    Explicit function arguments are defaults only: any conflicting event or
    artifact value is rejected.
    """

    supplied_requested_model = _merge_default(
        _optional_text(
            requested_model_identifier, "requested_model_identifier argument"
        ),
        _optional_text(model_id, "legacy model_id argument"),
        "requested model identifier",
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


def cell_transfer_rates(
    decisions: Iterable[FocalDecision],
    *,
    scheduled_calls: Iterable[Mapping[str, Any]],
) -> tuple[CellTransferRate, ...]:
    """Compute labeled cell estimands after every scheduled call is classified.

    ``scheduled_calls`` contains the immutable manifest schedule rows.  Its
    call IDs must exactly match the terminal focal decisions, and each row's
    condition and expected prompt hash must agree with its decision.  Requiring
    identity-level schedule evidence prevents an incomplete or substituted
    event artifact from masquerading as a complete denominator.
    """

    records = _validated_decisions(decisions)
    _require_single_stage(record.study_stage for record in records)
    _require_one_prompt_hash_per_condition_version(records)
    grouped: dict[tuple[str, ...], list[FocalDecision]] = defaultdict(list)
    for record in records:
        grouped[_decision_group_key(record)].append(record)
    normalized_schedule = _validated_schedule(scheduled_calls, records)

    rows: list[CellTransferRate] = []
    for key in sorted(grouped):
        group = grouped[key]
        observed_count = len(group)
        scheduled_count = normalized_schedule[group[0].condition_id]
        if scheduled_count != observed_count:
            raise ResultValidationError(
                "incomplete scheduled cell: "
                f"{group[0].condition_id} has {observed_count} terminal decisions "
                f"for {scheduled_count} scheduled calls"
            )
        prompt_hashes = {record.prompt_sha256 for record in group}
        if len(prompt_hashes) != 1:
            raise ResultValidationError(
                f"multiple prompt hashes would be combined for "
                f"{group[0].condition_id}/{group[0].prompt_version}"
            )
        prompt_hash = next(iter(prompt_hashes))
        valid = [record for record in group if record.valid_parse]
        transfer_count = sum(record.action is DomainAction.TRANSFER for record in valid)
        invalid_count = observed_count - len(valid)
        retain_count = len(valid) - transfer_count
        rows.append(
            CellTransferRate(
                study_stage=group[0].study_stage,
                scenario=group[0].scenario,
                social_world=group[0].social_world,
                context_condition=group[0].context_condition,
                surface_version=group[0].surface_version,
                requested_model_identifier=group[0].requested_model_identifier,
                returned_model_identifier=group[0].returned_model_identifier,
                prompt_version=group[0].prompt_version,
                prompt_sha256=prompt_hash,
                source_artifact_sha256s=_source_hashes(group),
                scheduled_count=scheduled_count,
                observed_count=observed_count,
                valid_count=len(valid),
                invalid_count=invalid_count,
                transfer_count=transfer_count,
                retain_count=retain_count,
                conditional_transfer_rate_among_valid_pp=(
                    100.0 * transfer_count / len(valid) if valid else None
                ),
                invalid_response_rate_over_all_scheduled_pp=(
                    100.0 * invalid_count / scheduled_count
                ),
                transfer_lower_bound_over_all_scheduled_pp=(
                    100.0 * transfer_count / scheduled_count
                ),
                transfer_upper_bound_over_all_scheduled_pp=(
                    100.0 * (transfer_count + invalid_count) / scheduled_count
                ),
            )
        )
    return tuple(rows)


def compute_surface_sensitivity(
    cells: Iterable[CellTransferRate],
) -> tuple[SurfaceSensitivity, ...]:
    """Compute valid-only transfer and invalid-rate surface contrasts."""

    rows = _validated_cells(cells)
    _require_single_stage(row.study_stage for row in rows)
    grouped: dict[tuple[str, ...], dict[SurfaceVersion, CellTransferRate]] = defaultdict(dict)
    for row in rows:
        key = (
            row.study_stage.value,
            row.scenario,
            row.social_world.value,
            row.context_condition.value,
            row.requested_model_identifier,
            row.returned_model_identifier,
            row.prompt_version,
        )
        if row.surface_version in grouped[key]:
            raise ResultValidationError(f"duplicate cell summary for {'/'.join(key)}")
        grouped[key][row.surface_version] = row

    estimates: list[SurfaceSensitivity] = []
    alternatives = tuple(
        surface for surface in SurfaceVersion if surface is not SurfaceVersion.CANONICAL_PROSE
    )
    expected = set(SurfaceVersion)
    for key in sorted(grouped):
        by_surface = grouped[key]
        _require_complete_levels(by_surface, expected, f"surface-sensitivity group {'/'.join(key)}")
        canonical_row = by_surface[SurfaceVersion.CANONICAL_PROSE]
        canonical_transfer = canonical_row.conditional_transfer_rate_among_valid_pp
        canonical_invalid = canonical_row.invalid_response_rate_over_all_scheduled_pp
        differences = tuple(
            SurfaceDifference(
                surface_version=surface,
                canonical_prompt_sha256=canonical_row.prompt_sha256,
                comparison_prompt_sha256=by_surface[surface].prompt_sha256,
                source_artifact_sha256s=_cell_source_hashes(
                    (canonical_row, by_surface[surface])
                ),
                absolute_conditional_transfer_rate_difference_pp=(
                    _absolute_optional_difference(
                        canonical_transfer,
                        by_surface[surface].conditional_transfer_rate_among_valid_pp,
                    )
                ),
                signed_invalid_response_rate_difference_pp=(
                    by_surface[surface].invalid_response_rate_over_all_scheduled_pp
                    - canonical_invalid
                ),
                absolute_invalid_response_rate_difference_pp=abs(
                    by_surface[surface].invalid_response_rate_over_all_scheduled_pp
                    - canonical_invalid
                ),
            )
            for surface in alternatives
        )
        transfer_differences = tuple(
            item.absolute_conditional_transfer_rate_difference_pp
            for item in differences
        )
        estimates.append(
            SurfaceSensitivity(
                study_stage=canonical_row.study_stage,
                scenario=canonical_row.scenario,
                social_world=canonical_row.social_world,
                context_condition=canonical_row.context_condition,
                requested_model_identifier=(
                    canonical_row.requested_model_identifier
                ),
                returned_model_identifier=canonical_row.returned_model_identifier,
                prompt_version=canonical_row.prompt_version,
                prompt_provenance=_prompt_provenance_from_cells(by_surface.values()),
                source_artifact_sha256s=_cell_source_hashes(by_surface.values()),
                canonical_conditional_transfer_rate_among_valid_pp=canonical_transfer,
                canonical_invalid_response_rate_pp=canonical_invalid,
                differences=differences,
                average_absolute_conditional_transfer_rate_difference_pp=(
                    _average_optional(transfer_differences)
                ),
                average_absolute_invalid_response_rate_difference_pp=(
                    sum(
                        item.absolute_invalid_response_rate_difference_pp
                        for item in differences
                    )
                    / len(differences)
                ),
            )
        )
    return tuple(estimates)


def compute_substantive_response(
    cells: Iterable[CellTransferRate],
) -> tuple[SubstantiveResponse, ...]:
    """Compute regional-minus-local valid-transfer and invalid-rate contrasts."""

    rows = _validated_cells(cells)
    _require_single_stage(row.study_stage for row in rows)
    grouped: dict[
        tuple[str, ...], dict[tuple[SurfaceVersion, SocialWorld], CellTransferRate]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            row.study_stage.value,
            row.scenario,
            row.context_condition.value,
            row.requested_model_identifier,
            row.returned_model_identifier,
            row.prompt_version,
        )
        level = (row.surface_version, row.social_world)
        if level in grouped[key]:
            raise ResultValidationError(f"duplicate cell summary for {'/'.join(key)}")
        grouped[key][level] = row

    expected = {(surface, world) for surface in SurfaceVersion for world in SocialWorld}
    estimates: list[SubstantiveResponse] = []
    for key in sorted(grouped):
        by_level = grouped[key]
        _require_complete_levels(by_level, expected, f"substantive-response group {'/'.join(key)}")
        differences: list[WorldDifference] = []
        for surface in SurfaceVersion:
            regional_row = by_level[(surface, SocialWorld.REGIONAL_MUTUAL_AID)]
            local_row = by_level[(surface, SocialWorld.LOCAL_EMERGENCY_RESERVE)]
            regional_transfer = (
                regional_row.conditional_transfer_rate_among_valid_pp
            )
            local_transfer = local_row.conditional_transfer_rate_among_valid_pp
            regional_invalid = (
                regional_row.invalid_response_rate_over_all_scheduled_pp
            )
            local_invalid = local_row.invalid_response_rate_over_all_scheduled_pp
            differences.append(
                WorldDifference(
                    surface_version=surface,
                    regional_prompt_sha256=regional_row.prompt_sha256,
                    local_prompt_sha256=local_row.prompt_sha256,
                    source_artifact_sha256s=_cell_source_hashes(
                        (regional_row, local_row)
                    ),
                    regional_conditional_transfer_rate_among_valid_pp=(
                        regional_transfer
                    ),
                    local_conditional_transfer_rate_among_valid_pp=local_transfer,
                    signed_conditional_transfer_rate_difference_pp=(
                        _signed_optional_difference(regional_transfer, local_transfer)
                    ),
                    regional_invalid_response_rate_pp=regional_invalid,
                    local_invalid_response_rate_pp=local_invalid,
                    signed_invalid_response_rate_difference_pp=(
                        regional_invalid - local_invalid
                    ),
                )
            )
        first = next(iter(by_level.values()))
        transfer_differences = tuple(
            item.signed_conditional_transfer_rate_difference_pp
            for item in differences
        )
        estimates.append(
            SubstantiveResponse(
                study_stage=first.study_stage,
                scenario=first.scenario,
                context_condition=first.context_condition,
                requested_model_identifier=first.requested_model_identifier,
                returned_model_identifier=first.returned_model_identifier,
                prompt_version=first.prompt_version,
                prompt_provenance=_prompt_provenance_from_cells(by_level.values()),
                source_artifact_sha256s=_cell_source_hashes(by_level.values()),
                differences=tuple(differences),
                average_signed_conditional_transfer_rate_difference_pp=(
                    _average_optional(transfer_differences)
                ),
                average_signed_invalid_response_rate_difference_pp=(
                    sum(
                        item.signed_invalid_response_rate_difference_pp
                        for item in differences
                    )
                    / len(differences)
                ),
            )
        )
    return tuple(estimates)


def summarize_results(
    decisions: Iterable[FocalDecision],
    *,
    scheduled_calls: Iterable[Mapping[str, Any]],
) -> DescriptiveSummary:
    """Produce all prespecified descriptive summaries; no inference is run."""

    records = tuple(decisions)
    cells = cell_transfer_rates(records, scheduled_calls=scheduled_calls)
    return DescriptiveSummary(
        prompt_provenance=_prompt_provenance_from_decisions(records),
        source_artifact_sha256s=_source_hashes(records),
        cells=cells,
        surface_sensitivity=compute_surface_sensitivity(cells),
        substantive_response=compute_substantive_response(cells),
    )


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


def _decision_group_key(record: FocalDecision) -> tuple[str, ...]:
    return (
        record.study_stage.value,
        record.scenario,
        record.social_world.value,
        record.context_condition.value,
        record.surface_version.value,
        record.requested_model_identifier,
        record.returned_model_identifier,
        record.prompt_version,
    )


def _validated_schedule(
    scheduled_calls: Iterable[Mapping[str, Any]],
    records: tuple[FocalDecision, ...],
) -> dict[str, int]:
    if isinstance(scheduled_calls, (str, bytes, Mapping)) or not isinstance(
        scheduled_calls, Iterable
    ):
        raise TypeError("scheduled_calls must contain immutable manifest schedule rows")
    rows = tuple(scheduled_calls)
    if not rows:
        raise ResultValidationError("scheduled_calls must not be empty")
    by_call_id: dict[str, Mapping[str, Any]] = {}
    normalized: dict[str, int] = defaultdict(int)
    replicate_numbers: dict[str, int] = {}
    replicate_conditions: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"scheduled_calls[{index}]")
        call_id = _required_text(
            row.get("call_id"), f"scheduled_calls[{index}].call_id"
        )
        if call_id in by_call_id:
            raise ResultValidationError(f"duplicate scheduled call_id {call_id!r}")
        condition_id = _required_text(
            row.get("condition_id"), f"scheduled_calls[{index}].condition_id"
        )
        _, world, context, surface = _condition_factors(
            condition_id, f"scheduled_calls[{index}]"
        )
        _required_sha256(
            row.get("expected_prompt_sha256"),
            f"scheduled_calls[{index}].expected_prompt_sha256",
        )
        schedule_index = _required_int(
            row.get("schedule_index"),
            f"scheduled_calls[{index}].schedule_index",
            minimum=1,
        )
        if schedule_index != index + 1:
            raise ResultValidationError(
                "scheduled_calls must retain contiguous manifest schedule order"
            )
        replicate_id = _required_text(
            row.get("replicate_id"), f"scheduled_calls[{index}].replicate_id"
        )
        replicate_number = _required_int(
            row.get("replicate_number"),
            f"scheduled_calls[{index}].replicate_number",
            minimum=1,
        )
        _required_int(
            row.get("call_seed"), f"scheduled_calls[{index}].call_seed"
        )
        prior_replicate_number = replicate_numbers.setdefault(
            replicate_id, replicate_number
        )
        if prior_replicate_number != replicate_number:
            raise ResultValidationError(
                f"replicate_id {replicate_id!r} has inconsistent replicate numbers"
            )
        replicate_condition = (replicate_id, condition_id)
        if replicate_condition in replicate_conditions:
            raise ResultValidationError(
                f"replicate {replicate_id!r} schedules {condition_id!r} more than once"
            )
        replicate_conditions.add(replicate_condition)
        _required_text(
            row.get("prompt_version"),
            f"scheduled_calls[{index}].prompt_version",
        )
        expected_factors = {
            "social_world": world.value,
            "context_condition": context.value,
            "surface_version": surface.value,
        }
        for factor_name, expected in expected_factors.items():
            if row.get(factor_name) != expected:
                raise ResultValidationError(
                    f"scheduled_calls[{index}].{factor_name} conflicts with condition_id"
                )
        by_call_id[call_id] = row
        normalized[condition_id] += 1
    observed_by_call_id = {record.call_id: record for record in records}
    scheduled_ids = set(by_call_id)
    observed_ids = set(observed_by_call_id)
    if scheduled_ids != observed_ids:
        missing_terminal = sorted(scheduled_ids - observed_ids)
        unscheduled = sorted(observed_ids - scheduled_ids)
        raise ResultValidationError(
            "scheduled calls and terminal decisions disagree "
            f"(without_terminal={missing_terminal}, unscheduled={unscheduled})"
        )
    for call_id, record in observed_by_call_id.items():
        schedule_row = by_call_id[call_id]
        disagreements: list[str] = []
        if schedule_row["condition_id"] != record.condition_id:
            disagreements.append("condition_id")
        if schedule_row["expected_prompt_sha256"] != record.prompt_sha256:
            disagreements.append("expected_prompt_sha256")
        if schedule_row["prompt_version"] != record.prompt_version:
            disagreements.append("prompt_version")
        if schedule_row["replicate_id"] != record.replicate_id:
            disagreements.append("replicate_id")
        if schedule_row["call_seed"] != record.seed:
            disagreements.append("call_seed")
        if disagreements:
            raise ResultValidationError(
                f"scheduled call {call_id!r} disagrees with its decision: {disagreements}"
            )
    strata = {
        (
            record.study_stage,
            record.scenario,
            record.requested_model_identifier,
            record.returned_model_identifier,
            record.prompt_version,
        )
        for record in records
    }
    if len(strata) != 1:
        raise ResultValidationError(
            "scheduled_calls requires exactly one stage/scenario/model/prompt-version run"
        )
    return dict(normalized)


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


def _prompt_provenance_from_decisions(
    records: Iterable[FocalDecision],
) -> tuple[PromptHashProvenance, ...]:
    values = {
        (record.condition_id, record.prompt_version, record.prompt_sha256)
        for record in records
    }
    return tuple(PromptHashProvenance(*value) for value in sorted(values))


def _prompt_provenance_from_cells(
    cells: Iterable[CellTransferRate],
) -> tuple[PromptHashProvenance, ...]:
    values = {
        (
            "/".join(
                (
                    row.scenario,
                    row.social_world.value,
                    row.context_condition.value,
                    row.surface_version.value,
                )
            ),
            row.prompt_version,
            row.prompt_sha256,
        )
        for row in cells
    }
    return tuple(PromptHashProvenance(*value) for value in sorted(values))


def _source_hashes(records: Iterable[FocalDecision]) -> tuple[str, ...]:
    return tuple(sorted({record.source_artifact_sha256 for record in records}))


def _cell_source_hashes(cells: Iterable[CellTransferRate]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                artifact_hash
                for row in cells
                for artifact_hash in row.source_artifact_sha256s
            }
        )
    )


def _signed_optional_difference(
    first: float | None, second: float | None
) -> float | None:
    if first is None or second is None:
        return None
    return first - second


def _absolute_optional_difference(
    first: float | None, second: float | None
) -> float | None:
    difference = _signed_optional_difference(first, second)
    return None if difference is None else abs(difference)


def _average_optional(values: Iterable[float | None]) -> float | None:
    items = tuple(values)
    if not items or any(value is None for value in items):
        return None
    return sum(value for value in items if value is not None) / len(items)


def _validated_decisions(decisions: Iterable[FocalDecision]) -> tuple[FocalDecision, ...]:
    records = tuple(decisions)
    if not records:
        raise ResultValidationError("at least one focal decision is required")
    if any(not isinstance(record, FocalDecision) for record in records):
        raise TypeError("decisions must contain only FocalDecision values")
    call_ids = [record.call_id for record in records]
    if len(set(call_ids)) != len(call_ids):
        raise ResultValidationError("focal decisions contain duplicate call IDs")
    for index, record in enumerate(records):
        _required_text(record.call_id, f"decision {index} call_id")
        _required_text(record.replicate_id, f"decision {index} replicate_id")
        if record.episode_id != record.call_id:
            raise ResultValidationError(
                f"decision {index} episode_id must equal call_id"
            )
        _required_sha256(record.prompt_sha256, f"decision {index} prompt_sha256")
        _required_sha256(
            record.source_artifact_sha256,
            f"decision {index} source_artifact_sha256",
        )
        if sha256(record.raw_prompt.encode("utf-8")).hexdigest() != record.prompt_sha256:
            raise ResultValidationError(
                f"decision {index} prompt_sha256 does not match raw_prompt"
            )
        parsed = parse_model_response(record.raw_response)
        if parsed.action is not record.action or parsed.method is not record.parse_method:
            raise ResultValidationError(
                f"decision {index} disagrees with the canonical strict parser"
            )
        expected_state = ParseState.VALID if parsed.valid else ParseState.INVALID
        if record.parse_state is not expected_state:
            raise ResultValidationError(
                f"decision {index} parse state disagrees with the canonical strict parser"
            )
    _require_one_prompt_hash_per_condition_version(records)
    _require_consistent_returned_models(records)
    return records


def _validated_cells(cells: Iterable[CellTransferRate]) -> tuple[CellTransferRate, ...]:
    rows = tuple(cells)
    if not rows:
        raise ResultValidationError("at least one cell summary is required")
    if any(not isinstance(row, CellTransferRate) for row in rows):
        raise TypeError("cells must contain only CellTransferRate values")
    for index, row in enumerate(rows):
        _validate_cell(row, index)
    prompt_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    returned_models: dict[tuple[StudyStage, str], set[str]] = defaultdict(set)
    for row in rows:
        condition_id = "/".join(
            (
                row.scenario,
                row.social_world.value,
                row.context_condition.value,
                row.surface_version.value,
            )
        )
        prompt_hashes[(condition_id, row.prompt_version)].add(row.prompt_sha256)
        returned_models[
            (row.study_stage, row.requested_model_identifier)
        ].add(row.returned_model_identifier)
    if any(len(hashes) != 1 for hashes in prompt_hashes.values()):
        raise ResultValidationError(
            "cell summaries contain multiple prompt hashes for one condition/version"
        )
    if any(len(models) != 1 for models in returned_models.values()):
        raise ResultValidationError(
            "cell summaries contain inconsistent returned model identifiers"
        )
    return rows


def _validate_cell(row: CellTransferRate, index: int) -> None:
    prefix = f"cell {index}"
    _required_sha256(row.prompt_sha256, f"{prefix} prompt_sha256")
    if (
        not row.source_artifact_sha256s
        or row.source_artifact_sha256s
        != tuple(sorted(set(row.source_artifact_sha256s)))
    ):
        raise ResultValidationError(
            f"{prefix} source_artifact_sha256s must be sorted and unique"
        )
    for artifact_hash in row.source_artifact_sha256s:
        _required_sha256(artifact_hash, f"{prefix} source artifact SHA-256")
    counts = (
        row.scheduled_count,
        row.observed_count,
        row.valid_count,
        row.invalid_count,
        row.transfer_count,
        row.retain_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ResultValidationError(f"{prefix} counts must be non-negative integers")
    if row.scheduled_count < 1 or row.observed_count != row.scheduled_count:
        raise ResultValidationError(
            f"{prefix} must contain one terminal classification per scheduled call"
        )
    if row.valid_count + row.invalid_count != row.observed_count:
        raise ResultValidationError(f"{prefix} valid and invalid counts disagree")
    if row.transfer_count + row.retain_count != row.valid_count:
        raise ResultValidationError(f"{prefix} action counts disagree with valid_count")
    expected_conditional = (
        100.0 * row.transfer_count / row.valid_count if row.valid_count else None
    )
    expected_rates = (
        (
            row.conditional_transfer_rate_among_valid_pp,
            expected_conditional,
            "conditional transfer rate",
        ),
        (
            row.invalid_response_rate_over_all_scheduled_pp,
            100.0 * row.invalid_count / row.scheduled_count,
            "invalid-response rate",
        ),
        (
            row.transfer_lower_bound_over_all_scheduled_pp,
            100.0 * row.transfer_count / row.scheduled_count,
            "transfer lower bound",
        ),
        (
            row.transfer_upper_bound_over_all_scheduled_pp,
            100.0 * (row.transfer_count + row.invalid_count) / row.scheduled_count,
            "transfer upper bound",
        ),
    )
    for actual, expected, label in expected_rates:
        if not _same_optional_rate(actual, expected):
            raise ResultValidationError(f"{prefix} {label} disagrees with counts")


def _same_optional_rate(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    return abs(first - second) <= 1e-12


def _require_single_stage(stages: Iterable[StudyStage]) -> StudyStage:
    unique = set(stages)
    if len(unique) != 1:
        labels = ", ".join(sorted(stage.value for stage in unique)) or "none"
        raise ResultValidationError(
            f"descriptive summaries require exactly one study stage; found {labels}"
        )
    return next(iter(unique))


def _require_complete_levels(
    observed: Mapping[Any, Any], expected: set[Any], label: str
) -> None:
    actual = set(observed)
    if actual == expected:
        return
    missing = sorted((_level_label(level) for level in expected - actual))
    extra = sorted((_level_label(level) for level in actual - expected))
    raise ResultValidationError(
        f"{label} is incomplete (missing={missing}, extra={extra})"
    )


def _level_label(level: Any) -> str:
    if isinstance(level, tuple):
        return "/".join(_level_label(item) for item in level)
    return level.value if isinstance(level, StrEnum) else str(level)


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


def _aliased_text(data: Mapping[str, Any], primary: str, legacy: str, prefix: str) -> str:
    value = _first_present(data, names=(primary, legacy))
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
