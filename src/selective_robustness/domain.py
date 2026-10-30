"""Pure domain types for the hospital allocation study."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Final


class DomainAction(StrEnum):
    TRANSFER = "TRANSFER"
    RETAIN = "RETAIN"


class StrategicAction(StrEnum):
    COOPERATE = "cooperate"
    DEFECT = "defect"


class SocialWorld(StrEnum):
    REGIONAL_MUTUAL_AID = "regional_mutual_aid"
    LOCAL_EMERGENCY_RESERVE = "local_emergency_reserve"


class ContextCondition(StrEnum):
    MINIMAL = "minimal"
    NEUTRAL_PADDED = "neutral_padded"
    GENERIC_BACKGROUND = "generic_background"
    DECISION_GROUNDED = "decision_grounded"
    NORMATIVE_GROUNDED = "normative_grounded"


class SurfaceVersion(StrEnum):
    CANONICAL_PROSE = "canonical_prose"
    BULLETS = "bullets"
    KEY_VALUE_TABLE = "key_value_table"
    ALTERNATIVE_PROSE = "alternative_prose"


ACTION_MAPPING: Final[dict[DomainAction, StrategicAction]] = {
    DomainAction.TRANSFER: StrategicAction.COOPERATE,
    DomainAction.RETAIN: StrategicAction.DEFECT,
}


@dataclass(frozen=True, slots=True)
class Payoffs:
    focal: int
    counterpart: int


PAYOFF_MATRIX: Final[dict[tuple[DomainAction, DomainAction], Payoffs]] = {
    (DomainAction.TRANSFER, DomainAction.TRANSFER): Payoffs(3, 3),
    (DomainAction.TRANSFER, DomainAction.RETAIN): Payoffs(0, 5),
    (DomainAction.RETAIN, DomainAction.TRANSFER): Payoffs(5, 0),
    (DomainAction.RETAIN, DomainAction.RETAIN): Payoffs(1, 1),
}


def resolve_payoffs(focal: DomainAction, counterpart: DomainAction) -> Payoffs:
    if not isinstance(focal, DomainAction) or not isinstance(counterpart, DomainAction):
        raise TypeError("both choices must be DomainAction values")
    return PAYOFF_MATRIX[(focal, counterpart)]


@dataclass(frozen=True, slots=True)
class PromptCell:
    world: SocialWorld
    context: ContextCondition
    surface: SurfaceVersion
    scenario: str = "hospital"
    prompt_version: str = "pilot_v4"

    def __post_init__(self) -> None:
        if not isinstance(self.world, SocialWorld):
            raise TypeError("world must be a SocialWorld")
        if not isinstance(self.context, ContextCondition):
            raise TypeError("context must be a ContextCondition")
        if not isinstance(self.surface, SurfaceVersion):
            raise TypeError("surface must be a SurfaceVersion")
        for name in ("scenario", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string")

    @property
    def condition_id(self) -> str:
        return "/".join(
            (self.scenario, self.world.value, self.context.value, self.surface.value)
        )


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    action: DomainAction | None
    source: str
    prompt: str | None = None
    prompt_sha256: str | None = None
    raw_response: str | None = None
    valid_parse: bool | None = None
    parse_method: str = ""
    parse_note: str = ""
    attempts: int = 0
    provider_attempts: int = 0
    provider_metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.source not in {"model", "scripted_counterpart"}:
            raise ValueError("unknown decision source")
        if self.source == "model":
            if type(self.valid_parse) is not bool or self.attempts < 1:
                raise ValueError("model decisions require parse state and attempts")
            if not isinstance(self.prompt, str) or not self.prompt:
                raise ValueError("model decisions require the rendered prompt")
            expected_prompt_hash = sha256(self.prompt.encode("utf-8")).hexdigest()
            if self.prompt_sha256 != expected_prompt_hash:
                raise ValueError(
                    "model decision prompt_sha256 does not match its prompt"
                )
            if not isinstance(self.raw_response, str):
                raise ValueError("model decisions require the final raw response")
            if self.valid_parse and not isinstance(self.action, DomainAction):
                raise ValueError("a valid model parse requires a domain action")
            if not self.valid_parse and self.action is not None:
                raise ValueError("an invalid model parse cannot carry an action")
            expected_method = "exact_action" if self.valid_parse else "failed"
            if self.parse_method != expected_method:
                raise ValueError("model parse_method disagrees with parse validity")
            if not self.parse_note:
                raise ValueError("model decisions require a parse_note")
            if self.provider_attempts < 1:
                raise ValueError(
                    "model decisions require at least one provider attempt"
                )
        else:
            if not isinstance(self.action, DomainAction):
                raise TypeError("scripted decisions require a DomainAction")
            if self.valid_parse is not None or self.attempts != 0:
                raise ValueError("scripted decisions cannot have model parse state")
            if self.prompt_sha256 is not None or self.parse_method:
                raise ValueError(
                    "scripted decisions cannot have model prompt/parse metadata"
                )


FOCAL_AGENT: Final[str] = "Maya"
COUNTERPART_AGENT: Final[str] = "HospitalBManager"
