"""Strict response parsing for the hospital domain actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import DomainAction


class ParseMethod(StrEnum):
    EXACT_ACTION = "exact_action"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    action: DomainAction | None
    method: ParseMethod

    @property
    def valid(self) -> bool:
        return self.action is not None


def parse_model_response(raw_text: str) -> ParsedResponse:
    """Parse one uppercase action token, allowing surrounding whitespace only."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    candidate = raw_text.strip()
    try:
        action = DomainAction(candidate)
    except ValueError:
        return ParsedResponse(action=None, method=ParseMethod.FAILED)
    return ParsedResponse(action=action, method=ParseMethod.EXACT_ACTION)
