"""Canonical sampling configuration shared by every live-call layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Any


class TopPMode(StrEnum):
    """Whether ``top_p`` is explicit or deliberately left to the provider."""

    OMITTED_PROVIDER_DEFAULT = "omitted_provider_default"
    EXPLICIT = "explicit"


class ReasoningEffort(StrEnum):
    """Provider reasoning effort requested for every focal call."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class LogprobMode(StrEnum):
    """Whether token log probabilities are part of the measurement protocol."""

    OMITTED = "omitted"
    FIRST_ACTION_BRANCH = "first_action_branch"


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """The single authoritative sampling specification for a focal call.

    ``top_p`` is represented by a mode as well as a value so an omitted
    provider default is never mistaken for a known numerical setting.
    """

    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    reasoning_effort: ReasoningEffort
    top_p_mode: TopPMode = TopPMode.OMITTED_PROVIDER_DEFAULT
    top_p: float | None = None
    logprob_mode: LogprobMode = LogprobMode.OMITTED
    top_logprobs: int | None = None
    transport_retries: int = 0
    content_retries: int = 0

    def __post_init__(self) -> None:
        temperature = _finite_number(self.temperature, "temperature")
        timeout = _finite_number(self.timeout_seconds, "timeout_seconds")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if isinstance(self.max_output_tokens, bool) or not isinstance(
            self.max_output_tokens, int
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if not isinstance(self.reasoning_effort, ReasoningEffort):
            raise TypeError("reasoning_effort must be a ReasoningEffort")
        if not isinstance(self.top_p_mode, TopPMode):
            raise TypeError("top_p_mode must be a TopPMode")
        if not isinstance(self.logprob_mode, LogprobMode):
            raise TypeError("logprob_mode must be a LogprobMode")
        if self.top_p_mode is TopPMode.OMITTED_PROVIDER_DEFAULT:
            if self.top_p is not None:
                raise ValueError("omitted_provider_default top_p cannot have a value")
        else:
            value = _finite_number(self.top_p, "top_p")
            if not 0 < value <= 1:
                raise ValueError("explicit top_p must be greater than 0 and at most 1")
        if self.logprob_mode is LogprobMode.OMITTED:
            if self.top_logprobs is not None:
                raise ValueError("omitted logprobs cannot set top_logprobs")
        elif (
            isinstance(self.top_logprobs, bool)
            or not isinstance(self.top_logprobs, int)
            or not 1 <= self.top_logprobs <= 20
        ):
            raise ValueError(
                "first_action_branch logprobs require top_logprobs between 1 and 20"
            )
        for name, value in (
            ("transport_retries", self.transport_retries),
            ("content_retries", self.content_retries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.transport_retries != 0 or self.content_retries != 0:
            raise ValueError(
                "live sampling requires zero transport and content retries"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SamplingSpec:
        """Parse the JSON/YAML representation without accepting stray fields."""

        if not isinstance(value, Mapping):
            raise TypeError("sampling_spec must be a mapping")
        allowed = {
            "temperature",
            "reasoning_effort",
            "top_p",
            "max_output_tokens",
            "timeout_seconds",
            "transport_retries",
            "content_retries",
            "response_api",
            "store",
            "provider_seed",
            "logprobs",
        }
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError("unknown sampling_spec fields: " + ", ".join(unknown))
        if value.get("response_api", "responses") != "responses":
            raise ValueError("the hospital pilot requires the Responses API")
        if value.get("store", False) is not False:
            raise ValueError("provider response storage must be disabled")
        if value.get("provider_seed", "unsupported_omitted") != "unsupported_omitted":
            raise ValueError("provider seed must remain unsupported and omitted")
        top_p = value.get(
            "top_p", {"mode": TopPMode.OMITTED_PROVIDER_DEFAULT.value, "value": None}
        )
        if not isinstance(top_p, Mapping):
            raise TypeError("sampling_spec.top_p must be a mapping")
        unknown_top_p = sorted(set(top_p).difference({"mode", "value"}))
        if unknown_top_p:
            raise ValueError("unknown top_p fields: " + ", ".join(unknown_top_p))
        logprobs = value.get(
            "logprobs",
            {"mode": LogprobMode.OMITTED.value, "top_logprobs": None},
        )
        if not isinstance(logprobs, Mapping):
            raise TypeError("sampling_spec.logprobs must be a mapping")
        unknown_logprobs = sorted(
            set(logprobs).difference({"mode", "top_logprobs"})
        )
        if unknown_logprobs:
            raise ValueError(
                "unknown logprobs fields: " + ", ".join(unknown_logprobs)
            )
        try:
            logprob_mode = LogprobMode(str(logprobs.get("mode", "")))
        except ValueError as exc:
            raise ValueError("sampling_spec.logprobs.mode is invalid") from exc
        try:
            mode = TopPMode(str(top_p.get("mode", "")))
        except ValueError as exc:
            raise ValueError("sampling_spec.top_p.mode is invalid") from exc
        required = (
            "temperature",
            "reasoning_effort",
            "max_output_tokens",
            "timeout_seconds",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError("sampling_spec is missing: " + ", ".join(missing))
        try:
            reasoning_effort = ReasoningEffort(str(value["reasoning_effort"]))
        except ValueError as exc:
            raise ValueError("sampling_spec.reasoning_effort is invalid") from exc
        return cls(
            temperature=value["temperature"],
            reasoning_effort=reasoning_effort,
            max_output_tokens=value["max_output_tokens"],
            timeout_seconds=value["timeout_seconds"],
            top_p_mode=mode,
            top_p=top_p.get("value"),
            logprob_mode=logprob_mode,
            top_logprobs=logprobs.get("top_logprobs"),
            transport_retries=value.get("transport_retries", 0),
            content_retries=value.get("content_retries", 0),
        )

    def as_dict(self) -> dict[str, Any]:
        value = {
            "temperature": float(self.temperature),
            "reasoning_effort": self.reasoning_effort.value,
            "top_p": {
                "mode": self.top_p_mode.value,
                "value": float(self.top_p) if self.top_p is not None else None,
            },
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": float(self.timeout_seconds),
            "transport_retries": self.transport_retries,
            "content_retries": self.content_retries,
            "response_api": "responses",
            "store": False,
            "provider_seed": "unsupported_omitted",
        }
        if self.logprob_mode is not LogprobMode.OMITTED:
            value["logprobs"] = {
                "mode": self.logprob_mode.value,
                "top_logprobs": self.top_logprobs,
            }
        return value

    def provider_request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "reasoning": {"effort": self.reasoning_effort.value}
        }
        if self.top_p_mode is TopPMode.EXPLICIT:
            assert self.top_p is not None
            options["top_p"] = float(self.top_p)
        if self.logprob_mode is LogprobMode.FIRST_ACTION_BRANCH:
            assert self.top_logprobs is not None
            options["include"] = ["message.output_text.logprobs"]
            options["top_logprobs"] = self.top_logprobs
        return options


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result
