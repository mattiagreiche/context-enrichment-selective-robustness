"""Focal model and hidden-counterpart agents for SiliSocS."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from collections.abc import Mapping
from typing import Any

from silisocs.agents.base_agent import Agent
from silisocs.runtime.types import ActionOutput, ActionSpec

from .domain import DomainAction
from .parsing import parse_model_response
from .sampling import SamplingSpec


class FocalHospitalAgent(Agent):
    def __init__(
        self,
        model: Any,
        name: str,
        *,
        model_identifier: str,
        sampling_spec: Mapping[str, Any] | SamplingSpec,
        **extra_params: Any,
    ) -> None:
        super().__init__(model)
        del extra_params
        self._name = str(name)
        self._model_identifier = str(model_identifier).strip()
        self._sampling = (
            sampling_spec
            if isinstance(sampling_spec, SamplingSpec)
            else SamplingSpec.from_mapping(sampling_spec)
        )
        if not self._name or not self._model_identifier:
            raise ValueError("agent name and model_identifier are required")

    @property
    def name(self) -> str:
        return self._name

    def observe(self, observation: str) -> None:
        del observation

    def act(self, action_spec: ActionSpec) -> ActionOutput:
        if not isinstance(action_spec, ActionSpec):
            raise TypeError("action_spec must be a SiliSocS ActionSpec")
        requested_at = datetime.now(timezone.utc).isoformat()
        raw = str(
            self.model.sample_text(
                action_spec.prompt,
                temperature=self._sampling.temperature,
                max_tokens=self._sampling.max_output_tokens,
                timeout=self._sampling.timeout_seconds,
                seed=None,
                **self._sampling.provider_request_options(),
            )
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        metadata = _last_call_metadata(self.model)
        if not metadata and _is_scripted_model(self.model):
            metadata = {
                "provider": "scripted",
                "provider_attempts": 1,
                "transport_max_retries": 0,
                "requested_model_identifier": self._model_identifier,
                "returned_model_identifier": self._model_identifier,
                "model_id": self._model_identifier,
                "sampling_spec": self._sampling.as_dict(),
            }
        requested_model = metadata.get(
            "requested_model_identifier", metadata.get("requested_model")
        )
        returned_model = metadata.get(
            "returned_model_identifier",
            metadata.get("returned_model", metadata.get("model_id")),
        )
        if requested_model is None and _is_scripted_model(self.model):
            requested_model = self._model_identifier
        if returned_model is None and _is_scripted_model(self.model):
            returned_model = self._model_identifier
        if requested_model != self._model_identifier or not isinstance(
            returned_model, str
        ):
            raise ValueError("provider model identifiers disagree with the focal agent")
        metadata["requested_model_identifier"] = requested_model
        metadata["returned_model_identifier"] = returned_model
        parsed = parse_model_response(raw)
        content_attempts = [
            {
                "attempt": 1,
                "request_prompt": action_spec.prompt,
                "request_prompt_sha256": _text_sha256(action_spec.prompt),
                "raw_response": raw,
                "requested_at_utc": requested_at,
                "completed_at_utc": completed_at,
                "provider_metadata": deepcopy(metadata),
            }
        ]
        return ActionOutput.structured_response(
            _payload(
                action=parsed.action,
                prompt=action_spec.prompt,
                raw_attempts=[raw],
                content_attempts=content_attempts,
                valid_parse=parsed.valid,
                parse_method=parsed.method.value,
                parse_note=(
                    parsed.method.value
                    if parsed.valid
                    else "invalid_after_0_content_retries"
                ),
                provider_attempts=int(metadata.get("provider_attempts", 1) or 1),
                metadata=metadata,
            ),
            raw=raw,
        )


class HiddenCounterpartAgent(Agent):
    def __init__(
        self,
        model: Any,
        name: str,
        *,
        action: str = DomainAction.RETAIN.value,
        **extra_params: Any,
    ) -> None:
        super().__init__(model)
        del extra_params
        self._name = str(name)
        self._action = DomainAction(action)

    @property
    def name(self) -> str:
        return self._name

    def observe(self, observation: str) -> None:
        del observation

    def act(self, action_spec: ActionSpec) -> ActionOutput:
        del action_spec
        return ActionOutput.structured_response(
            {"action": self._action.value, "source": "scripted_counterpart"}
        )


def _last_call_metadata(model: Any) -> dict[str, Any]:
    getter = getattr(model, "get_last_call_metadata", None)
    value = getter() if callable(getter) else getattr(model, "last_call_metadata", None)
    return deepcopy(dict(value)) if isinstance(value, dict) else {}


def _payload(
    *,
    action: DomainAction | None,
    prompt: str,
    raw_attempts: list[str],
    content_attempts: list[dict[str, Any]],
    valid_parse: bool,
    parse_method: str,
    parse_note: str,
    provider_attempts: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if valid_parse != (action is not None):
        raise ValueError("parse validity and normalized action disagree")
    payload = {
        "action": action.value if action is not None else None,
        "source": "model",
        "prompt": prompt,
        "prompt_sha256": _text_sha256(prompt),
        "raw_response": raw_attempts[-1],
        "raw_attempts": list(raw_attempts),
        "content_attempts": deepcopy(content_attempts),
        "valid_parse": valid_parse,
        "parse_method": parse_method,
        "parse_note": parse_note,
        "attempts": len(raw_attempts),
        "provider_attempts": provider_attempts,
        "provider_metadata": deepcopy(metadata),
    }
    return payload


def _text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _is_scripted_model(model: Any) -> bool:
    return type(model).__name__ == "ScriptedLanguageModel" and type(
        model
    ).__module__.startswith("silisocs.")
