"""SiliSocS components for one hospital decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from silisocs.environments.gm.components.base import (
    ActionPromptComponent,
    NextActingComponent,
    ObservationComponent,
    ResolveComponent,
    UpdateComponent,
)
from silisocs.runtime.types import ActionOutput, ActionSpec, OutputType

from .domain import DecisionRecord, DomainAction
from .prompts import render_prompt


class HospitalNextActing(NextActingComponent):
    def __init__(
        self, *, agent_names: list[str] = (), context: Any = None, **kwargs: Any
    ) -> None:
        super().__init__()
        del kwargs
        self._agent_names = [str(name) for name in agent_names]
        self._context = context

    def acting_agent_names(self) -> list[str]:
        backend = getattr(self._context, "backend", None)
        step = backend.current_step() if backend is not None else 0
        return list(self._agent_names) if step == 0 else []


class HospitalActionPrompt(ActionPromptComponent):
    def __init__(self, *, backend: Any, **kwargs: Any) -> None:
        super().__init__()
        del kwargs
        self._backend = backend

    def action_prompt(self, agent_name: str) -> ActionSpec:
        if agent_name not in self._backend.agent_names:
            raise ValueError(f"unknown actor {agent_name!r}")
        rendered = render_prompt(self._backend.cell)
        return ActionSpec(
            prompt=rendered.text,
            output_type=OutputType.TEXT,
            tag="hospital_supply_decision",
            extra_args={
                "condition_id": self._backend.condition_id,
                "fact_ids": list(rendered.fact_ids),
                "prompt_version": self._backend.cell.prompt_version,
            },
        )


class HospitalObservation(ObservationComponent):
    def __init__(self, *, backend: Any, **kwargs: Any) -> None:
        super().__init__()
        del kwargs
        self._backend = backend

    def make_observation(self, agent_name: str) -> str:
        del agent_name
        return ""


class HospitalResolve(ResolveComponent):
    def __init__(self, *, backend: Any, **kwargs: Any) -> None:
        super().__init__()
        del kwargs
        self._backend = backend

    def resolve_action(self, agent_name: str, action: ActionOutput) -> str:
        if (
            not isinstance(action, ActionOutput)
            or action.output_type is not OutputType.STRUCTURED
        ):
            raise TypeError("hospital resolve requires normalized structured output")
        if not isinstance(action.structured, Mapping):
            raise ValueError("structured action payload is required")
        payload = dict(action.structured)
        source = str(payload.get("source", ""))
        if source == "scripted_counterpart":
            normalized = DomainAction(payload.get("action"))
            decision = DecisionRecord(action=normalized, source=source)
        elif source == "model":
            valid_parse = payload.get("valid_parse")
            if type(valid_parse) is not bool:
                raise ValueError("model payload requires boolean valid_parse")
            raw_action = payload.get("action")
            if valid_parse:
                normalized = DomainAction(raw_action)
            else:
                if raw_action is not None:
                    raise ValueError("invalid model payload cannot carry an action")
                normalized = None
            metadata = payload.get("provider_metadata")
            decision = DecisionRecord(
                action=normalized,
                source=source,
                prompt=_optional_string(payload.get("prompt")),
                prompt_sha256=_optional_string(payload.get("prompt_sha256")),
                raw_response=_optional_string(payload.get("raw_response")),
                valid_parse=valid_parse,
                parse_method=str(payload.get("parse_method", "")),
                parse_note=str(payload.get("parse_note", "")),
                attempts=int(payload.get("attempts", 0)),
                provider_attempts=int(payload.get("provider_attempts", 0)),
                provider_metadata=tuple(dict(metadata or {}).items()),
            )
        else:
            raise ValueError("structured output has an unknown source")
        attempts = payload.get("content_attempts")
        return self._backend.record_decision(
            agent_name,
            decision,
            content_attempts=list(attempts) if isinstance(attempts, list) else [],
        )


class HospitalUpdate(UpdateComponent):
    def __init__(self, *, backend: Any, **kwargs: Any) -> None:
        super().__init__()
        del kwargs
        self._backend = backend

    def update(self, *, step: int, agents: list[Any], context: Any = None) -> None:
        self._backend.update(
            step=step,
            agent_names=[str(agent.name) for agent in agents],
            context=context,
        )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
