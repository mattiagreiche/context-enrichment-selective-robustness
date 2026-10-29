"""Authoritative single-decision hospital backend for SiliSocS."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from silisocs.environments.backends.base import BackendApp

from .domain import DecisionRecord, DomainAction, PromptCell, resolve_payoffs


class HospitalBackend(BackendApp):
    provides_checkpoint_state = False
    supports_action_replay = False

    def __init__(
        self,
        *,
        action_logger: Any = None,
        app_description: str = "",
        db_path: str | None = None,
        perform_operations: bool = False,
        call_id: str,
        replicate_id: str,
        episode_id: str,
        condition_id: str,
        seed: int,
        social_world: str,
        context_condition: str,
        surface_version: str,
        prompt_version: str = "draft_v0",
        study_stage: str = "development",
    ) -> None:
        super().__init__()
        del db_path, perform_operations
        from .domain import ContextCondition, SocialWorld, SurfaceVersion

        self.action_logger = action_logger
        self.app_description = str(app_description or "Single-shot hospital decision.")
        self.call_id = str(call_id).strip()
        self.replicate_id = str(replicate_id).strip()
        self.episode_id = str(episode_id).strip()
        if not self.call_id or not self.replicate_id or self.episode_id != self.call_id:
            raise ValueError("episode_id must equal the unique non-empty call_id")
        self.condition_id = str(condition_id)
        self.seed = int(seed)
        self.study_stage = str(study_stage).strip()
        if self.study_stage not in {"development", "confirmatory"}:
            raise ValueError("study_stage must be development or confirmatory")
        self.cell = PromptCell(
            world=SocialWorld(social_world),
            context=ContextCondition(context_condition),
            surface=SurfaceVersion(surface_version),
            prompt_version=str(prompt_version),
        )
        if self.cell.condition_id != self.condition_id:
            raise ValueError("condition_id does not match the configured factor levels")
        self.agent_names: list[str] = []
        self._decisions: dict[str, DecisionRecord] = {}
        self._finalized = False
        self._step = 0

    def name(self) -> str:
        return "selective_robustness_hospital"

    def description(self) -> str:
        return self.app_description

    def current_step(self) -> int:
        logger_step = getattr(self.action_logger, "episode_idx", None)
        return int(logger_step) if logger_step is not None else self._step

    def initialize(self, agent_names: list[str], **kwargs: Any) -> None:
        del kwargs
        names = [str(name).strip() for name in agent_names]
        if len(names) != 2 or len(set(names)) != 2 or any(not name for name in names):
            raise ValueError("the hospital episode requires two distinct named actors")
        self.agent_names = names
        self._decisions = {}
        self._finalized = False
        self._step = 0
        self._log(
            source_user="system",
            action_type="initialize_hospital_episode",
            data={
                "players": names,
                "condition_id": self.condition_id,
                "call_id": self.call_id,
                "replicate_id": self.replicate_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "prompt_version": self.cell.prompt_version,
                "study_stage": self.study_stage,
                "social_world": self.cell.world.value,
                "context_condition": self.cell.context.value,
                "surface_version": self.cell.surface.value,
            },
        )

    def record_decision(
        self,
        agent_name: str,
        decision: DecisionRecord,
        *,
        content_attempts: list[dict[str, Any]] | None = None,
    ) -> str:
        if agent_name not in self.agent_names:
            raise ValueError(f"unknown actor {agent_name!r}")
        if self.current_step() != 0 or self._finalized:
            raise ValueError("the single-decision episode no longer accepts decisions")
        if agent_name in self._decisions:
            raise ValueError(f"{agent_name} already submitted a decision")
        if not isinstance(decision, DecisionRecord):
            raise TypeError("decision must be a DecisionRecord")
        self._decisions[agent_name] = decision
        self._log(
            source_user=agent_name,
            action_type="choose_hospital_action",
            data={
                "action": decision.action.value
                if decision.action is not None
                else None,
                "source": decision.source,
                "prompt": decision.prompt,
                "prompt_sha256": decision.prompt_sha256,
                "raw_response": decision.raw_response,
                "valid_parse": decision.valid_parse,
                "parse_method": decision.parse_method,
                "parse_note": decision.parse_note,
                "attempts": decision.attempts,
                "provider_attempts": decision.provider_attempts,
                "provider_metadata": dict(decision.provider_metadata),
                "content_attempts": deepcopy(content_attempts or []),
                "condition_id": self.condition_id,
                "call_id": self.call_id,
                "replicate_id": self.replicate_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "prompt_version": self.cell.prompt_version,
                "study_stage": self.study_stage,
                "scenario": self.cell.scenario,
                "social_world": self.cell.world.value,
                "context_condition": self.cell.context.value,
                "surface_version": self.cell.surface.value,
            },
        )
        return f"Recorded {agent_name}'s private choice."

    def update(
        self,
        *,
        step: int,
        agent_names: list[str],
        context: Any | None = None,
    ) -> None:
        del context
        if list(agent_names) != self.agent_names:
            raise ValueError("runtime actor names changed after initialization")
        self._step = int(step)
        if step < 1 or self._finalized:
            return
        missing = [name for name in self.agent_names if name not in self._decisions]
        if missing:
            raise ValueError(
                f"single-decision episode is missing choices for: {', '.join(missing)}"
            )
        focal_name, counterpart_name = self.agent_names
        focal = self._decisions[focal_name]
        counterpart = self._decisions[counterpart_name]
        if focal.action is None:
            self._finalized = True
            self._log(
                source_user="system",
                action_type="hospital_episode_invalid_response",
                data={
                    "choices": {
                        focal_name: None,
                        counterpart_name: counterpart.action.value,
                    },
                    "payoffs": None,
                    "primary_outcome": None,
                    "valid_parse": False,
                    "counterpart_excluded_from_primary_analysis": True,
                    "condition_id": self.condition_id,
                    "call_id": self.call_id,
                    "replicate_id": self.replicate_id,
                    "episode_id": self.episode_id,
                    "seed": self.seed,
                    "prompt_version": self.cell.prompt_version,
                    "study_stage": self.study_stage,
                },
            )
            return
        if counterpart.action is None:
            raise AssertionError("the scripted counterpart must have a domain action")
        payoffs = resolve_payoffs(focal.action, counterpart.action)
        self._finalized = True
        self._log(
            source_user="system",
            action_type="hospital_episode_payoff",
            data={
                "choices": {
                    focal_name: focal.action.value,
                    counterpart_name: counterpart.action.value,
                },
                "payoffs": {
                    focal_name: payoffs.focal,
                    counterpart_name: payoffs.counterpart,
                },
                "primary_outcome": focal.action.value,
                "counterpart_excluded_from_primary_analysis": True,
                "condition_id": self.condition_id,
                "call_id": self.call_id,
                "replicate_id": self.replicate_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "prompt_version": self.cell.prompt_version,
                "study_stage": self.study_stage,
            },
        )

    def _log(self, *, source_user: str, action_type: str, data: dict[str, Any]) -> None:
        log = getattr(self.action_logger, "log", None)
        if callable(log):
            log(
                {
                    "source_user": source_user,
                    "action_type": action_type,
                    "label": action_type,
                    "data": data,
                }
            )
