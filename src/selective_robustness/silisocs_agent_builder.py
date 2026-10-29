"""Scenario-specific SiliSocS agent builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf
from silisocs.runtime.construction.agent_builders import AgentBuilder
from silisocs.runtime.construction.specs import AgentConfig


class HospitalAgentBuilder(AgentBuilder):
    def build_agent_configs(self) -> list[AgentConfig]:
        entries = getattr(self.config, "entries", None)
        if isinstance(entries, (DictConfig, ListConfig)):
            entries = OmegaConf.to_container(entries, resolve=True)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("agents.entries must be a sequence")
        configs: list[AgentConfig] = []
        names: set[str] = set()
        for index, raw in enumerate(entries):
            if not isinstance(raw, Mapping):
                raise TypeError(f"agents.entries[{index}] must be a mapping")
            class_path = str(raw.get("class_path", "")).strip()
            params = dict(raw.get("params", {}) or {})
            name = str(params.get("name", "")).strip()
            if not class_path or not name:
                raise ValueError(f"agents.entries[{index}] requires class_path and name")
            if name in names:
                raise ValueError(f"duplicate agent name {name!r}")
            names.add(name)
            params.setdefault("sim_role", {"name": "participant", "module_path": ""})
            params.setdefault("flow_tag", "default")
            configs.append(AgentConfig(class_path=class_path, params=params))
        if not configs:
            raise ValueError("agents.entries cannot be empty")
        return configs
