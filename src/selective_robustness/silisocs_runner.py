"""Credential-free SiliSocS smoke runner."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .domain import PromptCell


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_scripted_smoke(
    cell: PromptCell,
    output: Path,
    *,
    scripted_response: str = "TRANSFER",
) -> None:
    if not isinstance(scripted_response, str) or not scripted_response:
        raise ValueError("scripted_response must be a non-empty string")
    root = repository_root()
    config_root = root / "scenarios" / "selective_robustness" / "conf"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "silisocs.runtime.runner",
        "--config-path",
        str(config_root),
        "agents=default",
        f"output_rootname={output}",
        f"hydra.run.dir={output / '.hydra-run'}",
        "seed=0",
        "num_agents=2",
        "num_steps=2",
        f"social_world={cell.world.value}",
        f"context_condition={cell.context.value}",
        f"surface_version={cell.surface.value}",
        f"prompt_version={cell.prompt_version}",
        f"condition_id={cell.condition_id}",
        f"call_id={cell.condition_id}/smoke-call-0001",
        "replicate_id=smoke-replicate-0001",
        f"episode_id={cell.condition_id}/smoke-call-0001",
        f"selective.condition_id={cell.condition_id}",
        f"selective.call_id={cell.condition_id}/smoke-call-0001",
        "selective.replicate_id=smoke-replicate-0001",
        f"selective.episode_id={cell.condition_id}/smoke-call-0001",
        f"++provider_extra_kwargs={{text_response:{scripted_response}}}",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    environment.pop("OPENAI_API_KEY", None)
    with tempfile.TemporaryDirectory(prefix="selective-robustness-smoke-") as cwd:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
            timeout=180,
        )
    if completed.returncode != 0:
        detail = " | ".join(completed.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"SiliSocS scripted smoke failed: {detail or 'no detail'}")
