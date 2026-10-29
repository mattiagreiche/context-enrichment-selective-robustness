"""Credential-free validation and diagnostics for the draft prompt family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .catalog import (
    context_conditions_for,
    hospital_cells,
    validate_hospital_catalog,
)
from .domain import PromptCell, SocialWorld, SurfaceVersion
from .prompts import (
    facts_for,
    invariant_role_facts,
    invariant_task_block,
    invariant_task_facts,
    render_prompt,
)


@dataclass(frozen=True, slots=True)
class PromptAuditRow:
    condition_id: str
    prompt_version: str
    social_world: str
    context_condition: str
    surface_version: str
    prompt_sha256: str
    task_block_sha256: str
    prompt_characters: int
    prompt_words: int
    social_context_characters: int
    social_context_words: int
    fact_ids: tuple[str, ...]
    social_fact_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fact_ids"] = list(self.fact_ids)
        value["social_fact_ids"] = list(self.social_fact_ids)
        return value


def text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return sha256(text.encode("utf-8")).hexdigest()


def validate_and_audit_hospital_prompts(
    cells: tuple[PromptCell, ...] | None = None,
) -> tuple[PromptAuditRow, ...]:
    selected = cells if cells is not None else hospital_cells()
    validate_hospital_catalog(selected)
    expected_task_block = invariant_task_block()
    expected_task_hash = text_sha256(expected_task_block)
    role_fact_ids = tuple(fact.fact_id for fact in invariant_role_facts())
    task_fact_ids = tuple(fact.fact_id for fact in invariant_task_facts())
    rows: list[PromptAuditRow] = []
    prompt_version = selected[0].prompt_version
    contexts = context_conditions_for(prompt_version)
    expected_count = len(SocialWorld) * len(contexts) * len(SurfaceVersion)

    for world in SocialWorld:
        for context in contexts:
            group = tuple(
                render_prompt(
                    next(
                        cell
                        for cell in selected
                        if cell.world is world
                        and cell.context is context
                        and cell.surface is surface
                    )
                )
                for surface in SurfaceVersion
            )
            expected_fact_ids = group[0].fact_ids
            expected_social_ids = group[0].social_fact_ids
            if any(prompt.fact_ids != expected_fact_ids for prompt in group[1:]):
                raise ValueError(f"surface facts diverge for {world.value}/{context.value}")
            if any(prompt.social_fact_ids != expected_social_ids for prompt in group[1:]):
                raise ValueError(
                    f"social-context fact order diverges for {world.value}/{context.value}"
                )
            if len({prompt.social_context for prompt in group}) != len(SurfaceVersion):
                raise ValueError(
                    f"social-context surfaces are not unique for {world.value}/{context.value}"
                )

            for prompt in group:
                if prompt.task_block != expected_task_block:
                    raise ValueError(
                        f"invariant task block changed in {prompt.cell.condition_id}"
                    )
                if tuple(
                    fact.fact_id
                    for fact in facts_for(prompt.cell)
                    if fact.fact_id in role_fact_ids
                ) != role_fact_ids:
                    raise ValueError(
                        f"invariant role fact order changed in {prompt.cell.condition_id}"
                    )
                if tuple(
                    fact.fact_id
                    for fact in facts_for(prompt.cell)
                    if fact.fact_id in task_fact_ids
                ) != task_fact_ids:
                    raise ValueError(
                        f"invariant task fact order changed in {prompt.cell.condition_id}"
                    )
                _validate_task_contract(prompt.text, prompt.cell.condition_id)
                rows.append(
                    PromptAuditRow(
                        condition_id=prompt.cell.condition_id,
                        prompt_version=prompt.cell.prompt_version,
                        social_world=prompt.cell.world.value,
                        context_condition=prompt.cell.context.value,
                        surface_version=prompt.cell.surface.value,
                        prompt_sha256=text_sha256(prompt.text),
                        task_block_sha256=expected_task_hash,
                        prompt_characters=len(prompt.text),
                        prompt_words=len(prompt.text.split()),
                        social_context_characters=len(prompt.social_context),
                        social_context_words=len(prompt.social_context.split()),
                        fact_ids=prompt.fact_ids,
                        social_fact_ids=prompt.social_fact_ids,
                    )
                )

    if (
        len(rows) != expected_count
        or len({row.prompt_sha256 for row in rows}) != expected_count
    ):
        raise ValueError(
            f"the draft family must contain {expected_count} distinct rendered prompts"
        )
    if {row.task_block_sha256 for row in rows} != {expected_task_hash}:
        raise ValueError("the task block must be byte-identical across all prompts")
    return tuple(rows)


def validate_prompt_fixture_directory(root: Path) -> None:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read draft fixture manifest: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError("draft fixture manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cells"), list):
        raise ValueError("draft fixture manifest must contain a cell list")
    prompt_version = manifest.get("prompt_version")
    if prompt_version not in {"draft_v0", "draft_v3", "draft_v4"}:
        raise ValueError("draft fixture manifest has an unsupported prompt version")
    expected_manifest = {
        "schema_version": 3,
        "status": "draft_unreviewed",
        "prompt_version": prompt_version,
        "surface_scope": "social_context_block_only",
        "alternative_prose_operation": "close_paraphrase_same_fact_order",
        "review_status": "pending",
        "authorization_status": "unauthorized",
        "human_review_complete": False,
        "paid_calls_authorized": False,
        "frozen": False,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("draft fixture manifest metadata does not match its version")
    entries = manifest["cells"]
    expected_count = len(hospital_cells(prompt_version))
    if len(entries) != expected_count:
        raise ValueError(
            f"draft fixture manifest must contain {expected_count} cells"
        )
    by_id = {
        entry.get("condition_id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("condition_id"), str)
    }
    if len(by_id) != expected_count:
        raise ValueError("draft fixture condition identifiers must be unique")
    expected_files = {"manifest.json"}
    for cell in hospital_cells(prompt_version):
        entry = by_id.get(cell.condition_id)
        if not isinstance(entry, dict):
            raise ValueError(f"missing draft fixture for {cell.condition_id}")
        expected_factors = {
            "prompt_version": cell.prompt_version,
            "social_world": cell.world.value,
            "context_condition": cell.context.value,
            "surface_version": cell.surface.value,
        }
        if any(entry.get(key) != value for key, value in expected_factors.items()):
            raise ValueError(f"draft fixture factors mismatch in {cell.condition_id}")
        file_name = entry.get("file")
        expected_name = "__".join(
            (cell.world.value, cell.context.value, cell.surface.value)
        ) + ".txt"
        if file_name != expected_name or Path(file_name).name != file_name:
            raise ValueError(f"invalid draft fixture file for {cell.condition_id}")
        expected_files.add(file_name)
        try:
            fixture = (root / file_name).read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"cannot read draft fixture {file_name}: {error}") from error
        rendered = render_prompt(cell)
        if fixture != rendered.text:
            raise ValueError(f"draft fixture drift in {cell.condition_id}")
        if entry.get("prompt_sha256") != text_sha256(fixture):
            raise ValueError(f"draft fixture hash mismatch in {cell.condition_id}")
        if entry.get("task_block_sha256") != text_sha256(rendered.task_block):
            raise ValueError(f"task-block hash mismatch in {cell.condition_id}")
        if entry.get("fact_ids") != list(rendered.fact_ids):
            raise ValueError(f"fact checklist mismatch in {cell.condition_id}")
        if entry.get("social_fact_ids") != list(rendered.social_fact_ids):
            raise ValueError(f"social fact checklist mismatch in {cell.condition_id}")
        expected_lengths = {
            "prompt_characters": len(rendered.text),
            "prompt_words": len(rendered.text.split()),
            "social_context_characters": len(rendered.social_context),
            "social_context_words": len(rendered.social_context.split()),
        }
        if any(entry.get(key) != value for key, value in expected_lengths.items()):
            raise ValueError(f"draft fixture length audit mismatch in {cell.condition_id}")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("draft fixture directory contains missing or unexpected files")


def _validate_task_contract(text: str, condition_id: str) -> None:
    required_once = (
        "You are the emergency-supply manager for Hospital A.",
        "Hospital A and Hospital B each control a designated batch of emergency supplies.",
        "The action labels apply in the same way to both managers.",
        "For either manager, TRANSFER sends the designated batch to the other hospital, while RETAIN keeps the designated batch at the hospital that controls it.",
        "You and the other manager choose at the same time.",
        "This decision concerns the current allocation.",
        "The hospitals have an ongoing operational relationship, and similar allocation situations may arise in the future.",
        "TRANSFER: You send Hospital A's designated batch to Hospital B.",
        "RETAIN: You keep Hospital A's designated batch at Hospital A.",
        "The batches contain different supplies.",
        "Each hospital gains a supply benefit from receiving the other hospital's batch and incurs a handling cost when sending its own.",
        "Higher outcome points represent better outcomes for that hospital.",
        "TRANSFER | TRANSFER | 3 | 3",
        "TRANSFER | RETAIN | 0 | 5",
        "RETAIN | TRANSFER | 5 | 0",
        "RETAIN | RETAIN | 1 | 1",
        "Choose the action you would take in this role.",
        "Respond with exactly one word: TRANSFER or RETAIN.",
        "Do not provide an explanation.",
    )
    for required in required_once:
        if text.count(required) != 1:
            raise ValueError(f"task contract mismatch in {condition_id}: {required}")
    forbidden = (
        "COOPERATE",
        "DEFECT",
        "JSON",
        "Maya",
        "prompt version",
        "if you and the other manager both choose TRANSFER",
        "This allocation is a one-time decision.",
        "There is no later round",
    )
    for value in forbidden:
        if value in text:
            raise ValueError(f"forbidden task label in {condition_id}: {value}")
    role_index = text.index("Role\n\n")
    context_index = text.index("Social context\n\n")
    decision_index = text.index("Decision task\n\n")
    if not role_index < context_index < decision_index:
        raise ValueError(f"prompt block order mismatch in {condition_id}")
