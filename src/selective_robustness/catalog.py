"""Factorial catalog for the hospital development family."""

from __future__ import annotations

from collections import Counter

from .domain import ContextCondition, PromptCell, SocialWorld, SurfaceVersion


_FOUR_CONTEXT_CONDITIONS = (
    ContextCondition.MINIMAL,
    ContextCondition.NEUTRAL_PADDED,
    ContextCondition.GENERIC_BACKGROUND,
    ContextCondition.DECISION_GROUNDED,
)
_FIVE_CONTEXT_CONDITIONS = _FOUR_CONTEXT_CONDITIONS + (
    ContextCondition.NORMATIVE_GROUNDED,
)
_FIVE_CONTEXT_PROMPT_VERSIONS = frozenset({"draft_v4", "pilot_v4"})


def context_conditions_for(prompt_version: str) -> tuple[ContextCondition, ...]:
    """Return the context crossing defined by a prompt version."""

    if prompt_version in _FIVE_CONTEXT_PROMPT_VERSIONS:
        return _FIVE_CONTEXT_CONDITIONS
    return _FOUR_CONTEXT_CONDITIONS


def hospital_cells(prompt_version: str = "pilot_v4") -> tuple[PromptCell, ...]:
    contexts = context_conditions_for(prompt_version)
    return tuple(
        PromptCell(
            world=world,
            context=context,
            surface=surface,
            prompt_version=prompt_version,
        )
        for world in SocialWorld
        for context in contexts
        for surface in SurfaceVersion
    )


def validate_hospital_catalog(cells: tuple[PromptCell, ...] | None = None) -> None:
    selected = cells if cells is not None else hospital_cells()
    prompt_versions = {cell.prompt_version for cell in selected}
    if len(prompt_versions) != 1:
        raise ValueError("hospital catalog must contain exactly one prompt version")
    prompt_version = next(iter(prompt_versions))
    contexts = context_conditions_for(prompt_version)
    expected_count = len(SocialWorld) * len(contexts) * len(SurfaceVersion)
    if len(selected) != expected_count:
        raise ValueError(
            f"hospital catalog must contain {expected_count} cells, "
            f"found {len(selected)}"
        )
    ids = [cell.condition_id for cell in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("condition identifiers must be unique")
    expected = {
        (world, context, surface)
        for world in SocialWorld
        for context in contexts
        for surface in SurfaceVersion
    }
    actual = {(cell.world, cell.context, cell.surface) for cell in selected}
    if actual != expected:
        raise ValueError("hospital catalog does not contain the complete factorial crossing")
    world_counts = Counter(cell.world for cell in selected)
    context_counts = Counter(cell.context for cell in selected)
    surface_counts = Counter(cell.surface for cell in selected)
    cells_per_world = len(contexts) * len(SurfaceVersion)
    cells_per_context = len(SocialWorld) * len(SurfaceVersion)
    cells_per_surface = len(SocialWorld) * len(contexts)
    if set(world_counts.values()) != {cells_per_world}:
        raise ValueError(f"each social world must contain {cells_per_world} cells")
    if set(context_counts.values()) != {cells_per_context}:
        raise ValueError(
            f"each context condition must contain {cells_per_context} cells"
        )
    if set(surface_counts.values()) != {cells_per_surface}:
        raise ValueError(
            f"each surface version must contain {cells_per_surface} cells"
        )
