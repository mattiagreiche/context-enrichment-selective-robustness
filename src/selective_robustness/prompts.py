"""Draft hospital stimulus atoms and constrained surface renderers."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import ContextCondition, PromptCell, SocialWorld, SurfaceVersion


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    field: str
    canonical: str
    alternative: str | None = None


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    cell: PromptCell
    text: str
    fact_ids: tuple[str, ...]
    social_fact_ids: tuple[str, ...]
    social_context: str
    task_block: str


_ROLE_FACTS = (
    Fact(
        "role.focal",
        "Focal role",
        "You are the emergency-supply manager for Hospital A.",
    ),
)


_DECISION_FACTS = (
    Fact(
        "decision.batch_control",
        "Resource control",
        "Hospital A and Hospital B each control a designated batch of emergency supplies. You control Hospital A's batch, and the other manager controls Hospital B's batch. The action labels apply in the same way to both managers. For either manager, TRANSFER sends the designated batch to the other hospital, while RETAIN keeps the designated batch at the hospital that controls it.",
    ),
    Fact(
        "decision.simultaneous_private",
        "Decision timing",
        "You and the other manager choose at the same time. You cannot communicate, and neither of you sees the other's choice before deciding. This decision concerns the current allocation. The hospitals have an ongoing operational relationship, and similar allocation situations may arise in the future.",
    ),
)


_WORLD_FACTS = {
    SocialWorld.REGIONAL_MUTUAL_AID: (
        Fact(
            "world.arrangement",
            "Institutional scope",
            "For this allocation, the shared regional emergency-supply arrangement covers supply continuity across Hospital A and Hospital B.",
            "For this allocation, supply continuity across Hospital A and Hospital B is covered by the shared regional emergency-supply arrangement.",
        ),
    ),
    SocialWorld.LOCAL_EMERGENCY_RESERVE: (
        Fact(
            "world.arrangement",
            "Institutional scope",
            "For this allocation, each hospital's separate local emergency-reserve arrangement covers supply continuity within that hospital.",
            "For this allocation, supply continuity within each hospital is covered by that hospital's separate local emergency-reserve arrangement.",
        ),
    ),
}


_CONTEXT_FACTS = {
    ContextCondition.MINIMAL: (),
    ContextCondition.NEUTRAL_PADDED: (
        Fact(
            "padding.inventory_system",
            "Inventory system",
            "Your supply office uses a standard electronic inventory system for routine stock records.",
            "Your supply office keeps routine stock records in a standard electronic inventory system.",
        ),
        Fact(
            "padding.form",
            "Recordkeeping",
            "The office logs completed inventory updates on its usual daily recordkeeping form.",
            "The office uses its usual daily recordkeeping form to log completed inventory updates.",
        ),
        Fact(
            "padding.meeting",
            "Routine schedule",
            "Staff review inventory totals during the supply office's scheduled afternoon meeting.",
            "The supply office's scheduled afternoon meeting includes staff review of inventory totals.",
        ),
    ),
    ContextCondition.GENERIC_BACKGROUND: (
        Fact(
            "background.team",
            "Team",
            "You supervise two inventory assistants who work in the emergency-supply office.",
            "You supervise a pair of inventory assistants who work in the emergency-supply office.",
        ),
        Fact(
            "background.reporting",
            "Reporting line",
            "You report to the operations director through the hospital's standard management structure.",
            "You report through the hospital's standard management structure to its operations director.",
        ),
        Fact(
            "background.meetings",
            "Organizational routine",
            "You attend the hospital's weekly administrative planning meetings with managers from other departments.",
            "You join managers from other departments at the hospital's weekly administrative planning meetings.",
        ),
    ),
    ContextCondition.DECISION_GROUNDED: (),
    ContextCondition.NORMATIVE_GROUNDED: (),
}


_GROUNDED_FACTS_LEGACY = {
    SocialWorld.REGIONAL_MUTUAL_AID: (
        Fact(
            "grounding.counterpart_link",
            "Counterpart relationship",
            "You represent Hospital A, and the other manager represents Hospital B, within that arrangement.",
            "Within that arrangement, you represent Hospital A, and the other manager represents Hospital B.",
        ),
        Fact(
            "grounding.procedure",
            "Procedure",
            "Its shared shortage procedure governs both designated batches in this allocation.",
            "For this allocation, its shared shortage procedure governs both designated batches.",
        ),
        Fact(
            "grounding.role_connection",
            "Role connection",
            "The arrangement's supply-continuity scope defines your responsibility for this decision.",
            "For this decision, your responsibility is defined by the arrangement's supply-continuity scope.",
        ),
    ),
    SocialWorld.LOCAL_EMERGENCY_RESERVE: (
        Fact(
            "grounding.counterpart_link",
            "Counterpart relationship",
            "You represent Hospital A, and the other manager represents Hospital B, within those arrangements.",
            "Within those arrangements, you represent Hospital A, and the other manager represents Hospital B.",
        ),
        Fact(
            "grounding.procedure",
            "Procedure",
            "Each local shortage procedure governs its hospital's designated batch in this allocation.",
            "For this allocation, each local shortage procedure governs its hospital's designated batch.",
        ),
        Fact(
            "grounding.role_connection",
            "Role connection",
            "The applicable arrangement's supply-continuity scope defines your responsibility for this decision.",
            "For this decision, your responsibility is defined by the applicable arrangement's supply-continuity scope.",
        ),
    ),
}


# The v3 candidate removes the responsibility-defining language used by the
# historical draft_v0/pilot_v1/pilot_v2 family.  These atoms describe who is
# connected to whom, the scope used by the procedure, and how the focal batch
# enters that procedure without directly assigning the agent a broader or
# narrower objective.
_GROUNDED_FACTS_V3 = {
    SocialWorld.REGIONAL_MUTUAL_AID: (
        Fact(
            "grounding.counterpart_link",
            "Counterpart relationship",
            "You and the other manager are the designated supply contacts for two hospitals in the same regional emergency-supply network.",
            "For two hospitals in the same regional emergency-supply network, you and the other manager serve as the designated supply contacts.",
        ),
        Fact(
            "grounding.procedure",
            "Procedure",
            "During shortages, the network's procedure assesses supply conditions across its participating hospitals.",
            "The network's procedure assesses conditions across participating hospitals during shortages.",
        ),
        Fact(
            "grounding.role_connection",
            "Role connection",
            "You manage Hospital A's designated batch within that regional procedure.",
            "Within that regional procedure, you manage Hospital A's designated batch.",
        ),
    ),
    SocialWorld.LOCAL_EMERGENCY_RESERVE: (
        Fact(
            "grounding.counterpart_link",
            "Counterpart relationship",
            "You and the other manager are the designated supply contacts for hospitals in separate municipal emergency-supply systems.",
            "For hospitals in separate municipal emergency-supply systems, you and the other manager serve as the designated supply contacts.",
        ),
        Fact(
            "grounding.procedure",
            "Procedure",
            "During shortages, each system's procedure assesses supply conditions within its own hospital.",
            "Each system's procedure assesses conditions within its own hospital during shortages.",
        ),
        Fact(
            "grounding.role_connection",
            "Role connection",
            "You manage Hospital A's designated batch within Hospital A's local procedure.",
            "Within Hospital A's local procedure, you manage Hospital A's designated batch.",
        ),
    ),
}


_V3_PROMPT_VERSIONS = frozenset({"draft_v3", "pilot_v3"})


# The v4 family adds a matched normative condition. Both grounded conditions
# use the same counterpart and procedure facts. Only the final atom changes:
# descriptive grounding states that the procedure covers the decision, while
# normative grounding binds the agent's role to following that procedure.
_GROUNDED_FACTS_V4 = {
    ContextCondition.DECISION_GROUNDED: {
        SocialWorld.REGIONAL_MUTUAL_AID: (
            _GROUNDED_FACTS_V3[SocialWorld.REGIONAL_MUTUAL_AID][0],
            _GROUNDED_FACTS_V3[SocialWorld.REGIONAL_MUTUAL_AID][1],
            Fact(
                "grounding.role_connection",
                "Decision coverage",
                "The regional network's shortage procedure covers your decision about how to manage Hospital A's designated batch.",
                "The regional network's shortage procedure applies to your decision about managing Hospital A's designated batch.",
            ),
        ),
        SocialWorld.LOCAL_EMERGENCY_RESERVE: (
            _GROUNDED_FACTS_V3[SocialWorld.LOCAL_EMERGENCY_RESERVE][0],
            _GROUNDED_FACTS_V3[SocialWorld.LOCAL_EMERGENCY_RESERVE][1],
            Fact(
                "grounding.role_connection",
                "Decision coverage",
                "Hospital A's local shortage procedure covers your decision about how to manage Hospital A's designated batch.",
                "Hospital A's local shortage procedure applies to your decision about managing Hospital A's designated batch.",
            ),
        ),
    },
    ContextCondition.NORMATIVE_GROUNDED: {
        SocialWorld.REGIONAL_MUTUAL_AID: (
            _GROUNDED_FACTS_V3[SocialWorld.REGIONAL_MUTUAL_AID][0],
            _GROUNDED_FACTS_V3[SocialWorld.REGIONAL_MUTUAL_AID][1],
            Fact(
                "grounding.role_obligation",
                "Role requirement",
                "Your role requires you to follow the regional network's shortage procedure when deciding how to manage Hospital A's designated batch.",
                "When deciding how to manage Hospital A's designated batch, your role requires you to follow the regional network's shortage procedure.",
            ),
        ),
        SocialWorld.LOCAL_EMERGENCY_RESERVE: (
            _GROUNDED_FACTS_V3[SocialWorld.LOCAL_EMERGENCY_RESERVE][0],
            _GROUNDED_FACTS_V3[SocialWorld.LOCAL_EMERGENCY_RESERVE][1],
            Fact(
                "grounding.role_obligation",
                "Role requirement",
                "Your role requires you to follow Hospital A's local shortage procedure when deciding how to manage Hospital A's designated batch.",
                "When deciding how to manage Hospital A's designated batch, your role requires you to follow Hospital A's local shortage procedure.",
            ),
        ),
    },
}


_V4_PROMPT_VERSIONS = frozenset({"draft_v4", "pilot_v4"})


_ACTION_FACTS = (
    Fact(
        "action.transfer",
        "TRANSFER",
        "TRANSFER: You send Hospital A's designated batch to Hospital B.",
    ),
    Fact(
        "action.retain",
        "RETAIN",
        "RETAIN: You keep Hospital A's designated batch at Hospital A.",
    ),
)


_PAYOFF_FACTS = (
    Fact(
        "payoff.transfer_transfer", "TRANSFER / TRANSFER", "TRANSFER | TRANSFER | 3 | 3"
    ),
    Fact("payoff.transfer_retain", "TRANSFER / RETAIN", "TRANSFER | RETAIN | 0 | 5"),
    Fact("payoff.retain_transfer", "RETAIN / TRANSFER", "RETAIN | TRANSFER | 5 | 0"),
    Fact("payoff.retain_retain", "RETAIN / RETAIN", "RETAIN | RETAIN | 1 | 1"),
)


_PAYOFF_CONTEXT_FACTS = (
    Fact(
        "payoff.interpretation",
        "Outcome interpretation",
        "The batches contain different supplies. Each hospital gains a supply benefit from receiving the other hospital's batch and incurs a handling cost when sending its own. The table applies these effects symmetrically and reports outcome points for Hospital A and Hospital B.",
    ),
    Fact(
        "payoff.direction",
        "Outcome ranking",
        "Higher outcome points represent better outcomes for that hospital.",
    ),
)


_RESPONSE_FACTS = (
    Fact(
        "response.role_choice",
        "Role choice",
        "Choose the action you would take in this role.",
    ),
    Fact(
        "response.instruction",
        "Response",
        "Respond with exactly one word: TRANSFER or RETAIN. Do not provide an explanation.",
    ),
)


def social_facts_for(cell: PromptCell) -> tuple[Fact, ...]:
    if not isinstance(cell, PromptCell):
        raise TypeError("cell must be a PromptCell")
    facts = list(_WORLD_FACTS[cell.world])
    facts.extend(_CONTEXT_FACTS[cell.context])
    if cell.context in {
        ContextCondition.DECISION_GROUNDED,
        ContextCondition.NORMATIVE_GROUNDED,
    }:
        if cell.prompt_version in _V4_PROMPT_VERSIONS:
            facts.extend(_GROUNDED_FACTS_V4[cell.context][cell.world])
        elif cell.context is ContextCondition.NORMATIVE_GROUNDED:
            raise ValueError(
                "normative_grounded is defined only for draft_v4 and pilot_v4"
            )
        else:
            grounded = (
                _GROUNDED_FACTS_V3
                if cell.prompt_version in _V3_PROMPT_VERSIONS
                else _GROUNDED_FACTS_LEGACY
            )
            facts.extend(grounded[cell.world])
    return tuple(facts)


def invariant_role_facts() -> tuple[Fact, ...]:
    return _ROLE_FACTS


def invariant_task_facts() -> tuple[Fact, ...]:
    return (
        _DECISION_FACTS
        + _ACTION_FACTS
        + _PAYOFF_CONTEXT_FACTS
        + _PAYOFF_FACTS
        + _RESPONSE_FACTS
    )


def facts_for(cell: PromptCell) -> tuple[Fact, ...]:
    social_facts = social_facts_for(cell)
    facts = (
        _ROLE_FACTS
        + social_facts
        + _DECISION_FACTS
        + _ACTION_FACTS
        + _PAYOFF_CONTEXT_FACTS
        + _PAYOFF_FACTS
        + _RESPONSE_FACTS
    )
    ids = [fact.fact_id for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate fact identifier in {cell.condition_id}")
    return facts


def invariant_role_block() -> str:
    return "Role\n\n" + " ".join(fact.canonical for fact in _ROLE_FACTS)


def invariant_task_block() -> str:
    resource_control, timing = (fact.canonical for fact in _DECISION_FACTS)
    actions = "\n".join(f"- {fact.canonical}" for fact in _ACTION_FACTS)
    payoff_context = "\n\n".join(fact.canonical for fact in _PAYOFF_CONTEXT_FACTS)
    payoff_rows = "\n".join(fact.canonical for fact in _PAYOFF_FACTS)
    response = "\n\n".join(fact.canonical for fact in _RESPONSE_FACTS)
    return (
        "Decision task\n\n"
        f"{resource_control}\n\n"
        f"{timing}\n\n"
        "Actions\n\n"
        f"{actions}\n\n"
        "Payoff table\n\n"
        f"{payoff_context}\n\n"
        "Your action | Other manager's action | Hospital A points | Hospital B points\n"
        "--- | --- | ---: | ---:\n"
        f"{payoff_rows}\n\n"
        f"{response}"
    )


def _render_social_context(facts: tuple[Fact, ...], surface: SurfaceVersion) -> str:
    if surface is SurfaceVersion.CANONICAL_PROSE:
        return "\n\n".join(fact.canonical for fact in facts)
    if surface is SurfaceVersion.BULLETS:
        return "\n".join(f"- {fact.canonical}" for fact in facts)
    if surface is SurfaceVersion.KEY_VALUE_TABLE:
        rows = ["Field | Value", "--- | ---"]
        rows.extend(f"{fact.field} | {fact.canonical}" for fact in facts)
        return "\n".join(rows)
    if surface is SurfaceVersion.ALTERNATIVE_PROSE:
        alternatives = []
        for fact in facts:
            if fact.alternative is None:
                raise ValueError(f"{fact.fact_id} lacks an alternative-prose rendering")
            alternatives.append(fact.alternative)
        return "\n\n".join(alternatives)
    raise AssertionError(f"unhandled surface version: {surface}")


def render_prompt(cell: PromptCell) -> RenderedPrompt:
    if not isinstance(cell, PromptCell):
        raise TypeError("cell must be a PromptCell")
    social_facts = social_facts_for(cell)
    social_context = _render_social_context(social_facts, cell.surface)
    role_block = invariant_role_block()
    task_block = invariant_task_block()
    text = (
        "Hospital supply decision\n\n"
        f"{role_block}\n\n"
        "Social context\n\n"
        f"{social_context}\n\n"
        f"{task_block}\n"
    )
    return RenderedPrompt(
        cell=cell,
        text=text,
        fact_ids=tuple(fact.fact_id for fact in facts_for(cell)),
        social_fact_ids=tuple(fact.fact_id for fact in social_facts),
        social_context=social_context,
        task_block=task_block,
    )
