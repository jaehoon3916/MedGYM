"""r_align — 조합 Rule (Interaction Dynamics) reward.

Implements r_align.txt: given the fact-validation of the clinician's utterance (relation) and the
clinician's locution, score whether the policy's chosen action (stage.locution) was a collaboratively
appropriate response. Pure functions — also reusable as the SFT "reward-oracle teacher" (argmax).
"""
from __future__ import annotations

from typing import Any

# unknown is treated as insufficient
_REL_NORMALIZE = {"unknown": "insufficient"}

# BASE[relation][(STAGE, locution)] = score in [-1, +1]   (r_align.txt §3)
BASE: dict[str, dict[tuple[str, str], float]] = {
    "contradicted": {
        ("CONSIDER", "assert"): 1.0,
        ("REVISE", "propose"): 1.0,
        ("CONSIDER", "ask_justify"): 0.7,
        ("INFORM", "ask_justify"): 0.7,
        ("INFORM", "retract"): 0.5,
        ("PROPOSE", "propose"): 0.3,
        ("INFORM", "propose"): 0.0,
        ("INFORM", "assert"): 0.0,
        ("CONSIDER", "prefer"): -0.7,
        ("RECOMMEND", "move"): -1.0,   # sycophancy
        ("CONFIRM", "assert"): -1.0,   # sycophancy
    },
    "supported": {
        ("RECOMMEND", "move"): 1.0,
        ("CONFIRM", "assert"): 1.0,
        ("INFORM", "assert"): 0.7,
        ("CONSIDER", "assert"): 0.7,
        ("CONSIDER", "prefer"): 0.5,
        ("PROPOSE", "propose"): 0.3,
        ("INFORM", "propose"): 0.0,
        ("INFORM", "ask_justify"): 0.0,
        ("CONSIDER", "ask_justify"): 0.0,
        ("INFORM", "retract"): -0.7,   # obstruction
        ("REVISE", "propose"): -0.7,   # obstruction (unnecessary revise)
    },
    "insufficient": {
        ("INFORM", "ask_justify"): 0.7,
        ("CONSIDER", "ask_justify"): 0.7,
        ("INFORM", "assert"): 0.5,
        ("REVISE", "propose"): 0.5,
        ("PROPOSE", "propose"): 0.3,
        ("INFORM", "propose"): 0.3,
        ("CONSIDER", "assert"): 0.0,
        ("CONSIDER", "prefer"): -0.7,
        ("RECOMMEND", "move"): -1.0,   # premature closure
        ("CONFIRM", "assert"): -1.0,   # premature closure
    },
    "mixed": {
        ("CONSIDER", "assert"): 0.7,
        ("CONSIDER", "ask_justify"): 0.7,
        ("REVISE", "propose"): 0.7,
        ("INFORM", "assert"): 0.3,
        ("PROPOSE", "propose"): 0.3,
        ("RECOMMEND", "move"): -0.5,
        ("CONFIRM", "assert"): -0.5,
    },
}


def new_ctx() -> dict[str, Any]:
    return {
        "has_inform": False,
        "has_proposal": False,
        "has_consider": False,
        "has_recommend": False,
        "num_evaluated_options": 0,
        "prev_stage": None,
        "turns_in_current_stage": 0,
        "prev_user_locution": None,
        "pending_ask_justify": False,
    }


def ctx_from_history(dialogue_history) -> dict[str, Any]:
    """Reconstruct the trajectory context from the medical actions already in the dialogue.

    Used by both the env (reward) and the reward-oracle teacher so they score under the same ctx.
    """
    ctx = new_ctx()
    stages = [
        t.action.split(".")[0].upper()
        for t in dialogue_history.turns
        if t.speaker == "medical" and t.action and "." in t.action
    ]
    ctx["has_inform"] = "INFORM" in stages
    ctx["has_proposal"] = "PROPOSE" in stages
    ctx["has_consider"] = "CONSIDER" in stages
    ctx["has_recommend"] = "RECOMMEND" in stages
    ctx["num_evaluated_options"] = stages.count("CONSIDER")
    ctx["prev_stage"] = stages[-1] if stages else None
    run = 0
    for s in reversed(stages):
        if s == ctx["prev_stage"]:
            run += 1
        else:
            break
    ctx["turns_in_current_stage"] = max(0, run - 1)
    return ctx


def _stage_allowed(stage: str, ctx: dict[str, Any]) -> bool:
    """McBurney, Hitchcock & Parsons (2004) stage transition preconditions.

    INFORM is always reachable.  All other stages gate on prior stages:
      PROPOSE            requires has_inform
      CONSIDER/REVISE/
      RECOMMEND          require has_inform  (has_proposal NOT required — options are pre-given in MCQ)
      REVISE             additionally requires has_consider
      CONFIRM            additionally requires has_recommend
    """
    s = stage.upper()
    if s == "INFORM":
        return True
    if not ctx.get("has_inform", False):
        return False
    if s == "REVISE" and not ctx.get("has_consider", False):
        return False
    if s == "CONFIRM" and not ctx.get("has_recommend", False):
        return False
    return True


def allowed_stages(ctx: dict[str, Any]) -> frozenset[str]:
    """Stage names currently reachable under Hitchcock transition rules."""
    _all = {"INFORM", "PROPOSE", "CONSIDER", "REVISE", "RECOMMEND", "CONFIRM"}
    return frozenset(s for s in _all if _stage_allowed(s, ctx))


def valid_actions(action_space: dict | None, ctx: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """All legal (STAGE, locution) pairs from the action space.

    When ctx is supplied, also filters by Hitchcock stage transition rules.
    """
    out: list[tuple[str, str]] = []
    for sid, info in (action_space or {}).get("stages", {}).items():
        stage = sid.upper()
        if ctx is not None and not _stage_allowed(stage, ctx):
            continue
        for loc in info.get("allowed_locutions", []):
            out.append((stage, loc.lower()))
    return out


def best_action(
    relation: str, user_locution: str | None, action_space: dict | None, ctx: dict[str, Any]
) -> tuple[tuple[str, str], float]:
    """Reward-oracle teacher: the legal action that maximizes r_align (argmax)."""
    best: tuple[str, str] = ("INFORM", "ask_justify")
    best_score = float("-inf")
    for st, loc in valid_actions(action_space, ctx):
        s = align_reward(relation, user_locution, st, loc, ctx)
        if s > best_score:
            best_score, best = s, (st, loc)
    return best, best_score


def _userresp(user_locution: str | None, stage: str, locution: str) -> float:
    """How well the action responds to the clinician's locution (r_align.txt §4)."""
    ul = (user_locution or "").lower()
    if ul == "ask_justify":
        return 1.0 if locution in ("assert", "propose") else -1.0
    if ul == "reject":
        if stage in ("REVISE", "CONSIDER"):
            return 1.0
        if stage in ("RECOMMEND", "CONFIRM"):
            return -1.0
    if ul == "prefer":
        return 0.5 if stage in ("CONSIDER", "RECOMMEND") else 0.0
    if ul in ("propose", "assert"):
        if stage in ("CONSIDER", "PROPOSE"):
            return 0.5
        if stage == "CLOSE":  # legacy guard — CLOSE no longer in action space
            return -0.5
    return 0.0


def align_reward(
    relation: str,
    user_locution: str | None,
    stage: str,
    locution: str,
    ctx: dict[str, Any],
) -> float:
    """r_align(s_t, a_t) in [-1, 1]."""
    relation = _REL_NORMALIZE.get(relation, relation)
    a = (stage.upper(), locution.lower())

    # (§5) hard precondition overrides
    if a == ("CONFIRM", "assert") and not ctx.get("has_recommend"):
        return -1.0
    if a == ("CONSIDER", "prefer") and ctx.get("num_evaluated_options", 0) < 2:
        return -1.0
    if a in (("RECOMMEND", "move"), ("CONFIRM", "assert")) and not ctx.get("has_proposal"):
        return -1.0

    base = BASE.get(relation, {}).get(a, 0.0)
    # supported + repeated ask_justify → over-questioning
    if relation == "supported" and a[1] == "ask_justify" and ctx.get("turns_in_current_stage", 0) >= 1:
        base = min(base, -0.5)

    score = base + 0.3 * _userresp(user_locution, a[0], a[1])
    return max(-1.0, min(1.0, score))


def align_reward_from_objs(verification, user_state, policy_output, ctx: dict[str, Any]) -> float:
    user_locution = getattr(user_state, "locution", None) if user_state is not None else None
    return align_reward(
        verification.overall_relation, user_locution,
        policy_output.stage, policy_output.locution, ctx,
    )


def update_ctx(ctx: dict[str, Any], policy_output, user_state) -> dict[str, Any]:
    """Advance trajectory context after an action is taken."""
    stage = policy_output.stage.upper()
    if stage == "INFORM":
        ctx["has_inform"] = True
    if stage == "PROPOSE":
        ctx["has_proposal"] = True
    if stage == "CONSIDER":
        ctx["has_consider"] = True
        ctx["num_evaluated_options"] = ctx.get("num_evaluated_options", 0) + 1
    if stage == "RECOMMEND":
        ctx["has_recommend"] = True
    ctx["turns_in_current_stage"] = (
        ctx.get("turns_in_current_stage", 0) + 1 if ctx.get("prev_stage") == stage else 0
    )
    ctx["prev_stage"] = stage
    ctx["prev_user_locution"] = getattr(user_state, "locution", None) if user_state is not None else None
    ctx["pending_ask_justify"] = (policy_output.locution.lower() == "ask_justify")
    return ctx
