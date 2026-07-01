#!/usr/bin/env python3
"""
Verify that the reward system (core/reward_align.py r_align + core/reward.py R(tau))
behaves exactly as specified in /home/kjy/Jaehoon/r_align.txt.

All checks are deterministic (no LLM calls). Golden values below are transcribed
independently from r_align.txt sections 3/4/5 -- NOT copy-pasted from reward_align.py's
BASE dict -- so this is a real conformance check, not a self-comparison.

Sections 1-5 are hard PASS/FAIL (process exits non-zero if any fails).
Section 6 (live dialogue integration, optional) is an observational check on an
existing outputs/poc_multiturn/results.json, reported as warnings, not hard failures.

Usage:
    python scripts/verify_reward.py [--results outputs/poc_multiturn/results.json]
    -> outputs/verify_reward/results.json + outputs/verify_reward/report.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import load_action_space
from core.reward_align import align_reward, ctx_from_history, valid_actions
from core.reward import DEFAULT_WEIGHTS, Trajectory, group_advantages, trajectory_return
from core.schemas import DialogueHistory

OUTPUT_DIR = _ROOT / "outputs" / "verify_reward"

# ---------------------------------------------------------------------------
# Golden spec, transcribed independently from r_align.txt SS3 (BASE table).
# Missing (relation, action) pairs are golden 0.0 per SS3 footer.
# ---------------------------------------------------------------------------
GOLDEN_BASE: dict[str, dict[tuple[str, str], float]] = {
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
        ("RECOMMEND", "move"): -1.0,
        ("CONFIRM", "assert"): -1.0,
        ("CLOSE", "withdraw_dialogue"): -0.7,
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
        ("INFORM", "retract"): -0.7,
        ("REVISE", "propose"): -0.7,
        ("CLOSE", "withdraw_dialogue"): -0.3,
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
        ("RECOMMEND", "move"): -1.0,
        ("CONFIRM", "assert"): -1.0,
        ("CLOSE", "withdraw_dialogue"): -1.0,
    },
    "mixed": {
        ("CONSIDER", "assert"): 0.7,
        ("CONSIDER", "ask_justify"): 0.7,
        ("REVISE", "propose"): 0.7,
        ("INFORM", "assert"): 0.3,
        ("PROPOSE", "propose"): 0.3,
        ("RECOMMEND", "move"): -0.5,
        ("CONFIRM", "assert"): -0.5,
        ("CLOSE", "withdraw_dialogue"): -0.5,
    },
}
# SS1: unknown is treated identically to insufficient
GOLDEN_BASE["unknown"] = GOLDEN_BASE["insufficient"]

_RELATIONS = ("contradicted", "supported", "insufficient", "mixed", "unknown")

# Neutral ctx: preconditions satisfied, no soft-cap triggers, USERRESP=0 (user_locution=None)
_NEUTRAL_CTX = {
    "has_proposal": True,
    "has_recommend": True,
    "num_evaluated_options": 2,
    "prev_stage": None,
    "turns_in_current_stage": 0,
    "prev_user_locution": None,
    "pending_ask_justify": False,
}


class PO:
    """Minimal stand-in for PolicyOutput / a (stage, locution) pair where only
    .stage/.locution attribute access is needed by ctx_from_history callers."""
    def __init__(self, stage: str, locution: str):
        self.stage = stage
        self.locution = locution


def _neutral_ctx() -> dict[str, Any]:
    return dict(_NEUTRAL_CTX)


# ---------------------------------------------------------------------------
# Section 1: BASE table conformance
# ---------------------------------------------------------------------------
def check_base_table(actions: list[tuple[str, str]]) -> dict[str, Any]:
    rows = []
    n_fail = 0
    for relation in _RELATIONS:
        for stage, loc in actions:
            golden = GOLDEN_BASE.get(relation, {}).get((stage, loc), 0.0)
            actual = align_reward(relation, None, stage, loc, _neutral_ctx())
            ok = abs(actual - golden) < 1e-9
            if not ok:
                n_fail += 1
            rows.append({
                "relation": relation, "stage": stage, "locution": loc,
                "golden": golden, "actual": actual, "pass": ok,
            })
    return {"name": "BASE table conformance", "rows": rows, "n_fail": n_fail, "pass": n_fail == 0}


# ---------------------------------------------------------------------------
# Section 2: Precondition overrides (SS5) -- BASE-independent hard -1.0
# ---------------------------------------------------------------------------
def check_preconditions() -> dict[str, Any]:
    cases = [
        {
            "name": "CONFIRM.assert without prior RECOMMEND.move",
            "stage": "CONFIRM", "locution": "assert",
            "ctx": {**_neutral_ctx(), "has_recommend": False},
            "relation": "supported",  # BASE would be +1.0 if not overridden
        },
        {
            "name": "CONSIDER.prefer with num_evaluated_options<2",
            "stage": "CONSIDER", "locution": "prefer",
            "ctx": {**_neutral_ctx(), "num_evaluated_options": 1},
            "relation": "supported",  # BASE would be +0.5 if not overridden
        },
        {
            "name": "RECOMMEND.move without has_proposal",
            "stage": "RECOMMEND", "locution": "move",
            "ctx": {**_neutral_ctx(), "has_proposal": False},
            "relation": "supported",  # BASE would be +1.0 if not overridden
        },
        {
            "name": "CONFIRM.assert without has_proposal",
            "stage": "CONFIRM", "locution": "assert",
            "ctx": {**_neutral_ctx(), "has_proposal": False},
            "relation": "supported",
        },
    ]
    rows = []
    n_fail = 0
    for c in cases:
        actual = align_reward(c["relation"], None, c["stage"], c["locution"], c["ctx"])
        ok = abs(actual - (-1.0)) < 1e-9
        if not ok:
            n_fail += 1
        rows.append({"name": c["name"], "expected": -1.0, "actual": actual, "pass": ok})
    return {"name": "Precondition overrides", "rows": rows, "n_fail": n_fail, "pass": n_fail == 0}


# ---------------------------------------------------------------------------
# Section 3: USERRESP (SS4) + soft cap + clip
# ---------------------------------------------------------------------------
def check_userresp() -> dict[str, Any]:
    # (user_locution, stage, locution, expected_userresp_contribution)
    cases = [
        ("ask_justify", "INFORM", "assert", 1.0),
        ("ask_justify", "INFORM", "propose", 1.0),
        ("ask_justify", "CONSIDER", "prefer", -1.0),
        ("reject", "REVISE", "propose", 1.0),
        ("reject", "CONSIDER", "assert", 1.0),
        ("reject", "RECOMMEND", "move", -1.0),
        ("reject", "CONFIRM", "assert", -1.0),
        ("prefer", "CONSIDER", "assert", 0.5),
        ("prefer", "RECOMMEND", "move", 0.5),
        ("prefer", "INFORM", "assert", 0.0),
        ("propose", "CONSIDER", "assert", 0.5),
        ("assert", "PROPOSE", "propose", 0.5),
        ("propose", "CLOSE", "withdraw_dialogue", -0.5),
        (None, "INFORM", "assert", 0.0),
    ]
    rows = []
    n_fail = 0
    for user_loc, stage, loc, expected_ur in cases:
        ctx = _neutral_ctx()
        base = GOLDEN_BASE.get("mixed", {}).get((stage, loc), 0.0)  # baseline before USERRESP is added
        actual = align_reward("mixed", user_loc, stage, loc, ctx)
        expected_score = max(-1.0, min(1.0, base + 0.3 * expected_ur))
        ok = abs(actual - expected_score) < 1e-9
        if not ok:
            n_fail += 1
        rows.append({
            "user_locution": user_loc, "stage": stage, "locution": loc,
            "base": base, "expected_userresp": expected_ur,
            "expected_score": expected_score, "actual": actual, "pass": ok,
        })

    # soft cap: supported + repeated ask_justify -> base capped at <= -0.5
    cap_ctx = {**_neutral_ctx(), "turns_in_current_stage": 1}
    cap_score = align_reward("supported", None, "INFORM", "ask_justify", cap_ctx)
    cap_ok = cap_score <= -0.5 + 1e-9
    if not cap_ok:
        n_fail += 1
    rows.append({
        "user_locution": None, "stage": "INFORM", "locution": "ask_justify",
        "base": "soft-cap (supported, turns_in_current_stage>=1)", "expected_userresp": "<=-0.5",
        "expected_score": "<=-0.5", "actual": cap_score, "pass": cap_ok,
    })

    # clip: a contrived score must not exceed [-1, 1]
    clip_score = align_reward("supported", "ask_justify", "RECOMMEND", "move", _neutral_ctx())
    clip_ok = -1.0 - 1e-9 <= clip_score <= 1.0 + 1e-9
    if not clip_ok:
        n_fail += 1
    rows.append({
        "user_locution": "ask_justify", "stage": "RECOMMEND", "locution": "move",
        "base": "clip range check", "expected_userresp": "n/a",
        "expected_score": "in [-1,1]", "actual": clip_score, "pass": clip_ok,
    })
    return {"name": "USERRESP + soft cap + clip", "rows": rows, "n_fail": n_fail, "pass": n_fail == 0}


# ---------------------------------------------------------------------------
# Section 4: ctx_from_history reconstruction
# ---------------------------------------------------------------------------
def _history_with(case_id: str, medical_actions: list[str]) -> DialogueHistory:
    h = DialogueHistory(case_id=case_id)
    for a in medical_actions:
        h.add_turn("medical", f"<{a}>", action=a)
    return h


def check_ctx_from_history() -> dict[str, Any]:
    cases = [
        {
            "name": "empty history",
            "actions": [],
            "expected": {"has_proposal": False, "has_recommend": False, "num_evaluated_options": 0,
                         "prev_stage": None, "turns_in_current_stage": 0},
        },
        {
            "name": "single PROPOSE",
            "actions": ["PROPOSE.propose"],
            "expected": {"has_proposal": True, "has_recommend": False, "num_evaluated_options": 0,
                         "prev_stage": "PROPOSE", "turns_in_current_stage": 0},
        },
        {
            "name": "PROPOSE then two CONSIDER (consecutive)",
            "actions": ["PROPOSE.propose", "CONSIDER.assert", "CONSIDER.prefer"],
            "expected": {"has_proposal": True, "has_recommend": False, "num_evaluated_options": 2,
                         "prev_stage": "CONSIDER", "turns_in_current_stage": 1},
        },
        {
            "name": "full happy path incl. RECOMMEND",
            "actions": ["PROPOSE.propose", "CONSIDER.assert", "RECOMMEND.move"],
            "expected": {"has_proposal": True, "has_recommend": True, "num_evaluated_options": 1,
                         "prev_stage": "RECOMMEND", "turns_in_current_stage": 0},
        },
        {
            "name": "three consecutive same-stage runs",
            "actions": ["INFORM.assert", "INFORM.assert", "INFORM.assert"],
            "expected": {"has_proposal": False, "has_recommend": False, "num_evaluated_options": 0,
                         "prev_stage": "INFORM", "turns_in_current_stage": 2},
        },
    ]
    rows = []
    n_fail = 0
    for c in cases:
        h = _history_with("synthetic", c["actions"])
        ctx = ctx_from_history(h)
        diffs = {k: (v, ctx.get(k)) for k, v in c["expected"].items() if ctx.get(k) != v}
        ok = not diffs
        if not ok:
            n_fail += 1
        rows.append({"name": c["name"], "actions": c["actions"], "diffs": diffs, "pass": ok})
    return {"name": "ctx_from_history reconstruction", "rows": rows, "n_fail": n_fail, "pass": n_fail == 0}


# ---------------------------------------------------------------------------
# Section 5: R(tau) trajectory_return + group_advantages
# ---------------------------------------------------------------------------
def check_trajectory_return() -> dict[str, Any]:
    w = DEFAULT_WEIGHTS
    cases = [
        {
            "name": "short correct trajectory (low burden)",
            "step_align": [0.5, 0.7, 1.0], "step_fmt": [1.0, 1.0, 1.0],
            "step_burden": [1.5, 2.0, 1.0], "is_correct": True, "num_turns": 3,
        },
        {
            "name": "long correct trajectory (high burden accumulates)",
            "step_align": [0.2] * 9, "step_fmt": [1.0] * 9,
            "step_burden": [3.5] * 9, "is_correct": True, "num_turns": 9,
        },
        {
            "name": "incorrect trajectory, all negative align",
            "step_align": [-1.0, -0.7], "step_fmt": [0.0, 1.0],
            "step_burden": [4.0, 4.5], "is_correct": False, "num_turns": 2,
        },
    ]
    rows = []
    n_fail = 0
    returns = []
    for c in cases:
        traj = Trajectory(step_align=c["step_align"], step_fmt=c["step_fmt"],
                           step_burden=c["step_burden"],
                           is_correct=c["is_correct"], num_turns=c["num_turns"])
        actual = trajectory_return(traj, {})
        r_align = sum(c["step_align"]); r_fmt = sum(c["step_fmt"])
        r_final = 1.0 if c["is_correct"] else 0.0
        burden_cost = sum(c["step_burden"])
        expected = (w["lambda_align"] * r_align + w["lambda_final"] * r_final
                    + w["lambda_fmt"] * r_fmt - w["lambda_burden"] * burden_cost)
        ok = abs(actual - expected) < 1e-9
        if not ok:
            n_fail += 1
        returns.append(actual)
        rows.append({"name": c["name"], "expected": round(expected, 4), "actual": round(actual, 4),
                      "burden_cost": round(burden_cost, 4), "pass": ok})

    # weights override
    traj = Trajectory(step_align=[1.0, 1.0], step_fmt=[0.0, 0.0],
                      step_burden=[0.0, 0.0], is_correct=True, num_turns=2)
    custom = {"lambda_align": 2.0, "lambda_final": 0.0, "lambda_fmt": 0.0, "lambda_burden": 0.0}
    actual = trajectory_return(traj, custom)
    expected = 2.0 * 2.0
    ok = abs(actual - expected) < 1e-9
    if not ok:
        n_fail += 1
    rows.append({"name": "weights override (lambda_align=2.0, others 0)", "expected": expected,
                 "actual": actual, "burden_cost": 0.0, "pass": ok})

    # group_advantages: mean ~0, pstdev ~1
    adv = group_advantages(returns)
    adv_mean = sum(adv) / len(adv)
    adv_ok = abs(adv_mean) < 1e-6
    if not adv_ok:
        n_fail += 1
    rows.append({"name": "group_advantages mean(A)~=0", "expected": 0.0, "actual": round(adv_mean, 6), "pass": adv_ok})

    # degenerate: single-element / all-equal -> sd=0 path uses eps, no div-by-zero
    degenerate = group_advantages([0.42])
    deg_ok = degenerate == [0.0]
    if not deg_ok:
        n_fail += 1
    rows.append({"name": "group_advantages single-element degenerate", "expected": [0.0],
                 "actual": degenerate, "pass": deg_ok})

    all_equal = group_advantages([1.0, 1.0, 1.0])
    eq_ok = all(abs(x) < 1e-3 for x in all_equal)
    if not eq_ok:
        n_fail += 1
    rows.append({"name": "group_advantages all-equal degenerate", "expected": "~[0,0,0]",
                 "actual": [round(x, 4) for x in all_equal], "pass": eq_ok})

    return {
        "name": "R(tau) trajectory_return + group_advantages", "rows": rows, "n_fail": n_fail,
        "pass": n_fail == 0, "chart_returns": [round(r, 4) for r in returns], "chart_advantages": [round(a, 4) for a in adv],
        "chart_labels": [c["name"] for c in cases],
    }


# ---------------------------------------------------------------------------
# Section 6: live dialogue integration (observational, optional)
# ---------------------------------------------------------------------------
def check_live_integration(results_path: Path) -> dict[str, Any] | None:
    if not results_path.exists():
        return None
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    records = data.get("records", [])
    if not records:
        return None

    by_condition: dict[str, list[float]] = {}
    missing = {"total_steps": 0, "missing_reward": 0, "out_of_range": 0}
    for r in records:
        cond = r.get("condition")
        for s in r.get("steps", []):
            missing["total_steps"] += 1
            rv = s.get("r_align")
            if rv is None:
                missing["missing_reward"] += 1
                continue
            if not (-1.0 - 1e-9 <= rv <= 1.0 + 1e-9):
                missing["out_of_range"] += 1
            by_condition.setdefault(cond, []).append(rv)

    means = {c: round(sum(v) / len(v), 4) for c, v in by_condition.items() if v}
    warnings = []
    if missing["missing_reward"] > 0:
        warnings.append(f"{missing['missing_reward']}/{missing['total_steps']} steps missing r_align")
    if missing["out_of_range"] > 0:
        warnings.append(f"{missing['out_of_range']} steps with r_align outside [-1,1]")

    baseline_mean = means.get("baseline")
    for cond in ("hac_command", "hac_reference"):
        if cond in means and baseline_mean is not None and means[cond] < baseline_mean:
            warnings.append(f"{cond} mean r_align ({means[cond]}) < baseline ({baseline_mean}) -- unexpected but not a hard failure")

    return {
        "name": "Live dialogue integration",
        "source": str(results_path),
        "n_records": len(records),
        "missing": missing,
        "means": means,
        "distributions": by_condition,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
_CSS = """\
body { font-family: -apple-system, Segoe UI, sans-serif; background: #f4f5f7; margin: 0; padding: 24px; color: #1f2430; }
h1 { font-size: 20px; }
h2 { font-size: 15px; margin: 18px 0 8px; }
.banner { font-size: 16px; font-weight: 600; padding: 12px 18px; border-radius: 8px; margin-bottom: 18px; }
.banner.pass { background: #e6f4ea; color: #1a7f37; }
.banner.fail { background: #fde2e1; color: #a3231f; }
.section { background: #fff; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
table { border-collapse: collapse; font-size: 12.5px; margin-bottom: 10px; width: 100%; }
th, td { border: 1px solid #e1e4e8; padding: 4px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.ok { color: #1a7f37; font-weight: 600; } .bad { color: #a3231f; font-weight: 600; }
.heat-pos { background: #e6f4ea; } .heat-neg { background: #fde2e1; } .heat-zero { background: #f4f5f7; }
.warn { color: #8a6400; background: #fff3cd; padding: 6px 10px; border-radius: 6px; font-size: 12.5px; margin: 4px 0; }
.chart-label { font-size: 11px; fill: #555; }
.muted { color: #888; font-size: 12.5px; }
"""


def _e(v) -> str:
    return html.escape(str(v if v is not None else ""))


def _heat_class(v) -> str:
    if not isinstance(v, (int, float)):
        return ""
    if v > 0.05:
        return "heat-pos"
    if v < -0.05:
        return "heat-neg"
    return "heat-zero"


def _pass_badge(ok: bool) -> str:
    return '<span class="ok">PASS</span>' if ok else '<span class="bad">FAIL</span>'


def _svg_bar_chart(labels: list[str], values: list[float], width: int = 700, bar_h: int = 22, vmin: float = -1.5, vmax: float = 1.5) -> str:
    if not values:
        return "<div class='muted'>(no data)</div>"
    h = bar_h * len(values) + 20
    zero_x = width * (0 - vmin) / (vmax - vmin)
    bars = []
    for i, (lbl, v) in enumerate(zip(labels, values)):
        y = i * bar_h + 4
        x0 = width * (min(0, v) - vmin) / (vmax - vmin)
        x1 = width * (max(0, v) - vmin) / (vmax - vmin)
        color = "#1a7f37" if v >= 0 else "#a3231f"
        bars.append(
            f'<rect x="{x0:.1f}" y="{y}" width="{max(1, x1 - x0):.1f}" height="{bar_h - 4}" fill="{color}" opacity="0.75"/>'
            f'<text x="{x1 + 4 if v >= 0 else x0 - 4}" y="{y + bar_h - 8}" font-size="11" '
            f'text-anchor="{"start" if v >= 0 else "end"}" class="chart-label">{v:.3f}</text>'
            f'<text x="2" y="{y - 1}" font-size="10" class="chart-label">{_e(lbl)[:60]}</text>'
        )
    return (
        f'<svg width="{width}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="{zero_x:.1f}" y1="0" x2="{zero_x:.1f}" y2="{h}" stroke="#ccc" stroke-width="1"/>'
        + "".join(bars) + "</svg>"
    )


def _svg_histogram(values: list[float], bins: int = 10, width: int = 320, height: int = 100, color: str = "#3366cc") -> str:
    if not values:
        return "<div class='muted'>(no data)</div>"
    lo, hi = -1.0, 1.0
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
        counts[idx] += 1
    maxc = max(counts) or 1
    bw = width / bins
    bars = []
    for i, c in enumerate(counts):
        bh = height * c / maxc
        bars.append(f'<rect x="{i * bw + 1:.1f}" y="{height - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.8"/>')
    return f'<svg width="{width}" height="{height + 14}" xmlns="http://www.w3.org/2000/svg">{"".join(bars)}<line x1="0" y1="{height}" x2="{width}" y2="{height}" stroke="#ccc"/></svg>'


def _render_base_table(check: dict) -> str:
    actions_seen = sorted({(r["stage"], r["locution"]) for r in check["rows"]})
    by_relation: dict[str, dict] = {}
    for r in check["rows"]:
        by_relation.setdefault(r["relation"], {})[(r["stage"], r["locution"])] = r
    header = "".join(f"<th>{_e(s)}.{_e(l)}</th>" for s, l in actions_seen)
    body = ""
    for relation in _RELATIONS:
        cells = ""
        for st, loc in actions_seen:
            cell = by_relation.get(relation, {}).get((st, loc))
            if cell is None:
                cells += "<td>-</td>"
                continue
            cls = _heat_class(cell["actual"])
            mark = "" if cell["pass"] else " <span class='bad'>!=</span>"
            cells += f"<td class='{cls}'>{cell['actual']:.1f}{mark}</td>"
        body += f"<tr><th>{_e(relation)}</th>{cells}</tr>"
    return f"<table><tr><th>relation \\ action</th>{header}</tr>{body}</table>"


def _render_generic_rows(rows: list[dict]) -> str:
    if not rows:
        return "<div class='muted'>(no rows)</div>"
    keys = [k for k in rows[0].keys() if k != "pass"]
    header = "".join(f"<th>{_e(k)}</th>" for k in keys) + "<th>result</th>"
    body = ""
    for r in rows:
        cells = "".join(f"<td>{_e(r.get(k))}</td>" for k in keys)
        body += f"<tr>{cells}<td>{_pass_badge(r['pass'])}</td></tr>"
    return f"<table><tr>{header}</tr>{body}</table>"


def render_html(sections: list[dict], live: dict | None) -> str:
    total_fail = sum(s["n_fail"] for s in sections)
    overall_pass = total_fail == 0
    banner = (
        f'<div class="banner pass">ALL CONFORMANCE CHECKS PASS ({sum(len(s["rows"]) for s in sections)} checks)</div>'
        if overall_pass else
        f'<div class="banner fail">{total_fail} CONFORMANCE CHECK(S) FAILED</div>'
    )

    parts = [banner]
    for i, s in enumerate(sections, start=1):
        if s["name"] == "BASE table conformance":
            body = _render_base_table(s)
        else:
            body = _render_generic_rows(s["rows"])
        extra = ""
        if "chart_returns" in s:
            extra += "<h3 style='font-size:13px;margin:10px 0 4px;'>trajectory_return per case</h3>"
            extra += _svg_bar_chart(s["chart_labels"], s["chart_returns"])
            extra += "<h3 style='font-size:13px;margin:10px 0 4px;'>group_advantages</h3>"
            extra += _svg_bar_chart(s["chart_labels"], s["chart_advantages"], vmin=-2.5, vmax=2.5)
        parts.append(
            f'<div class="section"><h2>{i}. {_e(s["name"])} — {_pass_badge(s["pass"])} '
            f'({len(s["rows"]) - s["n_fail"]}/{len(s["rows"])})</h2>{body}{extra}</div>'
        )

    if live is None:
        parts.append(
            '<div class="section"><h2>6. Live dialogue integration</h2>'
            '<div class="muted">No outputs/poc_multiturn/results.json found (or no records) — run the PoC first to populate this section.</div></div>'
        )
    else:
        warn_html = "".join(f'<div class="warn">{_e(w)}</div>' for w in live["warnings"]) or "<div class='muted'>No warnings.</div>"
        means_rows = "".join(
            f"<tr><td>{_e(c)}</td><td>{_e(m)}</td><td>{len(live['distributions'].get(c, []))}</td></tr>"
            for c, m in live["means"].items()
        )
        hist_html = "".join(
            f"<div style='display:inline-block;margin:6px 12px;'><div class='muted'>{_e(c)} (n={len(v)})</div>{_svg_histogram(v)}</div>"
            for c, v in live["distributions"].items()
        )
        bar_html = _svg_bar_chart(list(live["means"].keys()), list(live["means"].values()))
        parts.append(
            f'<div class="section"><h2>6. Live dialogue integration (observational)</h2>'
            f'<div class="muted">source: {_e(live["source"])} · {live["n_records"]} records · '
            f'{live["missing"]["total_steps"]} total steps</div>'
            f'{warn_html}'
            f"<table><tr><th>condition</th><th>mean r_align</th><th>n_steps</th></tr>{means_rows}</table>"
            f"<h3 style='font-size:13px;margin:10px 0 4px;'>mean r_align by condition</h3>{bar_html}"
            f"<h3 style='font-size:13px;margin:10px 0 4px;'>r_align distributions</h3>{hist_html}"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>verify_reward report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Reward Verification Report (r_align + R(tau))</h1>
{''.join(parts)}
</body>
</html>
"""


def main(results_path_arg: str) -> int:
    action_space = load_action_space()
    actions = valid_actions(action_space)

    sections = [
        check_base_table(actions),
        check_preconditions(),
        check_userresp(),
        check_ctx_from_history(),
        check_trajectory_return(),
    ]
    live = check_live_integration(Path(results_path_arg))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps({"sections": sections, "live": live}, indent=2, default=str))
    (OUTPUT_DIR / "report.html").write_text(render_html(sections, live))

    total_fail = sum(s["n_fail"] for s in sections)
    total_checks = sum(len(s["rows"]) for s in sections)
    print(f"Reward verification: {total_checks - total_fail}/{total_checks} checks passed.")
    for s in sections:
        status = "PASS" if s["pass"] else f"FAIL ({s['n_fail']})"
        print(f"  [{status}] {s['name']}")
    if live:
        for w in live["warnings"]:
            print(f"  [warn] {w}")
    else:
        print("  (no outputs/poc_multiturn/results.json found — section 6 skipped)")
    print(f"Report: {OUTPUT_DIR / 'report.html'}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/poc_multiturn/results.json")
    args = parser.parse_args()
    sys.exit(main(args.results))
