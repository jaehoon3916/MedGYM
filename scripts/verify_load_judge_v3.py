#!/usr/bin/env python3
"""
POC: validate the v3 NASA-TLX cognitive-burden judge (prompts/user_simulator_v3.yaml's
burden_judge_* templates, plugins/user_llm/user_simulator/v3_burden.py) BEFORE trusting it as
a reward signal.

Extends scripts/verify_load_judge.py's known-groups methodology for v1's third-person 0-3
judge to v3's first-person, persona-aware, 1-5 NASA-TLX overall_workload judge:

  1. Ordinal validity      — does judge overall_workload rank-correlate with the designed
                             burden ladder (agree=0 -> minor note=1 -> surface alternative=2
                             -> demand justify=3)? Spearman rho, computed per-persona AND
                             pooled across all (item, persona) pairs.
  2. Discriminant validity — for matched-posture pairs differing ONLY in medical correctness,
                             overall_workload should be ~equal. Mean |delta| within pairs,
                             pooled across personas.
  3. Convergent validity   — overall_workload grouped by the DDF action each turn embodies
                             should follow the friction ordering (CONFIRM/RECOMMEND <
                             INFORM.assert < CONSIDER < ask_justify/REVISE), pooled across
                             personas.
  4. Persona-sensitivity   — NEW vs the v1 harness, since v3's judge is explicitly persona-
     validity               aware: for the same item, SENSITIVE personas (exhausted_attending,
                             burned_out_resident) should not self-report LOWER overall_workload
                             than ROBUST personas (veteran_attending, eager_resident), on items
                             where there is something to be sensitive/robust TO (expected_rank
                             > 0; "agree" items have nothing to react to either way).

No re-rollout: the only LLM calls are judge calls on the canonical (item, persona) pairs. Same
8 canonical items/scenario/doctor-utterance as v1's harness for ladder continuity, now
replicated across all 4 personas (32 (item, persona) pairs total).

Also runs an ORACLE generation+recovery check (complementary, not a replacement): the
known-groups check above only validates that overall_workload tracks a single externally-
imposed ladder label -- it does NOT validate that the judge discriminates the 5 individual
NASA-TLX dimensions from each other (a known LLM-judge failure mode: collapsing all 5 axes
into one "halo" gut-feeling number). The oracle check generates NEW AI utterances via an LLM,
each one deliberately engineered to target a SPECIFIED 5-dimension profile (prompts/
burden_oracle_v3.yaml), then judges that generated utterance with the SAME burden_judge_*
prompt and checks whether the judge's measured profile recovers the intended target. Single
fixed persona (config: oracle_persona), since this question is about the judge's per-axis
text-discrimination ability, which check #4 above already establishes is persona-modulated
only on the aggregate overall_workload, not (a priori) on cross-axis discrimination.

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    conda activate medgym; set -a; source ../.env; set +a; export PYTHONNOUSERSITE=1
    python scripts/verify_load_judge_v3.py --config configs/verify_load_judge_v3.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.token_tracker import tracker
from plugins.user_llm.user_simulator.v3_burden import (
    _TLX_DIMS,
    _parse_burden_judge,
    build_burden_judge_prompt_v3,
)

# ── Known-groups fixture (same base content as scripts/verify_load_judge.py, for ladder
# continuity/comparability) ──────────────────────────────────────────────────────────────

_SCENARIO = (
    "62yo man, 3 days of progressive dyspnea and bilateral leg edema, history of HTN, mild "
    "bibasilar crackles, borderline-high BNP, ambiguous CXR congestion, mildly elevated D-dimer. "
    "Differential: H1 congestive heart failure (CHF) vs H2 pulmonary embolism (PE)."
)
_DOCTOR = "I'm fairly confident this is heart failure — I'll start IV diuresis and admit for CHF."
_DIALOGUE_TEXT = f"[Doctor]: {_DOCTOR}"

_ITEMS: list[dict[str, Any]] = [
    {
        "id": "agree_confirm", "expected_rank": 0, "friction_tag": "CONFIRM.assert", "validity": "incorrect",
        "ai": "Agreed — the bilateral edema and elevated BNP fit heart failure well. Starting IV diuresis is reasonable; let's proceed.",
    },
    {
        "id": "agree_recommend", "expected_rank": 0, "friction_tag": "RECOMMEND.move", "validity": "incorrect",
        "ai": "That makes sense. I'd confirm the CHF plan and admit for diuresis and daily weights — no objection from me.",
    },
    {
        "id": "agree_minor_note", "expected_rank": 1, "friction_tag": "INFORM.assert", "validity": "incorrect",
        "ai": "Heart failure is a reasonable read and diuresis is fine. I'd just note the D-dimer is also mildly up, but your assessment holds.",
    },
    {
        "id": "surface_alt_pe", "expected_rank": 2, "friction_tag": "CONSIDER.assert", "validity": "correct",
        "ai": "Heart failure is plausible, but the acute onset and elevated D-dimer also fit PE — I'd weigh a CTPA before committing to diuresis alone.",
    },
    {
        "id": "surface_alt_pneu", "expected_rank": 2, "friction_tag": "CONSIDER.assert", "validity": "incorrect",
        "ai": "Heart failure is plausible, but the bibasilar crackles also fit pneumonia — I'd weigh a chest CT for infection before committing to diuresis alone.",
    },
    {
        "id": "ask_justify_pe", "expected_rank": 3, "friction_tag": "INFORM.ask_justify", "validity": "correct",
        "ai": "Before we commit to diuresis — what specifically rules out PE here, given the acute dyspnea and the raised D-dimer? Walk me through why those don't change the plan.",
    },
    {
        "id": "ask_justify_pneu", "expected_rank": 3, "friction_tag": "INFORM.ask_justify", "validity": "incorrect",
        "ai": "Before we commit to diuresis — what specifically rules out pneumonia here, given the crackles and the dyspnea? Walk me through why those don't change the plan.",
    },
    {
        "id": "oppose_revise_pe", "expected_rank": 3, "friction_tag": "REVISE.propose", "validity": "correct",
        "ai": "I'd hold the CHF-only plan. The acute course and D-dimer push me toward PE first — let's revise to get a CTPA, and treat for PE if positive.",
    },
]

_FRICTION_ORDER = {
    "CONFIRM.assert": 0, "RECOMMEND.move": 0,
    "INFORM.assert": 1,
    "CONSIDER.assert": 2,
    "INFORM.ask_justify": 3, "REVISE.propose": 3,
}

_DISCRIMINANT_PAIRS = [("surface_alt_pe", "surface_alt_pneu"), ("ask_justify_pe", "ask_justify_pneu")]

_PERSONAS = ("veteran_attending", "exhausted_attending", "eager_resident", "burned_out_resident")
_ROBUST_PERSONAS = ("veteran_attending", "eager_resident")
_SENSITIVE_PERSONAS = ("exhausted_attending", "burned_out_resident")

# Pass thresholds.
_SPEARMAN_MIN = 0.80
# v1's threshold (1.0) was on a 0-3 scale (range 3) -- this is the same proportion of the
# full range (1/3) translated to v3's 1-5 scale (range 4): 1.0/3 * 4 ~= 1.33.
_DISCRIMINANT_MAX = 1.33
# TODO: calibrate empirically once a first real run's numbers are seen -- left at the
# loosest possible "not inverted" bound (0.0) rather than guessing a margin with zero data.
_PERSONA_SENSITIVITY_MIN_DELTA = 0.0

# ── Oracle generation+recovery fixture ──────────────────────────────────────────
# 5 "uniform ladder" profiles: rung k targets ALL 5 dims at level k (1..5), mirroring the
# known-groups check's expected_rank 0-3 ladder but built via generation instead of
# hand-written text -- does the SAME burden judge recover a monotonic ladder out of LLM-
# generated utterances explicitly engineered to hit each rung?
_LADDER_PROFILES: list[dict[str, Any]] = [
    {"id": f"ladder_{level}", "level": level, "target": {d: level for d in _TLX_DIMS}}
    for level in (1, 2, 3, 4, 5)
]

# TODO: calibrate empirically once a first real run's numbers are seen -- loosest defensible
# bound (half the 1-5 scale's range), same "no guessed tight number" philosophy as
# _PERSONA_SENSITIVITY_MIN_DELTA above.
_ORACLE_MAE_MAX = 2.0


def _load_v3_prompts() -> dict:
    """burden_judge_* now lives in prompts/burden_judge.yaml (split out of
    user_simulator_v3.yaml for ease of editing) -- merge both so tmpl[...] lookups stay
    unaffected."""
    with open(_ROOT / "prompts" / "user_simulator_v3.yaml") as f:
        tmpl = yaml.safe_load(f)
    with open(_ROOT / "prompts" / "burden_judge.yaml") as f:
        tmpl.update(yaml.safe_load(f))
    return tmpl


def _load_personas() -> dict:
    personas: dict = {}
    for path in sorted((_ROOT / "source" / "persona").glob("persona_*.yaml")):
        with open(path) as f:
            personas.update(yaml.safe_load(f))
    return personas


def _load_oracle_prompts() -> dict:
    with open(_ROOT / "prompts" / "burden_oracle_v3.yaml") as f:
        return yaml.safe_load(f)


# ── Judge ─────────────────────────────────────────────────────────────────────

async def _judge_once(
    client: AsyncOpenAI, model: str, max_tokens: int, temperature: float,
    tmpl: dict, persona_instruction: str, item: dict,
) -> dict | None:
    messages = build_burden_judge_prompt_v3(
        tmpl=tmpl, scenario=_SCENARIO, persona_instruction=persona_instruction,
        dialogue_text=_DIALOGUE_TEXT, ai_utterance=item["ai"], cumulative_burden=0.0,
    )
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    tracker.record(model, messages, text, resp.usage)
    return _parse_burden_judge(text)


async def judge_item(
    client, model, max_tokens, temperature, samples, sem, tmpl, persona, persona_instruction, item,
) -> dict:
    async def one() -> dict | None:
        async with sem:
            return await _judge_once(client, model, max_tokens, temperature, tmpl, persona_instruction, item)

    parsed = [p for p in await asyncio.gather(*(one() for _ in range(samples))) if p is not None]
    overall_workloads = [p["overall_workload"] for p in parsed]
    mean = round(statistics.mean(overall_workloads), 3) if overall_workloads else None
    return {
        **{k: item[k] for k in ("id", "expected_rank", "friction_tag", "validity")},
        "persona": persona, "ai": item["ai"],
        "overall_workloads": overall_workloads, "overall_workload_mean": mean,
        "n_ok": len(parsed), "n_attempted": samples,
    }


# ── Oracle generation+recovery ─────────────────────────────────────────────────

def build_oracle_generator_prompt(
    tmpl: dict, scenario: str, dialogue_text: str, persona_instruction: str, target: dict[str, int],
) -> list[dict[str, str]]:
    """Pure function -- builds the generator's messages from a target NASA-TLX profile.
    `tmpl` is the loaded prompts/burden_oracle_v3.yaml dict."""
    ctx = {**target, "scenario": scenario, "dialogue_text": dialogue_text, "persona_instruction": persona_instruction}
    return [
        {"role": "system", "content": tmpl["generator_system"].format(**ctx)},
        {"role": "user", "content": tmpl["generator_user"].format(**ctx)},
    ]


def _parse_oracle_generation(raw: str) -> dict[str, Any] | None:
    """All-or-nothing, mirrors _parse_burden_judge's philosophy: requires a non-empty
    `ai_utterance` AND a `design_rationale` dict with all 5 _TLX_DIMS keys present as
    non-empty strings. Returns None (the caller treats this as a generation failure to be
    retried, then ultimately recorded as a failure -- never silently faked) otherwise."""
    from core.json_utils import safe_json_load

    data = safe_json_load(raw)
    utterance = data.get("ai_utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        return None
    rationale = data.get("design_rationale")
    if not isinstance(rationale, dict):
        return None
    cleaned_rationale = {}
    for dim in _TLX_DIMS:
        text = rationale.get(dim)
        if not isinstance(text, str) or not text.strip():
            return None
        cleaned_rationale[dim] = text.strip()
    return {"ai_utterance": utterance.strip(), "design_rationale": cleaned_rationale}


async def run_oracle_profile(
    client: AsyncOpenAI, gen_cfg: dict, judge_cfg: dict, judge_samples: int, sem: asyncio.Semaphore,
    gen_tmpl: dict, judge_tmpl: dict, persona: str, persona_instruction: str, profile: dict,
) -> dict:
    """Generate ONE utterance targeting `profile["target"]`, then judge it `judge_samples`
    times with the SAME burden_judge_* prompt used elsewhere (same top-level
    `samples_per_item` config the known-groups check uses, passed in explicitly rather than
    read from `judge_cfg` -- `samples_per_item` lives at the top level of the config, not
    nested under `judge:`).
    Returns a result row with target/observed/ai_utterance/design_rationale, or
    generation_failed=True (never a faked utterance) if generation exhausts its retries."""
    gen_messages = build_oracle_generator_prompt(
        tmpl=gen_tmpl, scenario=_SCENARIO, dialogue_text=_DIALOGUE_TEXT,
        persona_instruction=persona_instruction, target=profile["target"],
    )
    generation = None
    for _ in range(int(gen_cfg.get("max_retries", 3))):
        async with sem:
            resp = await client.chat.completions.create(
                model=gen_cfg["model"], messages=gen_messages,
                temperature=float(gen_cfg.get("temperature", 0.7)),
                max_tokens=int(gen_cfg.get("max_tokens", 400)),
                response_format={"type": "json_object"},
            )
        text = resp.choices[0].message.content or ""
        tracker.record(gen_cfg["model"], gen_messages, text, resp.usage)
        generation = _parse_oracle_generation(text)
        if generation is not None:
            break

    if generation is None:
        return {**profile, "persona": persona, "generation_failed": True,
                "ai_utterance": None, "design_rationale": None, "observed": None, "overall_workload_mean": None,
                "judge_rationale": None, "n_judge_ok": 0, "n_judge_attempted": 0}

    async def judge_once() -> dict | None:
        async with sem:
            return await _judge_once(
                client, judge_cfg["model"], int(judge_cfg.get("max_tokens", 256)),
                float(judge_cfg.get("temperature", 0.0)), judge_tmpl, persona_instruction,
                {"ai": generation["ai_utterance"]},
            )

    parsed = [p for p in await asyncio.gather(*(judge_once() for _ in range(judge_samples))) if p is not None]
    observed = {d: round(statistics.mean(p[d] for p in parsed), 3) for d in _TLX_DIMS} if parsed else None
    overall_mean = round(statistics.mean(p["overall_workload"] for p in parsed), 3) if parsed else None
    # Representative judge reasoning -- one sample's short_rationale (illustrative only, no
    # principled way to combine 5 dims' worth of free text across samples; mirrors v3.py's
    # _score_burden_tlx "most recent sample's rationale" precedent). Previously discarded here
    # entirely even though the judge always returns it -- now kept so it can be shown in HTML.
    judge_rationale = parsed[0]["short_rationale"] if parsed else None
    return {
        **profile, "persona": persona, "generation_failed": False,
        "ai_utterance": generation["ai_utterance"], "design_rationale": generation["design_rationale"],
        "observed": observed, "overall_workload_mean": overall_mean, "judge_rationale": judge_rationale,
        "n_judge_ok": len(parsed), "n_judge_attempted": judge_samples,
    }


def compute_oracle_metrics(results: list[dict]) -> dict:
    """Ordinal validity (does the ladder's target level rank-correlate with judge
    overall_workload?) + per-dim/pooled MAE (does the ABSOLUTE scale match, not just the
    order?). No dominant-driver/anchor checks -- uniform-target profiles have no single
    "spiked" axis, so that concept doesn't apply here."""
    ok = [r for r in results if not r["generation_failed"] and r["observed"] is not None]
    n_generation_failed = sum(1 for r in results if r["generation_failed"])

    levels = [r["level"] for r in ok]
    overall_workloads = [r["overall_workload_mean"] for r in ok]
    ordinal_rho = _spearman(levels, overall_workloads) if len(ok) > 1 else 0.0
    ordinal_pass = ordinal_rho >= _SPEARMAN_MIN

    per_dim_errors: dict[str, list[float]] = {d: [] for d in _TLX_DIMS}
    for r in ok:
        for d in _TLX_DIMS:
            per_dim_errors[d].append(abs(r["target"][d] - r["observed"][d]))
    per_dim_mae = {d: round(statistics.mean(v), 3) for d, v in per_dim_errors.items() if v}
    pooled_mae = round(statistics.mean(e for v in per_dim_errors.values() for e in v), 3) if ok else None
    mae_pass = pooled_mae is not None and pooled_mae <= _ORACLE_MAE_MAX

    return {
        "ordinal": {"spearman": ordinal_rho, "threshold": _SPEARMAN_MIN, "pass": ordinal_pass, "n": len(ok)},
        "mae": {"per_dim": per_dim_mae, "pooled": pooled_mae, "threshold": _ORACLE_MAE_MAX, "pass": mae_pass},
        "n_generation_failed": n_generation_failed,
        "all_pass": bool(ordinal_pass and mae_pass),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho = Pearson on ranks. Average ranks for ties."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else 0.0


def compute_metrics(results: list[dict]) -> dict:
    scored = [r for r in results if r["overall_workload_mean"] is not None]
    by_persona_id = {(r["persona"], r["id"]): r for r in scored}

    # 1. Ordinal validity -- per persona + pooled.
    ordinal_by_persona = {}
    for persona in _PERSONAS:
        rows = [r for r in scored if r["persona"] == persona]
        rho = _spearman([r["expected_rank"] for r in rows], [r["overall_workload_mean"] for r in rows]) if rows else 0.0
        ordinal_by_persona[persona] = {"spearman": rho, "n": len(rows)}
    pooled_rho = _spearman([r["expected_rank"] for r in scored], [r["overall_workload_mean"] for r in scored])
    ordinal_pass = pooled_rho >= _SPEARMAN_MIN

    # 2. Discriminant validity -- pooled across personas.
    pair_diffs = []
    for persona in _PERSONAS:
        for a, b in _DISCRIMINANT_PAIRS:
            ra, rb = by_persona_id.get((persona, a)), by_persona_id.get((persona, b))
            if ra and rb:
                pair_diffs.append({
                    "persona": persona, "pair": [a, b],
                    "delta": round(abs(ra["overall_workload_mean"] - rb["overall_workload_mean"]), 3),
                })
    disc_mean = round(statistics.mean(d["delta"] for d in pair_diffs), 3) if pair_diffs else None
    discriminant_pass = disc_mean is not None and disc_mean <= _DISCRIMINANT_MAX

    # 3. Convergent validity -- pooled across personas.
    by_friction: dict[str, list[float]] = {}
    for r in scored:
        by_friction.setdefault(r["friction_tag"], []).append(r["overall_workload_mean"])
    convergent = sorted(
        ({"friction_tag": t, "friction_order": _FRICTION_ORDER.get(t, -1),
          "mean_overall_workload": round(statistics.mean(v), 3), "n": len(v)} for t, v in by_friction.items()),
        key=lambda d: d["friction_order"],
    )
    conv_rho = _spearman([c["friction_order"] for c in convergent], [c["mean_overall_workload"] for c in convergent]) \
        if len(convergent) > 1 else 0.0

    # 4. Persona-sensitivity validity -- NEW for v3. Skip expected_rank==0 ("agree") items,
    # where there is nothing to be sensitive/robust TO.
    reactive = [r for r in scored if r["expected_rank"] > 0]
    sensitive_vals = [r["overall_workload_mean"] for r in reactive if r["persona"] in _SENSITIVE_PERSONAS]
    robust_vals = [r["overall_workload_mean"] for r in reactive if r["persona"] in _ROBUST_PERSONAS]
    sensitive_mean = round(statistics.mean(sensitive_vals), 3) if sensitive_vals else None
    robust_mean = round(statistics.mean(robust_vals), 3) if robust_vals else None
    persona_delta = (
        round(sensitive_mean - robust_mean, 3) if sensitive_mean is not None and robust_mean is not None else None
    )
    persona_sensitivity_pass = persona_delta is not None and persona_delta >= _PERSONA_SENSITIVITY_MIN_DELTA

    return {
        "ordinal": {
            "pooled_spearman": pooled_rho, "threshold": _SPEARMAN_MIN, "pass": ordinal_pass,
            "by_persona": ordinal_by_persona,
        },
        "discriminant": {
            "pairs": pair_diffs, "mean_abs_delta": disc_mean,
            "threshold": _DISCRIMINANT_MAX, "pass": discriminant_pass,
        },
        "convergent": {"by_friction": convergent, "spearman": conv_rho},
        "persona_sensitivity": {
            "sensitive_mean": sensitive_mean, "robust_mean": robust_mean, "delta": persona_delta,
            "threshold_min_delta": _PERSONA_SENSITIVITY_MIN_DELTA, "pass": persona_sensitivity_pass,
        },
        "all_pass": bool(ordinal_pass and discriminant_pass and persona_sensitivity_pass),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _e(v) -> str:
    return html.escape(str(v if v is not None else ""))


def render_oracle_section(oracle_results: list[dict], oracle_metrics: dict) -> str:
    o, m = oracle_metrics["ordinal"], oracle_metrics["mae"]
    badge = lambda ok: f"<span style='color:{'#1a7f37' if ok else '#a3231f'};font-weight:600'>{'PASS' if ok else 'FAIL'}</span>"
    rows = []
    for r in sorted(oracle_results, key=lambda r: r["level"]):
        if r["generation_failed"]:
            rows.append(
                f"<tr><td>{_e(r['id'])}</td><td style='text-align:center'>{r['level']}</td>"
                f"<td colspan='3' style='color:#a3231f'>generation_failed</td></tr>"
            )
            continue
        observed_line = ", ".join(f"{d}={r['observed'][d]}" for d in _TLX_DIMS)
        rationale_line = " / ".join(f"{d}: {r['design_rationale'][d]}" for d in _TLX_DIMS)
        rows.append(
            f"<tr><td>{_e(r['id'])}</td><td style='text-align:center'>{r['level']}</td>"
            f"<td style='text-align:center'><b>{_e(r['overall_workload_mean'])}</b></td>"
            f"<td style='font-size:12px'>{_e(observed_line)}</td>"
            f"<td style='font-size:12px'>{_e(r['ai_utterance'])}</td>"
            f"<td style='font-size:11px;color:#555'>{_e(rationale_line)}</td></tr>"
        )
    per_dim = "".join(f"<li>{_e(d)}: MAE={v}</li>" for d, v in m["per_dim"].items())
    return f"""
<h1 style="margin-top:32px">v3 NASA-TLX Burden Judge — Oracle Generation+Recovery Validation</h1>
<div class="summary">
  <p><b>Overall:</b> {badge(oracle_metrics['all_pass'])} &nbsp; (n_generation_failed = {oracle_metrics['n_generation_failed']})</p>
  <p><b>1. Ordinal validity</b> (target ladder level 1-5 ↔ judge overall_workload): Spearman ρ =
     <b>{o['spearman']}</b> (≥ {o['threshold']}, n={o['n']}) → {badge(o['pass'])}</p>
  <p><b>2. MAE</b> (target vs observed, per-dim mean absolute error): pooled =
     <b>{m['pooled']}</b> (≤ {m['threshold']}) → {badge(m['pass'])}</p>
  <ul style="margin:4px 0 0 0;font-size:13px">{per_dim}</ul>
</div>
<h2>Per-rung generated utterances (target uniform across all 5 dims)</h2>
<table><tr><th>rung</th><th>target level</th><th>overall_workload_mean</th><th>observed (5 dims)</th>
<th>generated AI utterance</th><th>design_rationale</th></tr>{"".join(rows)}</table>
"""


def render_html(results: list[dict], metrics: dict, oracle_results: list[dict] | None = None, oracle_metrics: dict | None = None) -> str:
    rows = "".join(
        f"<tr><td>{_e(r['id'])}</td><td>{_e(r['persona'])}</td>"
        f"<td style='text-align:center'>{r['expected_rank']}</td>"
        f"<td style='text-align:center'><b>{_e(r['overall_workload_mean'])}</b></td>"
        f"<td style='text-align:center'>{_e(r['overall_workloads'])}</td>"
        f"<td>{_e(r['friction_tag'])}</td><td>{_e(r['validity'])}</td>"
        f"<td style='font-size:12px'>{_e(r['ai'])}</td></tr>"
        for r in sorted(results, key=lambda r: (r["expected_rank"], r["id"], r["persona"]))
    )
    o, d, p = metrics["ordinal"], metrics["discriminant"], metrics["persona_sensitivity"]
    by_persona_rows = "".join(
        f"<tr><td>{_e(persona)}</td><td style='text-align:center'>{v['spearman']}</td>"
        f"<td style='text-align:center'>{v['n']}</td></tr>"
        for persona, v in o["by_persona"].items()
    )
    conv = "".join(
        f"<tr><td>{_e(c['friction_tag'])}</td><td style='text-align:center'>{c['friction_order']}</td>"
        f"<td style='text-align:center'>{c['mean_overall_workload']}</td><td style='text-align:center'>{c['n']}</td></tr>"
        for c in metrics["convergent"]["by_friction"]
    )
    pairs = "".join(
        f"<li>[{_e(pr['persona'])}] {_e(pr['pair'][0])} vs {_e(pr['pair'][1])}: |Δ| = {pr['delta']}</li>"
        for pr in d["pairs"]
    )
    badge = lambda ok: f"<span style='color:{'#1a7f37' if ok else '#a3231f'};font-weight:600'>{'PASS' if ok else 'FAIL'}</span>"
    oracle_section = (
        render_oracle_section(oracle_results, oracle_metrics)
        if oracle_results is not None and oracle_metrics is not None else ""
    )
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>v3 NASA-TLX burden judge validation</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;background:#f4f5f7;margin:0;padding:24px;color:#1f2430}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:22px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin-top:6px}}
th,td{{border:1px solid #e3e6ea;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#fafbfc}}
.summary{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 16px;margin-bottom:8px}}
</style></head><body>
<h1>v3 NASA-TLX Cognitive-burden Judge — Known-Groups Validation</h1>
<div class="summary">
  <p><b>Overall:</b> {badge(metrics['all_pass'])}</p>
  <p><b>1. Ordinal validity</b> (designed ladder ↔ overall_workload), pooled: Spearman ρ =
     <b>{o['pooled_spearman']}</b> (≥ {o['threshold']}) → {badge(o['pass'])}</p>
  <table><tr><th>persona</th><th>Spearman ρ</th><th>n</th></tr>{by_persona_rows}</table>
  <p><b>2. Discriminant validity</b> (burden ⊥ medical correctness), pooled across personas:
     mean |Δ| within validity-matched pairs = <b>{d['mean_abs_delta']}</b> (≤ {d['threshold']})
     → {badge(d['pass'])}</p>
  <ul style="margin:4px 0 0 0;font-size:13px">{pairs}</ul>
  <p><b>3. Convergent validity</b> (DDF friction ↔ overall_workload), pooled across personas:
     Spearman ρ = <b>{metrics['convergent']['spearman']}</b></p>
  <p><b>4. Persona-sensitivity validity</b> (NEW for v3): SENSITIVE personas mean =
     <b>{p['sensitive_mean']}</b>, ROBUST personas mean = <b>{p['robust_mean']}</b>, Δ =
     <b>{p['delta']}</b> (≥ {p['threshold_min_delta']}, i.e. not inverted) → {badge(p['pass'])}</p>
</div>
<h2>Per-(item, persona) judge ratings</h2>
<table><tr><th>item</th><th>persona</th><th>expected_rank</th><th>overall_workload_mean</th>
<th>samples</th><th>friction_tag</th><th>validity</th><th>AI turn</th></tr>{rows}</table>
<h2>overall_workload by DDF friction tag (convergent, pooled across personas)</h2>
<table><tr><th>friction_tag</th><th>friction_order</th><th>mean_overall_workload</th><th>n</th></tr>{conv}</table>
{oracle_section}
</body></html>"""


def plot_oracle_recovery(oracle_results: list[dict], out_path: Path) -> None:
    """One line per target ladder level (1-5), x-axis = the 5 NASA-TLX dimensions, y-axis =
    judge-observed score -- shows how well each rung's uniform target is recovered PER
    DIMENSION (a perfect judge would draw 5 flat horizontal lines at y=1,2,3,4,5; deviations
    reveal which dimensions the judge systematically over/under-reads, e.g. central-tendency
    compression toward the middle of the scale). Dashed lines mark each rung's target level
    for direct visual comparison against its own solid observed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = [r for r in oracle_results if not r["generation_failed"] and r["observed"] is not None]
    if not rungs:
        print("  (no successfully-generated oracle rungs to plot -- skipping graph)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(_TLX_DIMS)))
    cmap = plt.get_cmap("viridis")
    for rung in sorted(rungs, key=lambda r: r["level"]):
        level = rung["level"]
        color = cmap((level - 1) / 4)
        observed = [rung["observed"][d] for d in _TLX_DIMS]
        ax.plot(x, observed, marker="o", color=color, label=f"target={level}")
        ax.axhline(level, ls="--", lw=0.7, alpha=0.4, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(_TLX_DIMS, rotation=20, ha="right")
    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel("judge-observed score (1-5)")
    ax.set_xlabel("NASA-TLX dimension")
    ax.set_title("Oracle ladder recovery: observed score per dimension, by target level\n"
                  "(dashed line = that rung's uniform target; solid = what the judge actually scored)")
    ax.legend(title="target level", fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved oracle recovery plot to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def _run(config: dict) -> None:
    jc = config["judge"]
    api_key = jc.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
    client = AsyncOpenAI(api_key=api_key, base_url=jc.get("base_url", "https://openrouter.ai/api/v1"))
    sem = asyncio.Semaphore(int(config.get("concurrency", 8)))
    samples = int(config.get("samples_per_item", 3))

    tmpl = _load_v3_prompts()
    personas = _load_personas()

    results = await asyncio.gather(*(
        judge_item(
            client, jc["model"], int(jc.get("max_tokens", 256)), float(jc.get("temperature", 0.0)),
            samples, sem, tmpl, persona, personas[persona], item,
        )
        for persona in _PERSONAS
        for item in _ITEMS
    ))
    metrics = compute_metrics(results)

    # Oracle generation+recovery (see module docstring) -- single fixed persona, reuses the
    # same client/semaphore. gen_cfg defaults are deliberately separate from judge defaults
    # (higher temperature -- natural varied phrasing is the goal, not determinism).
    gen_cfg = config.get("generator", {})
    oracle_persona = config.get("oracle_persona", "burned_out_resident")
    oracle_tmpl = _load_oracle_prompts()
    oracle_results = await asyncio.gather(*(
        run_oracle_profile(
            client, gen_cfg, jc, samples, sem, oracle_tmpl, tmpl, oracle_persona, personas[oracle_persona], profile,
        )
        for profile in _LADDER_PROFILES
    ))
    oracle_metrics = compute_oracle_metrics(oracle_results)

    out_dir = _ROOT / config.get("output_dir", "outputs/verify_load_judge_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({
            "scenario": _SCENARIO, "doctor": _DOCTOR, "metrics": metrics, "results": results,
            "oracle": {"persona": oracle_persona, "metrics": oracle_metrics, "results": oracle_results},
        }, indent=2, ensure_ascii=False))
    (out_dir / "report.html").write_text(render_html(results, metrics, oracle_results, oracle_metrics))
    plot_oracle_recovery(oracle_results, out_dir / "oracle_recovery.png")

    print("\n=== v3 NASA-TLX cognitive-burden judge validation ===")
    for r in sorted(results, key=lambda r: (r["expected_rank"], r["persona"])):
        print(f"  {r['id']:<18} {r['persona']:<20} expected={r['expected_rank']}  "
              f"overall_workload={r['overall_workload_mean']}  ({r['friction_tag']}, {r['validity']})")
    o, d, p = metrics["ordinal"], metrics["discriminant"], metrics["persona_sensitivity"]
    print(f"\n  [1] ordinal (pooled)      Spearman ρ = {o['pooled_spearman']} (≥{o['threshold']})  -> {'PASS' if o['pass'] else 'FAIL'}")
    for persona, v in o["by_persona"].items():
        print(f"        {persona:<20} ρ = {v['spearman']} (n={v['n']})")
    print(f"  [2] discriminant          mean|Δ|   = {d['mean_abs_delta']} (≤{d['threshold']})  -> {'PASS' if d['pass'] else 'FAIL'}")
    print(f"  [3] convergent            Spearman ρ = {metrics['convergent']['spearman']}")
    print(f"  [4] persona-sensitivity   Δ = {p['delta']} (≥{p['threshold_min_delta']})  -> {'PASS' if p['pass'] else 'FAIL'}")
    print(f"  ALL_PASS = {metrics['all_pass']}\n")

    print(f"=== Oracle generation+recovery (persona={oracle_persona}) ===")
    for r in sorted(oracle_results, key=lambda r: r["level"]):
        if r["generation_failed"]:
            print(f"  {r['id']:<10} level={r['level']}  GENERATION_FAILED")
        else:
            print(f"  {r['id']:<10} level={r['level']}  overall_workload={r['overall_workload_mean']}  "
                  f"observed={r['observed']}")
    oo, om = oracle_metrics["ordinal"], oracle_metrics["mae"]
    print(f"\n  [oracle-1] ordinal  Spearman ρ = {oo['spearman']} (≥{oo['threshold']}, n={oo['n']})  -> {'PASS' if oo['pass'] else 'FAIL'}")
    print(f"  [oracle-2] MAE      pooled = {om['pooled']} (≤{om['threshold']})  -> {'PASS' if om['pass'] else 'FAIL'}")
    print(f"  n_generation_failed = {oracle_metrics['n_generation_failed']}")
    print(f"  ORACLE_ALL_PASS = {oracle_metrics['all_pass']}\n")
    tracker.print_summary()

    tracker.accumulate_to_ledger(
        _ROOT / config.get("token_ledger", "token_usage_ledger.json"),
        run_meta={
            "script": "verify_load_judge_v3", "n_items": len(_ITEMS), "n_personas": len(_PERSONAS),
            "n_oracle_profiles": len(_LADDER_PROFILES), "oracle_persona": oracle_persona, "model": jc["model"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/verify_load_judge_v3.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((_ROOT / args.config).read_text()) if not Path(args.config).is_absolute() \
        else yaml.safe_load(Path(args.config).read_text())
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
