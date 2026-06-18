#!/usr/bin/env python3
"""
Render poc_medcobe_hac.py results.json into a static HTML debug report.

Does NOT call any LLM — reconstructs the exact instruction text that was injected
for HAC-command/HAC-reference via core.prompt_builder.frame_directive() applied to
the saved raw action_prompt (deterministic given the same raw text + style).

Usage:
    python scripts/render_poc_debug_html.py --results outputs/poc_medcobe_hac/results.json
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.prompt_builder import frame_directive

_BASELINE_ACTION_PROMPT = "Respond to the clinician's statement based on the case evidence."

_CONDITIONS = ("baseline", "hac_command", "hac_reference")
_CONDITION_LABELS = {"baseline": "Baseline", "hac_command": "HAC-command", "hac_reference": "HAC-reference"}

_CSS = """\
body { font-family: -apple-system, Segoe UI, sans-serif; background: #f4f5f7; margin: 0; padding: 24px; color: #1f2430; }
h1 { font-size: 20px; }
h2 { font-size: 15px; margin: 4px 0 10px; }
.summary { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
table { border-collapse: collapse; font-size: 13px; margin-bottom: 14px; }
th, td { border: 1px solid #e1e4e8; padding: 4px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.controls { margin-bottom: 16px; font-size: 13px; }
.card { background: #fff; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.08); border-left: 4px solid #d0d4da; }
.card.mismatch { border-left: 4px solid #e0524d; background: #fff8f7; }
.card-header { font-size: 13px; color: #555; margin-bottom: 8px; line-height: 1.5; }
.card-header b { color: #1f2430; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 11px; margin-left: 6px; }
.badge.mismatch { background: #fde2e1; color: #a3231f; }
.badge.diverge { background: #fff3cd; color: #8a6400; }
.alone { font-size: 12.5px; background: #f0f3f7; border-radius: 6px; padding: 6px 10px; margin-bottom: 10px; }
.cols { display: flex; gap: 12px; }
.col { flex: 1; min-width: 0; background: #fafbfc; border: 1px solid #eceff2; border-radius: 6px; padding: 8px 10px; font-size: 12.5px; }
.col h3 { font-size: 12.5px; margin: 0 0 6px; }
.col pre { white-space: pre-wrap; word-break: break-word; font-family: inherit; font-size: 12px; background: #fff; border: 1px solid #eceff2; border-radius: 4px; padding: 6px; margin: 4px 0; max-height: 220px; overflow-y: auto; }
.verdict { font-size: 12px; margin-top: 4px; color: #444; }
.verdict b { color: #1f2430; }
.ok { color: #1a7f37; } .bad { color: #a3231f; }
"""

_JS = """\
function toggleMismatch(cb) {
  document.querySelectorAll('.card').forEach(function(card) {
    card.style.display = (!cb.checked || card.classList.contains('mismatch')) ? '' : 'none';
  });
}
"""


def _e(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _scores_table(scores: dict) -> str:
    rows = "".join(
        f"<tr><td>{_CONDITION_LABELS[c]}</td><td>{scores[c]['recall_correction']:.4f}</td>"
        f"<td>{scores[c]['recall_confirmation']:.4f}</td><td>{scores[c]['medcobe_score']:.4f}</td></tr>"
        for c in _CONDITIONS
    )
    return f"""
    <h2>Scores (ARGUE/ACCEPT recall)</h2>
    <table><tr><th>Condition</th><th>recall_corr</th><th>recall_conf</th><th>MedCOBE</th></tr>{rows}</table>
    """


def _complementarity_table(comp: dict) -> str:
    rows = "".join(
        f"<tr><td>{_CONDITION_LABELS[c]}</td><td>{comp[c]['team_accuracy']:.4f}</td>"
        f"<td>{comp[c]['team_accuracy_error_mode']:.4f}</td><td>{comp[c]['team_accuracy_correct_mode']:.4f}</td>"
        f"<td>{comp[c]['complementarity_gain']:+.4f}</td></tr>"
        for c in _CONDITIONS
    )
    return f"""
    <h2>Complementarity (AI-alone accuracy: {comp['ai_alone_accuracy']:.4f})</h2>
    <table><tr><th>Condition</th><th>team_acc</th><th>err_mode</th><th>corr_mode</th><th>gain vs alone</th></tr>{rows}</table>
    """


def _judge_block(j: dict) -> str:
    sel_ok = "ok" if j.get("selected_option") else "bad"
    return f"""
    <div class="verdict">
      <b>doctor_claim_mode:</b> {_e(j['doctor_claim_mode'])} &nbsp;
      <b>ai_action:</b> {_e(j['ai_action'])} &nbsp;
      <b>validity:</b> {_e(j['reasoning_validity'])}<br>
      <b>selected_option:</b> <span class="{sel_ok}">{_e(j.get('selected_option') or '—')}</span>
      <b>reason:</b> {_e(j.get('brief_reason'))}
    </div>
    """


def _condition_col(label: str, instruction: str, answer: str, judge: dict, correct_option: str, alone_selected: str | None) -> str:
    team_correct = judge.get("selected_option") == correct_option
    diverge = alone_selected is not None and judge.get("selected_option") is not None and team_correct != (alone_selected == correct_option)
    badge = '<span class="badge diverge">vs alone diverges</span>' if diverge else ""
    return f"""
    <div class="col">
      <h3>{label}{badge}</h3>
      <div><b>Instruction sent:</b></div>
      <pre>{_e(instruction)}</pre>
      <div><b>AI response:</b></div>
      <pre>{_e(answer)}</pre>
      {_judge_block(judge)}
    </div>
    """


def _case_card(record: dict) -> str:
    cm = record["case_meta"]
    correct_option = cm["correct_option"]
    alone = record["alone"]
    alone_selected = alone.get("selected_option")
    alone_ok = "ok" if alone_selected == correct_option else "bad"

    bj, rj = record["baseline_judge"], record["hac_reference_judge"]
    mismatch = (bj["ai_action"], bj["reasoning_validity"]) != (rj["ai_action"], rj["reasoning_validity"])
    mismatch_badge = '<span class="badge mismatch">baseline vs reference judge differ</span>' if mismatch else ""

    instructions = {
        "baseline": _BASELINE_ACTION_PROMPT,
        "hac_command": frame_directive(record["action_prompt"], "command"),
        "hac_reference": frame_directive(record["action_prompt"], "reference"),
    }
    cols = "".join(
        _condition_col(
            _CONDITION_LABELS[cond], instructions[cond], record[f"{cond}_ai"],
            record[f"{cond}_judge"], correct_option, alone_selected,
        )
        for cond in _CONDITIONS
    )

    return f"""
    <div class="card {'mismatch' if mismatch else ''}">
      <div class="card-header">
        <b>Case {_e(record['case_id'])}</b> &nbsp; mode=<b>{_e(record['mode'])}</b> &nbsp;
        action=<b>{_e(record['action_id'])}</b> &nbsp;
        verification=<b>{_e(record['verification_relation'])}</b>
        {mismatch_badge}<br>
        <b>Doctor's target belief:</b> {_e(record['target_belief'])} &nbsp;
        <b>Ground truth:</b> {_e(correct_option)}. {_e(cm['answer'])}<br>
        <b>Doctor utterance:</b> {_e(record['doctor_text'])}<br>
        <b>Raw action_prompt (RulePolicy):</b> {_e(record['action_prompt'])}
      </div>
      <div class="alone">
        <b>AI-alone:</b> selected <span class="{alone_ok}">{_e(alone_selected or '—')}</span>
        &nbsp; <b>reasoning:</b> {_e(alone.get('reasoning'))}
      </div>
      <div class="cols">{cols}</div>
    </div>
    """


def render(data: dict) -> str:
    results = data["results"]
    n_mismatch = sum(
        1 for r in results
        if (r["baseline_judge"]["ai_action"], r["baseline_judge"]["reasoning_validity"])
        != (r["hac_reference_judge"]["ai_action"], r["hac_reference_judge"]["reasoning_validity"])
    )
    cards = "".join(_case_card(r) for r in results)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>poc_medcobe_hac debug report</title>
<style>{_CSS}</style>
<script>{_JS}</script>
</head>
<body>
<h1>MedCOBE POC Debug Report</h1>
<div class="summary">
  {_scores_table(data['scores'])}
  {_complementarity_table(data['complementarity'])}
</div>
<div class="controls">
  <label><input type="checkbox" onchange="toggleMismatch(this)"> Show only baseline vs HAC-reference mismatches ({n_mismatch}/{len(results)})</label>
</div>
{cards}
</body>
</html>
"""


def main(results_path: str, output_path: str | None) -> None:
    data = json.loads(Path(results_path).read_text())
    out = Path(output_path) if output_path else Path(results_path).parent / "debug.html"
    out.write_text(render(data))
    print(f"Saved debug report to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/poc_medcobe_hac/results.json", help="Path to results.json")
    parser.add_argument("--output", default=None, help="Output HTML path (default: <results_dir>/debug.html)")
    args = parser.parse_args()
    main(args.results, args.output)
