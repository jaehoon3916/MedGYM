#!/usr/bin/env python3
"""
Render run_poc_multiturn.py results.json into a static HTML debug report.

Does NOT call any LLM. Re-derives the per-turn injected instruction text via
core.prompt_builder.frame_directive() applied to each step's raw action_prompt
(deterministic given the condition's known frame_style).

Usage:
    python scripts/render_poc_debug_html.py --results outputs/poc_multiturn/results.json
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

_CONDITIONS = ("baseline", "hac_command", "hac_reference")
_CONDITION_LABELS = {"baseline": "Baseline", "hac_command": "HAC-command", "hac_reference": "HAC-reference"}
_FRAME_STYLE = {"baseline": "command", "hac_command": "command", "hac_reference": "reference"}

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
.alone { font-size: 12.5px; background: #f0f3f7; border-radius: 6px; padding: 6px 10px; margin-bottom: 10px; }
.cols { display: flex; gap: 12px; }
.col { flex: 1; min-width: 0; background: #fafbfc; border: 1px solid #eceff2; border-radius: 6px; padding: 8px 10px; font-size: 12.5px; }
.col h3 { font-size: 12.5px; margin: 0 0 6px; }
.turn { border: 1px solid #eceff2; border-radius: 4px; padding: 6px; margin: 6px 0; background: #fff; }
.turn .lbl { font-weight: 600; font-size: 11px; color: #777; }
.turn pre { white-space: pre-wrap; word-break: break-word; font-family: inherit; font-size: 12px; margin: 2px 0 6px; }
.final { font-size: 12px; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #ddd; }
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


def load_cases_by_id(data_dir: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in sorted(data_dir.glob("jama_raw_*.json")):
        for case in json.loads(path.read_text()):
            by_id[case["case_id"]] = case
    return by_id


def _scores_table(scores: dict) -> str:
    rows = "".join(
        f"<tr><td>{_CONDITION_LABELS[c]}</td><td>{scores[c]['team_accuracy']:.4f}</td>"
        f"<td>{scores[c]['team_accuracy_error_mode']:.4f}</td><td>{scores[c]['team_accuracy_correct_mode']:.4f}</td>"
        f"<td>{scores[c]['complementarity_gain']:+.4f}</td><td>{scores[c]['avg_turns']:.2f}</td>"
        f"<td>{scores[c]['agreement_rate']*100:.1f}%</td></tr>"
        for c in _CONDITIONS
    )
    return f"""
    <h2>Scores (AI-alone accuracy: {scores['ai_alone_accuracy']:.4f})</h2>
    <table><tr><th>Condition</th><th>team_acc</th><th>err_mode</th><th>corr_mode</th>
    <th>gain vs alone</th><th>avg_turns</th><th>agree%</th></tr>{rows}</table>
    """


def _turn_block(step: dict, condition: str) -> str:
    instruction = frame_directive(step["action_prompt"], _FRAME_STYLE[condition]) if step.get("action_prompt") else ""
    r_align = step.get("r_align")
    r_align_str = f"{r_align:.3f}" if isinstance(r_align, (int, float)) else "—"
    return f"""
    <div class="turn">
      <div class="lbl">Turn {step['turn']} &middot; action={_e(step.get('action_id'))} &middot; r_align={r_align_str}</div>
      <div class="lbl">[Doctor]</div>
      <pre>{_e(step['user_utterance'])}</pre>
      <div class="lbl">Instruction injected</div>
      <pre>{_e(instruction)}</pre>
      <div class="lbl">[AI]</div>
      <pre>{_e(step['medical_response'])}</pre>
    </div>
    """


def _condition_col(condition: str, record: dict) -> str:
    turns_html = "".join(_turn_block(s, condition) for s in record["steps"])
    fj = record.get("final_judgement") or {}
    is_correct = bool(fj.get("is_correct"))
    ok_cls = "ok" if is_correct else "bad"
    return f"""
    <div class="col">
      <h3>{_CONDITION_LABELS[condition]}</h3>
      {turns_html or "<div class='lbl'>(no AI turns — closed immediately)</div>"}
      <div class="final">
        <b>concluded_option:</b> <span class="{ok_cls}">{_e(fj.get('concluded_option', '—'))}</span>
        ({'correct' if is_correct else 'incorrect'}) &nbsp;
        <b>closed_by:</b> {_e(record['closed_by'])} &nbsp;
        <b>n_turns:</b> {record['n_turns']}<br>
        <b>reason:</b> {_e(fj.get('reason'))}
      </div>
    </div>
    """


def _case_card(case_id: str, mode: str, by_condition: dict[str, dict], case: dict | None, alone: dict | None) -> str:
    correct_option = case.get("correct_option") if case else None

    def _is_correct(cond: str) -> bool:
        fj = by_condition.get(cond, {}).get("final_judgement") or {}
        return bool(fj.get("is_correct"))

    mismatch = _is_correct("baseline") != _is_correct("hac_reference")
    mismatch_badge = '<span class="badge mismatch">baseline vs reference outcome differ</span>' if mismatch else ""

    alone_selected = (alone or {}).get("selected_option")
    alone_ok = "ok" if correct_option and alone_selected == correct_option else "bad"

    cols = "".join(
        _condition_col(cond, by_condition[cond]) for cond in _CONDITIONS if cond in by_condition
    )

    header_extra = ""
    if case:
        header_extra = f"<b>Ground truth:</b> {_e(correct_option)}. {_e(case.get('answer'))}<br>"

    return f"""
    <div class="card {'mismatch' if mismatch else ''}">
      <div class="card-header">
        <b>Case {_e(case_id)}</b> &nbsp; mode=<b>{_e(mode)}</b>
        {mismatch_badge}<br>
        {header_extra}
      </div>
      <div class="alone">
        <b>AI-alone:</b> selected <span class="{alone_ok}">{_e(alone_selected or '—')}</span>
        &nbsp; <b>reasoning:</b> {_e((alone or {}).get('reasoning'))}
      </div>
      <div class="cols">{cols}</div>
    </div>
    """


def render(data: dict, cases_by_id: dict[str, dict]) -> str:
    records = data["records"]
    alone = data.get("alone", {})

    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for r in records:
        grouped.setdefault((r["case_id"], r["mode"]), {})[r["condition"]] = r

    keys = sorted(grouped.keys())
    n_mismatch = 0
    cards = []
    for case_id, mode in keys:
        by_condition = grouped[(case_id, mode)]
        fj_b = (by_condition.get("baseline", {}).get("final_judgement") or {}).get("is_correct")
        fj_r = (by_condition.get("hac_reference", {}).get("final_judgement") or {}).get("is_correct")
        if bool(fj_b) != bool(fj_r):
            n_mismatch += 1
        cards.append(_case_card(case_id, mode, by_condition, cases_by_id.get(case_id), alone.get(case_id)))

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>run_poc_multiturn debug report</title>
<style>{_CSS}</style>
<script>{_JS}</script>
</head>
<body>
<h1>Multi-turn POC Debug Report</h1>
<div class="summary">
  {_scores_table(data['scores'])}
</div>
<div class="controls">
  <label><input type="checkbox" onchange="toggleMismatch(this)"> Show only baseline vs HAC-reference outcome mismatches ({n_mismatch}/{len(keys)})</label>
</div>
{''.join(cards)}
</body>
</html>
"""


def main(results_path: str, output_path: str | None) -> None:
    data = json.loads(Path(results_path).read_text())
    data_dir = _ROOT / data["run_meta"]["data_dir"]
    cases_by_id = load_cases_by_id(data_dir)
    out = Path(output_path) if output_path else Path(results_path).parent / "debug.html"
    out.write_text(render(data, cases_by_id))
    print(f"Saved debug report to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/poc_multiturn/results.json", help="Path to results.json")
    parser.add_argument("--output", default=None, help="Output HTML path (default: <results_dir>/debug.html)")
    args = parser.parse_args()
    main(args.results, args.output)
