#!/usr/bin/env python3
"""Renders a focused, card-per-rung HTML view of scripts/verify_load_judge_v3.py's oracle
generation+recovery results -- the existing report.html already has this in a compact table,
but the generated ai_utterance gets truncated in a tiny cell. This shows, for each target
ladder level, the FULL generated AI utterance plus the judge's per-dimension observed score
and rationale side by side, so a human can actually read what the judge was scoring.

Usage:
    python scripts/render_oracle_recovery_detail.py \\
        --results outputs/verify_load_judge_v3/results.json \\
        --out outputs/verify_load_judge_v3/oracle_recovery_detail.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _e(v) -> str:
    return html.escape(str(v if v is not None else ""))


def render(results_path: Path) -> str:
    data = json.loads(results_path.read_text())
    oracle = data["oracle"]
    rungs = sorted(oracle["results"], key=lambda r: r["level"])
    metrics = oracle["metrics"]
    persona = oracle["persona"]

    badge = lambda ok: (
        f"<span class='badge {'pass' if ok else 'fail'}'>{'PASS' if ok else 'FAIL'}</span>"
    )

    cards = []
    for r in rungs:
        if r["generation_failed"]:
            cards.append(f"""
<div class="card">
  <div class="card-head"><span class="level">target = {r['level']}</span>
  <span class="badge fail">GENERATION FAILED</span></div>
</div>""")
            continue
        dims = list(r["observed"].keys())
        dim_rows = "".join(
            f"<tr><td>{_e(d)}</td><td class='num'>{r['target'][d]}</td>"
            f"<td class='num'><b>{_e(r['observed'][d])}</b></td>"
            f"<td>{_e(r['design_rationale'].get(d, ''))}</td></tr>"
            for d in dims
        )
        cards.append(f"""
<div class="card">
  <div class="card-head">
    <span class="level">target = {r['level']}</span>
    <span class="overall">judge overall_workload = <b>{_e(r['overall_workload_mean'])}</b></span>
  </div>
  <div class="utterance">{_e(r['ai_utterance'])}</div>
  <table class="dims">
    <tr><th>dimension</th><th>target</th><th>observed</th><th>design rationale (why this wording hits the target)</th></tr>
    {dim_rows}
  </table>
</div>""")

    o, m = metrics["ordinal"], metrics["mae"]
    per_dim_mae = "".join(f"<li>{_e(d)}: MAE={v}</li>" for d, v in m["per_dim"].items())

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Oracle recovery -- generated utterances</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;background:#f4f5f7;margin:0;padding:24px;color:#1f2430}}
h1{{font-size:20px}}
.summary{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 16px;margin-bottom:20px}}
.badge{{font-weight:600;padding:1px 8px;border-radius:4px;font-size:12px}}
.badge.pass{{color:#1a7f37;background:#eaf7ee}}
.badge.fail{{color:#a3231f;background:#fdecea}}
.card{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px 20px;margin-bottom:18px}}
.card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.level{{font-size:16px;font-weight:700}}
.overall{{font-size:13px;color:#444}}
.utterance{{background:#f0f4ff;border-left:4px solid #5b7cff;border-radius:6px;padding:12px 14px;
            font-size:14px;line-height:1.6;margin-bottom:12px;white-space:pre-wrap}}
table.dims{{border-collapse:collapse;width:100%;font-size:13px}}
table.dims th,table.dims td{{border:1px solid #e3e6ea;padding:6px 8px;text-align:left;vertical-align:top}}
table.dims th{{background:#fafbfc}}
.num{{text-align:center;width:70px}}
</style></head><body>
<h1>Oracle recovery -- generated AI utterances per target rung (persona = {_e(persona)})</h1>
<div class="summary">
  <p><b>Overall:</b> {badge(metrics['all_pass'])}</p>
  <p>Ordinal validity (target level &harr; judge overall_workload): Spearman &rho; =
     <b>{o['spearman']}</b> (&ge; {o['threshold']}, n={o['n']}) &rarr; {badge(o['pass'])}</p>
  <p>MAE (target vs observed, per-dim): pooled = <b>{m['pooled']}</b> (&le; {m['threshold']}) &rarr; {badge(m['pass'])}</p>
  <ul style="margin:4px 0 0 0">{per_dim_mae}</ul>
</div>
{"".join(cards)}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/verify_load_judge_v3/results.json")
    parser.add_argument("--out", default="outputs/verify_load_judge_v3/oracle_recovery_detail.html")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.write_text(render(Path(args.results)))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
