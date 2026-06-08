from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.base import EvalPlugin


def build_eval_plugins(config: dict[str, Any]) -> list[EvalPlugin]:
    """Instantiate eval plugins from the top-level config's `eval` section."""
    from eval.medcobe_eval import MedCoBeEval

    _map = {"medcobe": MedCoBeEval}

    eval_cfg = config.get("eval", {})
    plugins: list[EvalPlugin] = []
    for method in eval_cfg.get("methods", ["medcobe"]):
        cls = _map.get(method)
        if cls is None:
            print(f"[WARN] Unknown eval method '{method}', skipping")
            continue
        plugins.append(cls(eval_cfg))
    return plugins


def load_rollout(path: Path) -> dict[str, list[dict]]:
    """Load JSONL file(s), group records by case_id in turn order."""
    cases: dict[str, list[dict]] = {}
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                case_id = record.get("case_id", p.stem)
                cases.setdefault(case_id, []).append(record)
    return cases


def run_eval(
    rollout_path: Path,
    plugins: list[EvalPlugin],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_rollout(rollout_path)

    if not cases:
        print(f"[WARN] No JSONL records found at {rollout_path}")
        return {}

    aggregate: dict[str, list[dict]] = {p.name(): [] for p in plugins}

    for case_id, turns in cases.items():
        case_info = turns[0].get("case_info", {}) if turns else {}
        case_result: dict[str, Any] = {"case_id": case_id, "eval": {}}

        for plugin in plugins:
            print(f"  [{plugin.name()}] {case_id} ({len(turns)} turns)...")
            result = plugin.evaluate(turns, case_info)
            case_result["eval"][plugin.name()] = {
                "scores": result.scores,
                "per_turn": result.per_turn,
                "metadata": result.metadata,
            }
            aggregate[plugin.name()].append(result.scores)

        out_file = output_dir / f"{case_id}_eval.json"
        with open(out_file, "w") as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)

    # Aggregate summary across all cases
    summary: dict[str, Any] = {}
    for plugin in plugins:
        scores_list = [s for s in aggregate[plugin.name()] if s]
        if not scores_list:
            continue
        numeric_keys = [k for k, v in scores_list[0].items() if isinstance(v, (int, float))]
        summary[plugin.name()] = {
            k: round(sum(s.get(k, 0) for s in scores_list) / len(scores_list), 4)
            for k in numeric_keys
        }
        summary[plugin.name()]["n_cases"] = len(scores_list)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary → {summary_path}")
    for plugin_name, agg in summary.items():
        print(
            f"  [{plugin_name}]  medcobe={agg.get('medcobe_score', 'N/A')}  "
            f"recall_corr={agg.get('recall_correction', 'N/A')}  "
            f"recall_conf={agg.get('recall_confirmation', 'N/A')}  "
            f"n_cases={agg.get('n_cases', 0)}"
        )

    return summary
