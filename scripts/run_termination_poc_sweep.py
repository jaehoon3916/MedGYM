#!/usr/bin/env python3
"""
Persona/info-condition sweep for scripts/run_termination_poc.py -- mirrors
scripts/run_persona_sweep.py's mechanics exactly (same persona/info_condition axis handling),
just pointed at run_termination_poc.run() instead of run_scaling_poc.run(), since the two
scripts' `scores` shapes differ (no checkpoint curve here -- see run_termination_poc.aggregate).

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    python scripts/run_termination_poc_sweep.py --config configs/termination_poc_persona_sweep.yaml
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import load_yaml
from scripts.run_termination_poc import run as run_termination_poc


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _print_combined_report(all_scores: dict[str, dict], conditions: list[str]) -> None:
    print()
    print("=" * 96)
    print("  Termination POC persona/info-condition sweep")
    print("=" * 96)
    print(f"\n  {'condition':<40}{'accuracy':>10}{'n':>6}")
    for cond in conditions:
        s = all_scores[cond]
        print(f"  {cond:<40}{s['accuracy']:>10.4f}{s['n_cases']:>6}")
    print()


def main(config_path: str) -> None:
    base_config = load_yaml(config_path)
    personas = _as_list(base_config["plugins"]["user_llm"].get("persona", "burned_out_resident"))
    info_conditions = _as_list(base_config["plugins"]["user_llm"].get("info_condition", "full"))
    base_name = base_config["experiment"]["name"]

    all_scores: dict[str, dict] = {}
    conditions: list[str] = []
    for persona in personas:
        for info_condition in info_conditions:
            cond = f"{persona}/{info_condition}"
            conditions.append(cond)
            cfg = copy.deepcopy(base_config)
            cfg["plugins"]["user_llm"]["persona"] = persona
            cfg["plugins"]["user_llm"]["info_condition"] = info_condition
            cfg["experiment"]["name"] = f"{base_name}/{persona}_{info_condition}"
            print(f"\n{'#' * 96}\n# Condition: {cond}\n{'#' * 96}")
            all_scores[cond] = run_termination_poc(cfg)

    _print_combined_report(all_scores, conditions)

    summary_dir = _ROOT / "outputs" / base_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    print(f"  Combined summary saved to {summary_dir / 'summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/termination_poc_persona_sweep.yaml")
    args = parser.parse_args()
    main(args.config)
