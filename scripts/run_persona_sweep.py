#!/usr/bin/env python3
"""
Persona/info-condition sweep: runs scripts/run_scaling_poc.py's scaling-curve POC once per
(persona, info_condition) combination, to isolate their effect on the turn-checkpoint
accuracy curve. info_condition (full|dense|sparse, see persona/information_sparsity.yaml's
3-condition note and run_scaling_poc.py:_resolve_info_condition) is the single named axis
covering both medical_llm.show_case_info and user_llm.information_sparsity -- "full" means
the AI gets the case directly (no real info asymmetry); "dense"/"sparse" mean it does NOT,
and only differ in how forthcoming the doctor is.

Reads both axes straight off the config: plugins.user_llm.persona and
plugins.user_llm.info_condition. Each can be a single string ("just run this one value") or
a YAML list ("sweep all of these") -- a string is treated as a 1-element list, so e.g.
persona as a list of 4 with info_condition as the single string "full" sweeps 4 conditions
(current default in configs/scaling_poc_persona_sweep.yaml), while making both lists would
sweep the full cross product.

For each (persona, info_condition) pair, deep-copies the base config, overrides
plugins.user_llm.persona, plugins.user_llm.info_condition, and experiment.name, then calls
run_scaling_poc.run() (NOT main()) so everything stays in-process -- no subprocess, no
on-disk per-condition YAML beyond what run_scaling_poc.run() already writes to each
condition's own outputs/<name>/config_used.yaml.

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    python scripts/run_persona_sweep.py --config configs/scaling_poc_persona_sweep.yaml
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import load_yaml
from scripts.run_scaling_poc import run as run_scaling_poc

_ALONE_BUCKETS = (True, False)


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _print_combined_report(all_scores: dict[str, dict], conditions: list[str]) -> None:
    sample = next(iter(all_scores.values()))
    all_cps = sample["all_checkpoints"]
    print()
    print("=" * 96)
    print("  Persona/info-condition sweep: turn-checkpoint accuracy by condition")
    print("=" * 96)
    for bucket in _ALONE_BUCKETS:
        label = "alone_correct" if bucket else "alone_incorrect"
        print(f"\n  {label}")
        header = "  condition                                  " + "  ".join(f"{c:>6}" for c in all_cps)
        print(header)
        for cond in conditions:
            row = all_scores[cond]["curve"][bucket]
            acc = "  ".join(f"{row[c]['accuracy']:>6.3f}" for c in all_cps)
            print(f"  {cond:<43}  {acc}")
    print("\n  doctor_alone_accuracy by condition:")
    for cond in conditions:
        print(f"    {cond:<43}  {all_scores[cond]['doctor_alone_accuracy']:.4f}")
    print()


def main(config_path: str) -> None:
    base_config = load_yaml(config_path)
    personas = _as_list(base_config["plugins"]["user_llm"].get("persona", "burned_out_resident"))
    info_conditions = _as_list(base_config["plugins"]["user_llm"].get("info_condition", "full"))
    # The config's OWN experiment.name is the experiment's root folder -- every condition this
    # sweep runs (and the combined summary) nests UNDER outputs/<base_name>/, as a subfolder
    # per persona/info_condition, instead of each condition getting its own flat top-level
    # outputs/<...> dir. A "/" in experiment.name just becomes a path separator (run_scaling_poc.py's
    # output_dir = outputs/exp["name"], mkdir(parents=True)), so this needs no other code changes.
    # This also means two different config files (e.g. naive vs oracle policy) never collide on
    # the same path as long as their experiment.name differs -- which is the user's responsibility
    # to set, not something this driver should guess at (e.g. by hardcoding policy.type into the name).
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
            all_scores[cond] = run_scaling_poc(cfg)

    _print_combined_report(all_scores, conditions)

    summary_dir = _ROOT / "outputs" / base_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    print(f"  Combined summary saved to {summary_dir / 'summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scaling_poc_persona_sweep.yaml")
    args = parser.parse_args()
    main(args.config)
