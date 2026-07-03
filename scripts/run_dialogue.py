"""Run a single dialogue episode and save the rollout to JSONL."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml, build_plugins
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo
from core.token_tracker import tracker


def load_dotenv() -> None:
    """Load the nearest .env (searching upward from this script) into os.environ.

    Existing environment variables take precedence, so an explicitly exported
    key is never overwritten. Avoids the need to `source .env` before running.
    """
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--case", default=None, help="Path to case JSON file (overrides config)")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: auto)")
    parser.add_argument("--max_turns", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(
        args.config)
    case_path = args.case or config.get("experiment", {}).get("data_path")
    if not case_path:
        raise ValueError("Specify --case or set experiment.data_path in config")
    raw = json.loads(Path(case_path).read_text())
    case_data = raw[0] if isinstance(raw, list) else raw
    case_info = CaseInfo(**case_data)

    user_llm, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    env = MedicalHACEnvironment(
        user_llm=user_llm,
        medical_llm=medical_llm,
        fact_validator_llm=fact_validator_llm,
        policy=policy,
        config=config,
        final_judge=final_judge,
    )

    output_dir = Path(config.get("experiment", {}).get("output_dir", "outputs"))
    exp_name = config.get("experiment", {}).get("name", "exp")
    # Experiment dir, guarding against a doubled name when output_dir already ends with exp_name.
    exp_dir = output_dir if output_dir.name == exp_name else output_dir / exp_name
    max_turns = args.max_turns or config.get("experiment", {}).get("max_turns", 2)

    ledger_path = config.get("experiment", {}).get("token_ledger", "token_usage_ledger.json")

    tracker.reset()
    output_path = args.output or str(exp_dir / f"{case_info.case_id}.jsonl")
    results = env.run_episode(
        case_info, max_turns=max_turns, output_path=output_path, episode_config=None
    )

    # Per-rollout token logs in a tokens/ subdir (tracker was reset above → episode-scoped)
    rollout_p = Path(output_path)
    token_dir = rollout_p.parent / "tokens"
    tracker.save_calls(str(token_dir / f"{rollout_p.stem}_calls.jsonl"))
    tracker.save_summary(str(token_dir / f"{rollout_p.stem}_token_summary.json"))
    # Add this episode's usage to the persistent cumulative ledger (survives overwrites)
    ledger = tracker.accumulate_to_ledger(
        ledger_path,
        {"exp": exp_name, "case_id": case_info.case_id},
    )

    fj = results[-1].metadata.get("final_judgement") if results else None
    closed_by = results[-1].metadata.get("closed_by") if results else None
    mark = "?" if fj is None else ("✓" if fj["is_correct"] else "✗")
    concluded = fj["concluded_option"] if fj else "?"
    print(f"{len(results)} turns | {mark} "
          f"concluded={concluded} correct={case_info.correct_option} "
          f"closed_by={closed_by} -> {output_path}")

    if ledger is not None:
        gt = ledger["grand_total"]
        print(f"[cumulative tokens] episodes={ledger['total_episodes']} "
              f"total={gt['total_tokens']:,} "
              f"(prompt={gt['prompt_tokens']:,} completion={gt['completion_tokens']:,}) "
              f"-> {ledger_path}")

    if config.get("eval", {}).get("enabled", False):
        from eval.runner import build_eval_plugins, run_eval
        eval_output = Path(config["eval"].get("output_dir", "eval_results")) / exp_name
        print(f"\nRunning eval → {eval_output}")
        run_eval(output_dir / exp_name, build_eval_plugins(config), eval_output)


if __name__ == "__main__":
    main()
