"""Run a single dialogue episode and save the rollout to JSONL."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml, build_plugins
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--case", default=None, help="Path to case JSON file (overrides config)")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: auto)")
    parser.add_argument("--max_turns", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    case_path = args.case or config.get("experiment", {}).get("data_path")
    if not case_path:
        raise ValueError("Specify --case or set experiment.data_path in config")
    raw = json.loads(Path(case_path).read_text())
    case_data = raw[0] if isinstance(raw, list) else raw
    case_info = CaseInfo(**case_data)

    user_llm, medical_llm, extractor_llm, policy = build_plugins(config)
    env = MedicalHACEnvironment(
        user_llm=user_llm,
        medical_llm=medical_llm,
        extractor_llm=extractor_llm,
        policy=policy,
        config=config,
    )

    output_dir = Path(config.get("experiment", {}).get("output_dir", "outputs"))
    exp_name = config.get("experiment", {}).get("name", "exp")
    output_path = args.output or str(output_dir / exp_name / f"{case_info.case_id}.jsonl")

    max_turns = args.max_turns or config.get("experiment", {}).get("max_turns", 2)
    results = env.run_episode(case_info, max_turns=max_turns, output_path=output_path)

    print(f"Episode complete: {len(results)} turns")
    print(f"Rollout saved to: {output_path}")
    for r in results:
        print(f"  Turn {r.turn_id}: action={r.selected_action}, user_state={r.user_state.summary!r}")


if __name__ == "__main__":
    main()
