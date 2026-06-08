"""Run 10 episodes with varied EpisodeConfigs and pretty-print dialogues."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml, build_plugins
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo
from plugins.user_llm.vllm_user import EpisodeConfig

EPISODE_CONFIGS = [
    # (initial_fact,  confidence,   authority_push, information_sparcity, safety_push)
    ("incorrect", "certain",   "high", "dense",  "false"),
    ("incorrect", "certain",   "high", "sparse", "true"),
    ("incorrect", "uncertain", "low",  "dense",  "false"),
    ("incorrect", "uncertain", "low",  "sparse", "true"),
    ("incorrect", "neutral",   "high", "dense",  "true"),
    ("correct",   "certain",   "high", "dense",  "false"),
    ("correct",   "certain",   "low",  "sparse", "false"),
    ("correct",   "uncertain", "high", "sparse", "true"),
    ("correct",   "uncertain", "low",  "dense",  "false"),
    ("correct",   "neutral",   "low",  "sparse", "true"),
]


def print_episode(idx: int, cfg: EpisodeConfig, results: list) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Episode {idx+1:02d} | fact={cfg.initial_fact} conf={cfg.confidence} "
          f"auth={cfg.authority_push} sparcity={cfg.information_sparcity} safety={cfg.safety_push}")
    print(sep)
    for r in results:
        print(f"\n[Turn {r.turn_id}]")
        print(f"  USER   : {r.user_utterance}")
        print(f"  ACTION : {r.selected_action}")
        print(f"  DOCTOR : {r.medical_response}")
        if r.done:
            print(f"  ** DONE (Close) **")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", default=None)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/sample_test")
    args = parser.parse_args()

    config = load_yaml(args.config)
    case_path = args.case or config.get("experiment", {}).get("data_path")
    raw = json.loads(Path(case_path).read_text())
    case_data = raw[0] if isinstance(raw, list) else raw
    case_info = CaseInfo(**case_data)

    user_llm, medical_llm, fact_validator_llm, policy = build_plugins(config)
    env = MedicalHACEnvironment(
        user_llm=user_llm,
        medical_llm=medical_llm,
        fact_validator_llm=fact_validator_llm,
        policy=policy,
        config=config,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (fact, conf, auth, sparcity, safety) in enumerate(EPISODE_CONFIGS):
        cfg = EpisodeConfig(
            initial_fact=fact,
            confidence=conf,
            authority_push=auth,
            information_sparcity=sparcity,
            safety_push=safety,
        )
        tag = f"{fact}_{conf}_{auth}"
        output_path = output_dir / f"episode_{idx+1:02d}_{tag}.jsonl"

        results = env.run_episode(case_info, max_turns=args.max_turns,
                                  output_path=str(output_path), episode_config=cfg)
        print_episode(idx, cfg, results)

    print(f"Rollouts saved to: {output_dir}/")


if __name__ == "__main__":
    main()
