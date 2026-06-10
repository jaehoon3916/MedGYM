"""Train the dialogue policy with GRPO (agent-external env + r_align/r_final reward, LoRA)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dialogue import load_dotenv
from core.config import load_yaml, build_plugins, load_episode_configs
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo
from training.grpo.trainer import GRPOTrainer


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Full GRPO config (plugins + grpo + reward + training)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    # GRPO requires a trainable policy and the final judge (r_final = user reached gold).
    config.setdefault("plugins", {}).setdefault("policy", {})["trainable"] = True
    config["plugins"].setdefault("final_judge", {})["enabled"] = True

    user_llm, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    env = MedicalHACEnvironment(
        user_llm=user_llm,
        medical_llm=medical_llm,
        fact_validator_llm=fact_validator_llm,
        policy=policy,
        config=config,
        final_judge=final_judge,
    )

    # Dataset = (case × persona) items. Cases from experiment.data_path; personas from initial_user_state/.
    case_path = config.get("experiment", {}).get("data_path")
    raw = json.loads(Path(case_path).read_text())
    cases = raw if isinstance(raw, list) else [raw]
    personas = load_episode_configs(config.get("experiment", {}).get("initial_user_state"))
    items = [(CaseInfo(**c), ep) for c in cases for (_name, ep) in personas]
    print(f"GRPO training: {len(cases)} case(s) × {len(personas)} persona(s) = {len(items)} items")

    GRPOTrainer(env, policy, config).train(items)


if __name__ == "__main__":
    main()
