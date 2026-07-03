"""Train the dialogue policy with GRPO (agent-external env + r_align/r_final reward, LoRA)."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

# reduce CUDA fragmentation (set before torch is imported anywhere)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dialogue import load_dotenv
from core.config import load_yaml, build_plugins
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo, EpisodeConfig
from training.grpo.trainer import GRPOTrainer


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Full GRPO config (plugins + grpo + reward + training)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    # GRPO requires a trainable policy. r_final (= is_correct) comes from the v4 clinician
    # persona's own terminal self-reported belief (core/environment.py: Env._finalize), not from
    # final_judge — that plugin is only invoked by scripts/run_poc.py (eval), never on this path.
    config.setdefault("plugins", {}).setdefault("policy", {})["trainable"] = True

    user_llm, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    env = MedicalHACEnvironment(
        user_llm=user_llm,
        medical_llm=medical_llm,
        fact_validator_llm=fact_validator_llm,
        policy=policy,
        config=config,
        final_judge=final_judge,
    )

    # Dataset = cases from experiment.data_path. Persona is controlled by plugins.user_llm.persona:
    # a single name fixes the whole run; a list is expanded here into case × persona items so
    # each GRPO group (all group_size rollouts of one training item, see trainer.py) sees exactly
    # one persona, while successive items cycle through the pool (v4 only — see
    # UserSimulatorV4.reset_episode / EpisodeConfig.persona).
    case_path = config.get("experiment", {}).get("data_path")
    raw = json.loads(Path(case_path).read_text())
    cases = raw if isinstance(raw, list) else [raw]
    user_llm_cfg = config.get("plugins", {}).get("user_llm", {})
    if user_llm_cfg.get("type") == "v4" and "persona" in user_llm_cfg:
        personas = _as_list(user_llm_cfg["persona"])
        items = [
            (CaseInfo(**c), EpisodeConfig(persona=p))
            for c in cases
            for p in personas
        ]
        print(f"GRPO training: {len(cases)} case(s) x {len(personas)} persona(s) = {len(items)} item(s)")
    else:
        items = [(CaseInfo(**c), None) for c in cases]
        print(f"GRPO training: {len(cases)} case(s)")

    # Shuffle so `items[step % len(items)]` (trainer.py) doesn't walk the case×persona grid in
    # file order — with steps < len(items) that would only ever touch the first few cases.
    # Deterministic via experiment.seed; if steps > len(items), later epochs repeat this same
    # shuffled order (not re-shuffled per epoch).
    seed = config.get("experiment", {}).get("seed", 42)
    random.Random(seed).shuffle(items)

    GRPOTrainer(env, policy, config).train(items)


if __name__ == "__main__":
    main()
