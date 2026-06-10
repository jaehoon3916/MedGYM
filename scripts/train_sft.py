"""SFT-train the policy (LoRA) to imitate the reward-oracle golden actions."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml, load_action_space
from plugins.policy.qwen_policy import LocalQwenPolicy
from training.sft.trainer import SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="SFT jsonl (default: sft.data_out from config)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_path = args.data or config.get("sft", {}).get("data_out", "outputs/sft/sft_data.jsonl")
    data = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
    print(f"Loaded {len(data)} SFT pairs from {data_path}")

    pol_cfg = {**config.get("plugins", {}).get("policy", {}), "trainable": True, "mode": "sft"}
    policy = LocalQwenPolicy(pol_cfg, load_action_space())
    policy.load()

    SFTTrainer(policy, config).train(data)


if __name__ == "__main__":
    main()
