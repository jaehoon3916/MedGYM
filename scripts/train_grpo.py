"""Train GRPO policy model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    args = parser.parse_args()

    # TODO: Phase 4 — rollout sampler, reward model, group-relative advantage, policy update
    raise NotImplementedError("GRPO training not yet implemented (Phase 4)")


if __name__ == "__main__":
    main()
