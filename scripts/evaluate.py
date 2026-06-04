"""Evaluate policy on test cases."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--policy_checkpoint", default=None)
    args = parser.parse_args()

    # TODO: Phase 5 — compute acceptance_rate, challenge_rate, sycophancy_rate, etc.
    raise NotImplementedError("Evaluation not yet implemented (Phase 5)")


if __name__ == "__main__":
    main()
