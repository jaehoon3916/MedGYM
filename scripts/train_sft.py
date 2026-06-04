"""Train SFT policy model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    args = parser.parse_args()

    # TODO: Phase 3 — load SFT dataset, train BERT/RoBERTa policy, save checkpoint
    raise NotImplementedError("SFT training not yet implemented (Phase 3)")


if __name__ == "__main__":
    main()
