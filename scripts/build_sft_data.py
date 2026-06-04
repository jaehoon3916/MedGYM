"""Build SFT training data from raw cases using a rule-based labeler."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input cases JSONL")
    parser.add_argument("--output", required=True, help="Output SFT JSONL")
    parser.add_argument("--policy", default="rule", choices=["rule"])
    args = parser.parse_args()

    # TODO: Phase 3 — iterate cases, run rule extractor + rule policy, write SFT records
    raise NotImplementedError("SFT data builder not yet implemented (Phase 3)")


if __name__ == "__main__":
    main()
