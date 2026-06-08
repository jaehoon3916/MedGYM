"""Evaluate dialogue rollouts using pluggable eval methods."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml
from eval.runner import build_eval_plugins, run_eval


def main():
    parser = argparse.ArgumentParser(description="Run eval on rollout JSONL files")
    parser.add_argument("--rollout", required=True, help="JSONL file or directory of JSONL files")
    parser.add_argument("--config", required=True, help="Main config YAML (must have an `eval` section)")
    parser.add_argument("--output", default=None, help="Output directory (default: eval.output_dir from config)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    plugins = build_eval_plugins(config)

    if not plugins:
        print("No eval plugins loaded — check the `eval.methods` section in your config.")
        return

    output_dir = Path(args.output or config.get("eval", {}).get("output_dir", "eval_results"))
    run_eval(Path(args.rollout), plugins, output_dir)


if __name__ == "__main__":
    main()
