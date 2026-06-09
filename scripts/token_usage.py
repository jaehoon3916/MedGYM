"""Print the cumulative token-usage ledger (grand total across all runs).

Usage: python scripts/token_usage.py [ledger_path]   (default: token_usage_ledger.json)
"""
import json
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("token_usage_ledger.json")
    if not path.exists():
        print(f"No ledger at {path} (nothing accumulated yet).")
        return
    led = json.loads(path.read_text())
    gt = led.get("grand_total", {})
    print(f"Ledger: {path}")
    print(f"  updated_at:           {led.get('updated_at')}")
    print(f"  episodes accumulated: {led.get('total_episodes', 0)}")
    print(f"  GRAND TOTAL: {gt.get('total_tokens', 0):,} tokens "
          f"(prompt {gt.get('prompt_tokens', 0):,} / completion {gt.get('completion_tokens', 0):,} / "
          f"reasoning {gt.get('reasoning_tokens', 0):,}) over {gt.get('calls', 0):,} calls")
    per_model = led.get("per_model", {})
    if per_model:
        print("  per model:")
        for m, s in per_model.items():
            print(f"    {m}: {s.get('total_tokens', 0):,} tokens / {s.get('calls', 0):,} calls")


if __name__ == "__main__":
    main()
