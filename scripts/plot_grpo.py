"""Plot GRPO training curves (loss / reward / accuracy / KL vs step) from metrics.jsonl.

Usage: python scripts/plot_grpo.py --metrics outputs/grpo1/grpo/metrics.jsonl
Saves a PNG next to the metrics file (curves.png).
"""
import argparse
import json
from pathlib import Path


def plot_grpo(metrics_path, out=None) -> Path | None:
    path = Path(metrics_path)
    if not path.exists():
        print(f"No metrics at {path} — run training first (it appends per step).")
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        print("metrics file is empty.")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    panels = [
        ("loss", "loss (policy-gradient; oscillates)"),
        ("meanR", "mean reward  R(tau)"),
        ("acc", "accuracy (reached gold)"),
        ("kl", "KL to reference"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        ys = [r.get(key) for r in rows]
        ax.plot(steps, ys, marker="o", ms=3, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(f"GRPO training — {path.parent}", fontsize=11)
    fig.tight_layout()

    out_path = Path(out) if out else path.with_name("curves.png")
    fig.savefig(out_path, dpi=120)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/grpo1/grpo/metrics.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = plot_grpo(args.metrics, args.out)
    if out:
        rows = [line for line in Path(args.metrics).read_text().splitlines() if line.strip()]
        print(f"saved {out}  ({len(rows)} steps)")


if __name__ == "__main__":
    main()
