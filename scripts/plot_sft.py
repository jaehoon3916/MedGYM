"""Plot the SFT training curve (train NLL per step + holdout NLL per epoch) from metrics.jsonl.

metrics.jsonl (written by training/sft/trainer.py) has two kinds of rows:
  per-step   : {"step", "epoch", "train_nll"}                       -- the dense train-loss curve
  epoch-end  : {"step", "epoch", "epoch_end": true, "train_nll",    -- + the holdout (early-stop)
                "holdout_nll"}                                          signal, sparse

Reading it: train_nll should fall; holdout_nll is the one that matters -- when it stops falling /
turns up, the epoch BEFORE that is the checkpoint to hand GRPO (over-fit past there = entropy
collapse -> identical GRPO group samples -> zero advantage).

Usage:
    python scripts/plot_sft.py --metrics outputs/sft_v2/sft/metrics.jsonl
Saves curves.png next to the metrics file.
"""
import argparse
import json
from pathlib import Path


def plot_sft(metrics_path, out=None) -> Path:
    path = Path(metrics_path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"no metrics in {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_rows = [r for r in rows if not r.get("epoch_end")]
    epoch_rows = [r for r in rows if r.get("epoch_end")]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Dense per-step train NLL
    if step_rows:
        ax.plot([r["step"] for r in step_rows], [r["train_nll"] for r in step_rows],
                lw=1.0, color="#1a6fa8", alpha=0.8, label="train NLL (per step)")

    # Sparse epoch-end holdout NLL — the early-stop signal
    ho = [(r["step"], r["holdout_nll"], r["epoch"]) for r in epoch_rows if r.get("holdout_nll") is not None]
    if ho:
        ax.plot([s for s, _, _ in ho], [v for _, v, _ in ho],
                marker="o", ms=7, lw=1.6, color="#c0392b", label="holdout NLL (per epoch)")
        # mark the best (lowest) holdout epoch = the checkpoint to keep
        best_i = min(range(len(ho)), key=lambda i: ho[i][1])
        bs, bv, be = ho[best_i]
        ax.axvline(bs, ls="--", color="#c0392b", alpha=0.4)
        ax.annotate(f"best holdout\n(epoch {be}, {bv:.3f})",
                    xy=(bs, bv), xytext=(6, 10), textcoords="offset points",
                    fontsize=9, color="#c0392b")

    # epoch boundaries
    for r in epoch_rows:
        ax.axvline(r["step"], ls=":", color="#999", alpha=0.3)

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("NLL of golden action")
    ax.set_title(f"SFT training — {path.parent}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out) if out else path.with_name("curves.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/sft_v2/sft/metrics.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = Path(args.metrics)
    if not path.exists():
        print(f"No metrics at {path} — run training first (trainer appends per step).")
        return
    print(f"saved {plot_sft(path, args.out)}")


if __name__ == "__main__":
    main()
