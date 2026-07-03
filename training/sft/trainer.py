"""SFT trainer: LoRA-fine-tune the policy to imitate the reward-oracle's golden actions.

Loss = NLL of the golden action given the policy prompt = -policy.action_logprob(prompt_ids, action_ids).
Reuses the trainable LocalQwenPolicy (LoRA + action_logprob + save).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any


class SFTTrainer:
    def __init__(self, policy, config: dict[str, Any]):
        import torch

        self.policy = policy
        self.cfg = config
        s = config.get("sft", {})
        self.epochs = int(s.get("epochs", 3))
        self.bs = int(s.get("batch_size", 4))
        self.lr = float(s.get("learning_rate", 1e-4))
        self.save_every = int(s.get("save_every", 1))
        # Early-stop guardrail: hold out a fraction of pairs and report their NLL each epoch.
        # Rising holdout NLL = overfitting (and, for us, entropy collapse that would make GRPO
        # groups identical). 0.0 (default) = no holdout, behavior unchanged.
        self.holdout_frac = float(s.get("holdout_frac", 0.0))
        self.save_dir = Path(config.get("experiment", {}).get("output_dir", "outputs")) / "sft"
        # Per-step training-curve log (append-only jsonl, one row per optimizer step) so learning
        # is visible at step granularity, not just the 2 epoch-end points. plot_sft.py reads this.
        self.metrics_path = self.save_dir / "metrics.jsonl"
        self.log_every = int(s.get("log_every", 1))  # log train loss every N steps
        self.tok = policy._tokenizer
        params = [p for p in policy._model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.torch = torch

    def _encode(self, messages, action):
        prompt_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
        )
        action_ids = self.tok.encode(action, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            action_ids = action_ids + [self.tok.eos_token_id]
        return prompt_ids, action_ids

    def _holdout_nll(self, holdout: list[dict]) -> float:
        """Mean per-example NLL on held-out pairs (no grad) — the early-stop signal."""
        tot, n = 0.0, 0
        with self.torch.no_grad():
            for ex in holdout:
                pid, aid = self._encode(ex["messages"], ex["action"])
                if not aid:
                    continue
                tot += float(-self.policy.action_logprob(pid, aid))
                n += 1
        return tot / max(1, n)

    @staticmethod
    def _episode_key(ex: dict) -> tuple:
        """Groups turn-level examples back into their episode (case_id, persona) so the holdout
        split happens at EPISODE granularity, not per-turn.
        A per-turn random split would leak: turn 3 of an episode in train and turn 5 of the SAME
        episode in holdout share nearly all their content (same case, same dialogue prefix),
        so holdout NLL would reflect memorization, not generalization to unseen cases."""
        return (ex.get("case_id"), ex.get("persona"))

    def _log_metric(self, row: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def train(self, data: list[dict]) -> None:
        holdout: list[dict] = []
        if self.holdout_frac > 0.0:
            episodes: dict[tuple, list[dict]] = {}
            for ex in data:
                episodes.setdefault(self._episode_key(ex), []).append(ex)
            keys = list(episodes.keys())
            random.shuffle(keys)
            k = max(1, int(len(keys) * self.holdout_frac))
            holdout_keys, train_keys = set(keys[:k]), set(keys[k:])
            holdout = [ex for key in holdout_keys for ex in episodes[key]]
            data = [ex for key in train_keys for ex in episodes[key]]
            print(f"[sft] holdout={len(holdout_keys)} episodes ({len(holdout)} turns), "
                  f"train={len(train_keys)} episodes ({len(data)} turns) — split by episode, not turn")

        # Fresh metrics log per run (don't append onto a previous run's curve).
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("")
        gstep = 0
        steps_per_epoch = (len(data) + self.bs - 1) // self.bs
        total_steps = steps_per_epoch * self.epochs
        start_time = time.monotonic()
        print(
            f"[sft] training start: {len(data)} train turns, batch_size={self.bs}, "
            f"epochs={self.epochs}, total_updates~{total_steps}"
        )
        for ep in range(self.epochs):
            random.shuffle(data)
            tot, nb = 0.0, 0
            for i in range(0, len(data), self.bs):
                epoch_step = (i // self.bs) + 1
                batch = data[i:i + self.bs]
                self.optim.zero_grad()
                loss = None
                n = 0
                for ex in batch:
                    pid, aid = self._encode(ex["messages"], ex["action"])
                    if not aid:
                        continue
                    nll = -self.policy.action_logprob(pid, aid)
                    loss = nll if loss is None else loss + nll
                    n += 1
                if loss is None:
                    continue
                (loss / n).backward()
                self.optim.step()
                step_loss = float(loss.detach()) / n
                tot += step_loss
                nb += 1
                gstep += 1
                if gstep % self.log_every == 0:
                    self._log_metric({"step": gstep, "epoch": ep, "train_nll": round(step_loss, 4)})
                    elapsed_min = (time.monotonic() - start_time) / 60.0
                    print(
                        f"\r[sft] epoch {ep + 1}/{self.epochs} "
                        f"step {epoch_step}/{steps_per_epoch} "
                        f"global {gstep}/{total_steps} "
                        f"train_nll={step_loss:.4f} "
                        f"mean_nll={tot / max(1, nb):.4f} "
                        f"elapsed={elapsed_min:.1f}m",
                        end="",
                        flush=True,
                    )
            # Epoch-end: holdout NLL (episode-split; the early-stop signal) on the same curve.
            print()
            if holdout:
                print(f"[sft] epoch {ep + 1}/{self.epochs}: computing holdout NLL...")
            hnll = self._holdout_nll(holdout) if holdout else None
            self._log_metric({
                "step": gstep, "epoch": ep, "epoch_end": True,
                "train_nll": round(tot / max(1, nb), 4),
                "holdout_nll": round(hnll, 4) if hnll is not None else None,
            })
            msg = f"[sft epoch {ep}] mean_nll={tot / max(1, nb):.4f}"
            if hnll is not None:
                msg += f"  holdout_nll={hnll:.4f}"  # rising ⇒ overfit, stop here
            print(msg)
            if (ep + 1) % self.save_every == 0:
                print(f"[sft] saving checkpoint → {self.save_dir / f'epoch_{ep + 1}'}")
                self.policy.save(str(self.save_dir / f"epoch_{ep + 1}"))

        print(f"[sft] saving final adapter → {self.save_dir / 'final'}")
        self.policy.save(str(self.save_dir / "final"))
        print(f"SFT done. Final adapter → {self.save_dir / 'final'}")
        # Auto-render the training curve (best-effort; never fail training over a plotting issue).
        try:
            from scripts.plot_sft import plot_sft
            out = plot_sft(self.metrics_path)
            print(f"Training curve → {out}")
        except Exception as e:  # noqa: BLE001 -- plotting is optional/observational
            print(f"(training curve not rendered: {e}; run scripts/plot_sft.py manually)")
