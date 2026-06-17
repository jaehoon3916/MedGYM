"""Custom GRPO trainer for the multi-turn medical dialogue policy.

Per training item (case × persona) = one group: sample G full trajectories through the agent-external
env (policy in sampling mode), score each with R(τ) (core/reward.py: r_align + r_final + length), compute
group-relative advantages, and update the LoRA policy with a per-token policy-gradient + KL-to-reference
loss. Reference = adapter-disabled forward (no separate model copy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.reward import Trajectory, trajectory_return, group_advantages
from core.logger import RolloutLogger
from core.token_tracker import tracker as _tracker
from plugins.user_llm.vllm_user import EpisodeConfig


class GRPOTrainer:
    def __init__(self, env, policy, config: dict[str, Any]):
        import torch  # noqa: F401  (ensure torch present; used via policy)

        self.env = env
        self.policy = policy
        self.cfg = config

        grpo = config.get("grpo", {})
        self.group_size = int(grpo.get("group_size", 4))
        self.kl_coef = float(grpo.get("kl_coef", 0.01))

        tr = config.get("training", {})
        self.lr = float(tr.get("learning_rate", 1e-5))
        self.steps = int(tr.get("steps", 50))
        self.save_every = int(tr.get("save_every", 10))

        exp = config.get("experiment", {})
        self.max_turns = int(exp.get("max_turns", 6))
        self.weights = config.get("reward", {})
        self.save_dir = Path(exp.get("output_dir", "outputs")) / "grpo"
        self.rollout_dir = self.save_dir / "rollouts"   # saved dialogue transcripts (viewable in app.py)
        # Token accounting: every rollout's API + policy-generate tokens land here (same format as eval).
        self.exp_name = exp.get("name", "grpo")
        self.token_ledger = exp.get("token_ledger", "token_usage_ledger.json")
        self.token_dir = self.save_dir / "tokens"
        self._last_ledger = None
        self.model_names = {
            "user_llm": env.user_llm.name(),
            "medical_llm": env.medical_llm.name(),
            "fact_validator_llm": env.fact_validator_llm.name(),
            "policy": policy.name(),
        }

        import torch
        params = [p for p in self.policy._model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self._torch = torch

    # ── rollout one trajectory (sampling) ────────────────────────────────────
    def _rollout(self, case_info, episode_config, tag: str) -> Trajectory:
        eff = episode_config or EpisodeConfig()
        logger = RolloutLogger(
            case_info=case_info, model_names=self.model_names, episode_config=eff.model_dump(),
        )
        _tracker.reset()   # episode-scoped token accounting (counts even degenerate rollouts)
        obs = self.env.reset(case_info, episode_config, max_turns=self.max_turns)
        traj = Trajectory()
        while not obs.done:
            po = self.policy.sample_action(
                obs.case_info, obs.dialogue_history, obs.current_user_utterance, obs.verification,
            )
            res = self.env.step(po)
            logger.log_step(res, dialogue_snapshot=[t.model_dump() for t in self.env._history.turns])
            traj.step_align.append(float(res.metadata.get("r_align", 0.0)))
            traj.step_fmt.append(float(res.metadata.get("r_fmt", 0.0)))
            traj.steps.append((po.metadata.get("prompt_ids", []), po.metadata.get("action_ids", [])))
            if res.done:
                fj = res.metadata.get("final_judgement")
                traj.is_correct = bool(fj and fj.get("is_correct"))
                logger.finalize(fj, res.metadata.get("closed_by"))
            obs = self.env.observation
        traj.num_turns = len(traj.step_align)
        if traj.steps:
            logger.save(self.rollout_dir / f"{tag}.jsonl")   # transcript: dialogue + actions + rewards
        # Persist this rollout's token usage (raw per-call I/O + per-model summary) and fold its
        # totals into the cumulative ledger. Done for every rollout, including degenerate ones.
        _tracker.save_calls(self.token_dir / f"{tag}_calls.jsonl")
        _tracker.save_summary(self.token_dir / f"{tag}_token_summary.json")
        self._last_ledger = _tracker.accumulate_to_ledger(
            self.token_ledger,
            {"phase": "grpo", "exp": self.exp_name, "tag": tag, "case_id": case_info.case_id},
        )
        return traj

    # ── training loop ────────────────────────────────────────────────────────
    def train(self, items: list[tuple]) -> None:
        torch = self._torch
        for step in range(self.steps):
            case_info, episode_config = items[step % len(items)]

            group = [
                self._rollout(case_info, episode_config, f"{case_info.case_id}_step{step}_g{g}")
                for g in range(self.group_size)
            ]
            group = [t for t in group if t.steps]   # drop degenerate (immediate-close) trajectories
            torch.cuda.empty_cache()                # free the rollout generate KV-cache before the update
            if len(group) < 2:
                print(f"[grpo step {step}] skipped (only {len(group)} usable trajectories)")
                continue

            returns = [trajectory_return(t, self.weights) for t in group]
            advs = group_advantages(returns)

            # reward ledger: per-rollout R(τ) + group advantage (one line per sample per step)
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            with open(self.save_dir / "train_log.jsonl", "a") as lf:
                for g, (traj, R, adv) in enumerate(zip(group, returns, advs)):
                    lf.write(json.dumps({
                        "step": step, "sample": g, "case_id": case_info.case_id,
                        "R": round(R, 4), "advantage": round(adv, 4),
                        "is_correct": traj.is_correct, "num_turns": traj.num_turns,
                        "sum_r_align": round(sum(traj.step_align), 4),
                        "step_rewards": [round(x, 3) for x in traj.step_align],
                    }) + "\n")

            total_steps = sum(len(t.steps) for t in group) or 1
            self.optim.zero_grad()
            loss_sum = 0.0
            n_tok = 0
            kl_acc = 0.0
            for traj, adv in zip(group, advs):
                for prompt_ids, action_ids in traj.steps:
                    if not action_ids:
                        continue
                    logp = self.policy.action_logprob(prompt_ids, action_ids, use_ref=False)
                    ref = self.policy.action_logprob(prompt_ids, action_ids, use_ref=True)
                    # GRPO k3 KL estimator (≥0, low variance), grad flows through logp only
                    kl = torch.exp(ref - logp) - (ref - logp) - 1.0
                    step_loss = (-adv * logp + self.kl_coef * kl) / total_steps
                    step_loss.backward()          # backprop per step → frees this step's graph (memory)
                    loss_sum += float(step_loss.detach())
                    n_tok += 1
                    kl_acc += float(kl.detach())

            if n_tok == 0:
                continue
            self.optim.step()

            acc = sum(1 for t in group if t.is_correct) / len(group)
            mean_R = sum(returns) / len(returns)
            mean_turns = sum(t.num_turns for t in group) / len(group)
            print(
                f"[grpo step {step}] meanR={mean_R:.3f} acc={acc:.2f} "
                f"meanTurns={mean_turns:.1f} loss={loss_sum:.4f} kl={kl_acc / n_tok:.4f}"
            )
            # per-step metrics for plotting (loss / reward / acc / kl vs step)
            with open(self.save_dir / "metrics.jsonl", "a") as mf:
                mf.write(json.dumps({
                    "step": step, "meanR": round(mean_R, 4), "acc": round(acc, 4),
                    "meanTurns": round(mean_turns, 2), "loss": round(loss_sum, 4),
                    "kl": round(kl_acc / n_tok, 6),
                }) + "\n")

            if (step + 1) % self.save_every == 0:
                self.policy.save(str(self.save_dir / f"step_{step + 1}"))

        self.policy.save(str(self.save_dir / "final"))
        print(f"GRPO done. Final adapter → {self.save_dir / 'final'}")
        if self._last_ledger is not None:
            gt = self._last_ledger["grand_total"]
            print(f"[cumulative tokens] episodes={self._last_ledger['total_episodes']} "
                  f"total={gt['total_tokens']:,} (prompt={gt['prompt_tokens']:,} "
                  f"completion={gt['completion_tokens']:,} reasoning={gt['reasoning_tokens']:,}) "
                  f"-> {self.token_ledger}")
