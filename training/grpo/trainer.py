"""Custom GRPO trainer for the multi-turn medical dialogue policy.

Per training item (case × persona) = one group: sample G full trajectories through the agent-external
env (policy in sampling mode), score each with R(τ) (core/reward.py: r_align + r_final + length), compute
group-relative advantages, and update the LoRA policy with a per-token policy-gradient + KL-to-reference
loss. Reference = adapter-disabled forward (no separate model copy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.reward import Trajectory, trajectory_return, group_advantages


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

        self.max_turns = int(config.get("experiment", {}).get("max_turns", 6))
        self.weights = config.get("reward", {})
        self.save_dir = Path(config.get("experiment", {}).get("output_dir", "outputs")) / "grpo"

        import torch
        params = [p for p in self.policy._model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self._torch = torch

    # ── rollout one trajectory (sampling) ────────────────────────────────────
    def _rollout(self, case_info, episode_config) -> Trajectory:
        obs = self.env.reset(case_info, episode_config, max_turns=self.max_turns)
        traj = Trajectory()
        while not obs.done:
            po = self.policy.sample_action(
                obs.case_info, obs.dialogue_history, obs.current_user_utterance, obs.verification,
            )
            res = self.env.step(po)
            traj.step_align.append(float(res.metadata.get("r_align", 0.0)))
            traj.step_fmt.append(float(res.metadata.get("r_fmt", 0.0)))
            traj.steps.append((po.metadata.get("prompt_ids", []), po.metadata.get("action_ids", [])))
            if res.done:
                fj = res.metadata.get("final_judgement")
                traj.is_correct = bool(fj and fj.get("is_correct"))
            obs = self.env.observation
        traj.num_turns = len(traj.step_align)
        return traj

    # ── training loop ────────────────────────────────────────────────────────
    def train(self, items: list[tuple]) -> None:
        torch = self._torch
        for step in range(self.steps):
            case_info, episode_config = items[step % len(items)]

            group = [self._rollout(case_info, episode_config) for _ in range(self.group_size)]
            group = [t for t in group if t.steps]   # drop degenerate (immediate-close) trajectories
            if len(group) < 2:
                print(f"[grpo step {step}] skipped (only {len(group)} usable trajectories)")
                continue

            returns = [trajectory_return(t, self.weights) for t in group]
            advs = group_advantages(returns)

            self.optim.zero_grad()
            loss_tensor = None
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
                    step_loss = -adv * logp + self.kl_coef * kl
                    loss_tensor = step_loss if loss_tensor is None else loss_tensor + step_loss
                    n_tok += 1
                    kl_acc += float(kl.detach())

            if loss_tensor is None or n_tok == 0:
                continue
            loss = loss_tensor / n_tok
            loss.backward()
            self.optim.step()

            acc = sum(1 for t in group if t.is_correct) / len(group)
            mean_R = sum(returns) / len(returns)
            mean_turns = sum(t.num_turns for t in group) / len(group)
            print(
                f"[grpo step {step}] meanR={mean_R:.3f} acc={acc:.2f} "
                f"meanTurns={mean_turns:.1f} loss={float(loss):.4f} kl={kl_acc / n_tok:.4f}"
            )

            if (step + 1) % self.save_every == 0:
                self.policy.save(str(self.save_dir / f"step_{step + 1}"))

        self.policy.save(str(self.save_dir / "final"))
        print(f"GRPO done. Final adapter → {self.save_dir / 'final'}")
