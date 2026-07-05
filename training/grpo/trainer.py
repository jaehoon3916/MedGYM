"""Custom GRPO trainer for the multi-turn medical dialogue policy.

Per training item (case × persona) = one group: sample G full trajectories through the agent-external
env (policy in sampling mode), score each with R(τ) (core/reward.py: r_final + r_fmt + cost/leak terms), compute
group-relative advantages, and update the LoRA policy with a per-token policy-gradient + KL-to-reference
loss. Reference = frozen SFT adapter when available; otherwise adapter-disabled base forward.
"""
from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.reward import (
    Trajectory,
    group_advantages,
    leak_score_v3,
    trajectory_components,
    trajectory_return,
)
from core.logger import RolloutLogger
from core.token_tracker import tracker as _tracker, TokenTracker
from core.schemas import EpisodeConfig


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
        self.metrics_path = self.save_dir / "metrics.jsonl"
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

        # ── parallel rollouts ────────────────────────────────────────────────
        # A group's group_size rollouts are independent and I/O-bound (each is ~50 sequential
        # OpenRouter calls). Run them across worker threads that overlap those API waits. Each
        # worker checks out its OWN env from a pool (env + user_llm are stateful -> can't share);
        # all envs share the single GPU policy, whose generate is serialized by _policy_lock
        # (a small fraction of wall time). rollout_workers defaults to group_size; set it to 1
        # for the old strictly-sequential behavior (e.g. to A/B the two are equivalent).
        self.n_workers = max(1, min(self.group_size, int(grpo.get("rollout_workers", self.group_size))))
        self._policy_lock = threading.Lock()
        self._env_pool: queue.Queue = queue.Queue()
        self._env_pool.put(self.env)
        if self.n_workers > 1:
            from core.config import build_plugins
            from core.environment import MedicalHACEnvironment
            for _ in range(self.n_workers - 1):
                u, m, fv, _p, fj = build_plugins(self.cfg, reuse_policy=self.policy)
                self._env_pool.put(MedicalHACEnvironment(
                    user_llm=u, medical_llm=m, fact_validator_llm=fv,
                    policy=self.policy, config=self.cfg, final_judge=fj,
                ))

    # ── rollout one trajectory (sampling) ────────────────────────────────────
    @staticmethod
    def _phi_from_user_state(user_state, correct_option: str) -> float | None:
        if user_state is None:
            return None
        dist = user_state.model_dump().get("belief_dist") or {}
        try:
            return float(dist.get(str(correct_option).strip().upper()))
        except (TypeError, ValueError):
            return None

    def _rollout(self, env, case_info, episode_config, tag: str):
        """Run one sampled rollout on `env` (checked out from the pool). Returns
        (traj, sub_tracker, logger, tag); the caller persists tokens/transcript sequentially
        after all workers join, so the shared ledger's read-modify-write never races."""
        eff = episode_config or EpisodeConfig()
        logger = RolloutLogger(
            case_info=case_info, model_names=self.model_names, episode_config=eff.model_dump(),
        )
        sub = TokenTracker()   # per-rollout token accounting, isolated from sibling workers
        traj = Trajectory()
        with _tracker.redirect(sub):   # every plugin record()/begin_turn()/flush_turn() -> sub
            obs = env.reset(case_info, episode_config, max_turns=self.max_turns)
            traj.phi_start = self._phi_from_user_state(obs.user_state, case_info.correct_option)
            while not obs.done:
                with self._policy_lock:   # serialize the one shared GPU policy's generate
                    po = self.policy.sample_action(
                        obs.case_info, obs.dialogue_history, obs.current_user_utterance, obs.verification,
                    )
                res = env.step(po)
                logger.log_step(res, dialogue_snapshot=[t.model_dump() for t in env._history.turns])
                traj.step_fmt.append(float(res.metadata.get("r_fmt", 0.0)))
                # v3-aware O2 guard: flag answer-naming guidance only when it cannot be
                # attributed to the medical LLM's own belief (core/reward.py).
                traj.step_leak.append(leak_score_v3(
                    po.action_prompt,
                    case_info.correct_option,
                    case_info.answer,
                    res.metadata.get("medical_belief"),
                    case_info.options,
                ))
                traj.steps.append((po.metadata.get("prompt_ids", []), po.metadata.get("action_ids", [])))
                # Burden from the user's response to this AI turn (NASA-TLX overall_workload, 1-5).
                # Zero when no user turn follows (max_turns terminal: _advance_user_turn skipped).
                _next_us = env.observation.user_state if env.observation else None
                _burden = float(_next_us.model_dump().get("cognitive_burden", 0.0)) if _next_us else 0.0
                if res.done and res.metadata.get("closed_by") == "max_turns":
                    _burden = 0.0
                else:
                    _phi = self._phi_from_user_state(_next_us, case_info.correct_option)
                    if _phi is not None:
                        traj.step_phi.append(_phi)
                traj.step_burden.append(_burden)
                if res.done:
                    fj = res.metadata.get("final_judgement")
                    traj.closed_by = res.metadata.get("closed_by")
                    traj.is_correct = bool(fj and fj.get("is_correct"))
                    logger.finalize(fj, traj.closed_by)
                obs = env.observation
        traj.num_turns = len(traj.step_fmt)
        return traj, sub, logger, tag

    def _persist_rollout(self, traj, sub, logger, tag: str, case_id: str) -> None:
        """Sequential post-join I/O: transcript + this rollout's token usage folded into the
        cumulative ledger. Kept out of the worker threads so the ledger's read-modify-write and
        the append-only history are never interleaved across rollouts."""
        if traj.steps:
            logger.save(self.rollout_dir / f"{tag}.jsonl")   # transcript: dialogue + actions + rewards
        # Persist per-rollout token usage (raw per-call I/O + per-model summary), for every
        # rollout including degenerate ones, then fold totals into the cumulative ledger.
        sub.save_calls(self.token_dir / f"{tag}_calls.jsonl")
        sub.save_summary(self.token_dir / f"{tag}_token_summary.json")
        self._last_ledger = sub.accumulate_to_ledger(
            self.token_ledger,
            {"phase": "grpo", "exp": self.exp_name, "tag": tag, "case_id": case_id},
        )

    # ── training loop ────────────────────────────────────────────────────────
    def train(self, items: list[tuple]) -> None:
        torch = self._torch
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("")
        for step in range(self.steps):
            case_info, episode_config = items[step % len(items)]

            def _run(g, _ci=case_info, _ec=episode_config, _step=step):
                env = self._env_pool.get()   # unique env per concurrent worker (blocks if none free)
                try:
                    return self._rollout(env, _ci, _ec, f"{_ci.case_id}_step{_step}_g{g}")
                finally:
                    self._env_pool.put(env)

            if self.n_workers > 1:
                with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                    results = list(ex.map(_run, range(self.group_size)))
            else:
                results = [_run(g) for g in range(self.group_size)]
            # Sequential post-join I/O (transcript + ledger; ledger RMW must not race).
            for traj, sub, logger, tag in results:
                self._persist_rollout(traj, sub, logger, tag, case_info.case_id)
            group = [traj for (traj, _sub, _lg, _tag) in results if traj.steps]  # drop degenerate
            torch.cuda.empty_cache()                # free the rollout generate KV-cache before the update
            if len(group) < 2:
                print(f"[grpo step {step}] skipped (only {len(group)} usable trajectories)")
                continue

            returns = [trajectory_return(t, self.weights) for t in group]
            advs = group_advantages(returns)
            comps = [trajectory_components(t, self.weights) for t in group]

            # reward ledger: per-rollout R(τ) + group advantage (one line per sample per step)
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            with open(self.save_dir / "train_log.jsonl", "a") as lf:
                for g, (traj, R, adv, comp) in enumerate(zip(group, returns, advs, comps)):
                    phi_end = traj.step_phi[-1] if traj.step_phi else None
                    lf.write(json.dumps({
                        "step": step, "sample": g, "case_id": case_info.case_id,
                        "persona": episode_config.persona if episode_config else None,
                        "R": round(R, 4), "advantage": round(adv, 4),
                        "is_correct": traj.is_correct, "num_turns": traj.num_turns,
                        "closed_by": traj.closed_by,
                        "sum_burden": round(sum(traj.step_burden), 3),
                        "sum_leak": round(sum(traj.step_leak), 1),
                        "sum_fmt": round(sum(traj.step_fmt), 4),
                        "phi_start": round(traj.phi_start, 4) if traj.phi_start is not None else None,
                        "phi_end": round(phi_end, 4) if phi_end is not None else None,
                        "phi_gain": (
                            round(phi_end - traj.phi_start, 4)
                            if phi_end is not None and traj.phi_start is not None else None
                        ),
                        "step_phi": [round(x, 4) for x in traj.step_phi],
                        **{k: round(v, 4) for k, v in comp.items()},
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
                    # Compute the frozen reference first. If we compute train logp first, its
                    # autograd graph/activations stay resident while the ref forward runs; on
                    # 24GB QLoRA this can OOM inside bitsandbytes' temporary 4-bit dequant buffer.
                    ref = self.policy.action_logprob(prompt_ids, action_ids, use_ref=True)
                    logp = self.policy.action_logprob(prompt_ids, action_ids, use_ref=False)
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
            phi_gains = [
                t.step_phi[-1] - t.phi_start
                for t in group
                if t.phi_start is not None and t.step_phi
            ]
            mean_phi_gain = sum(phi_gains) / len(phi_gains) if phi_gains else 0.0
            n_drop = sum(1 for t in group if t.closed_by == "burden_dropout")
            n_leak = sum(sum(t.step_leak) for t in group)
            print(
                f"[grpo step {step}] meanR={mean_R:.3f} acc={acc:.2f} "
                f"meanTurns={mean_turns:.1f} meanPhiGain={mean_phi_gain:.3f} "
                f"drop={n_drop} leak={n_leak:.0f} loss={loss_sum:.4f} kl={kl_acc / n_tok:.4f}"
            )
            # per-step metrics for plotting (loss / reward / acc / kl vs step)
            with open(self.metrics_path, "a") as mf:
                mf.write(json.dumps({
                    "step": step, "meanR": round(mean_R, 4), "acc": round(acc, 4),
                    "meanTurns": round(mean_turns, 2), "loss": round(loss_sum, 4),
                    "kl": round(kl_acc / n_tok, 6),
                    "meanPhiGain": round(mean_phi_gain, 6),
                    "dropout_count": n_drop,
                    "leak_count": round(n_leak, 3),
                }) + "\n")

            if (step + 1) % self.save_every == 0:
                self.policy.save(str(self.save_dir / f"step_{step + 1}"))

        self.policy.save(str(self.save_dir / "final"))
        print(f"GRPO done. Final adapter → {self.save_dir / 'final'}")
        try:
            from scripts.plot_grpo import plot_grpo
            out = plot_grpo(self.metrics_path)
            if out:
                print(f"Training curve → {out}")
        except Exception as e:  # noqa: BLE001 -- plotting is observational
            print(f"(training curve not rendered: {e}; run scripts/plot_grpo.py manually)")
        if self._last_ledger is not None:
            gt = self._last_ledger["grand_total"]
            print(f"[cumulative tokens] episodes={self._last_ledger['total_episodes']} "
                  f"total={gt['total_tokens']:,} (prompt={gt['prompt_tokens']:,} "
                  f"completion={gt['completion_tokens']:,} reasoning={gt['reasoning_tokens']:,}) "
                  f"-> {self.token_ledger}")
