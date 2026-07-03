from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TokenTracker:
    """Thread-safe tracker for every LLM call: token usage + raw I/O."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[dict[str, Any]] = []
        # Per-turn buffer must be thread-local: scripts/run_scaling_poc.py runs many episodes
        # concurrently (ThreadPoolExecutor), each calling begin_turn()/flush_turn() around its
        # own step() -- a single shared list here would let one episode's begin_turn() wipe
        # another's in-flight calls, or have its flush_turn() pick up a DIFFERENT episode's
        # calls (confirmed: a rollout's "turn_prompts" log contained another case's prompt).
        self._local = threading.local()
        # Per-thread redirect target (see redirect()). When set, every record/begin_turn/
        # flush_turn on THIS tracker delegates to the sub-tracker instead. Lets the GRPO trainer
        # run group_size rollouts in parallel threads, each accumulating into its OWN sub-tracker
        # (per-rollout _calls, save_summary, accumulate_to_ledger stay isolated) even though the
        # plugins all record into the one global singleton. Untouched threads (eval) are unaffected.
        self._ctx = threading.local()

    def _active(self) -> "TokenTracker | None":
        return getattr(self._ctx, "sub", None)

    @contextmanager
    def redirect(self, sub: "TokenTracker"):
        """Within this context, the calling thread's record/begin_turn/flush_turn are captured by
        `sub` (a private, single-thread TokenTracker) instead of this shared instance."""
        prev = getattr(self._ctx, "sub", None)
        self._ctx.sub = sub
        try:
            yield sub
        finally:
            self._ctx.sub = prev

    def _turn_buffer(self) -> list[dict[str, Any]]:
        if not hasattr(self._local, "buffer"):
            self._local.buffer = []
        return self._local.buffer

    def record(
        self,
        model: str,
        messages: list[dict[str, str]],
        response_text: str,
        usage: Any,
    ) -> None:
        sub = self._active()
        if sub is not None:
            sub.record(model, messages, response_text, usage)
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "messages": messages,
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }
        with self._lock:
            self._calls.append(entry)
        self._turn_buffer().append(entry)  # thread-local, no lock needed

    def begin_turn(self) -> None:
        """Reset the CALLING THREAD's per-turn call buffer. Call at the start of each step()."""
        sub = self._active()
        if sub is not None:
            sub.begin_turn()
            return
        self._local.buffer = []

    def flush_turn(self) -> list[dict[str, Any]]:
        """Return and clear the calling thread's per-turn buffer."""
        sub = self._active()
        if sub is not None:
            return sub.flush_turn()
        buf = self._turn_buffer()
        self._local.buffer = []
        return buf

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
        self._local.buffer = []

    def accumulate_to_ledger(self, path: str | Path, run_meta: dict | None = None) -> dict:
        """Add this tracker's current totals into a persistent cumulative ledger.

        Read-modify-write with an atomic replace, so the running grand total survives across
        runs and is never lost by an overwrite of per-rollout files. Also appends one audit
        line to a sibling <name>_history.jsonl (append-only, recoverable). Returns the ledger.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
        current = self.summary()

        ledger: dict[str, Any] = {}
        if path.exists():
            try:
                ledger = json.loads(path.read_text())
            except Exception:
                ledger = {}

        per_model: dict[str, dict] = ledger.get("per_model", {})
        for model, s in current.items():
            tgt = per_model.setdefault(model, {k: 0 for k in keys})
            for k in keys:
                tgt[k] = tgt.get(k, 0) + int(s.get(k, 0))

        ledger["per_model"] = per_model
        ledger["grand_total"] = {k: sum(m.get(k, 0) for m in per_model.values()) for k in keys}
        ledger["total_episodes"] = ledger.get("total_episodes", 0) + 1
        ledger["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Atomic write: never leave a half-written/clobbered ledger.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        tmp.replace(path)

        # Append-only audit trail (safe even if the ledger json is ever lost).
        run_total = {k: sum(s.get(k, 0) for s in current.values()) for k in keys}
        hist = path.with_name(path.stem + "_history.jsonl")
        with open(hist, "a") as f:
            f.write(json.dumps(
                {"timestamp": ledger["updated_at"], **(run_meta or {}), "run_total": run_total},
                ensure_ascii=False,
            ) + "\n")

        return ledger

    def summary(self) -> dict[str, dict[str, int]]:
        with self._lock:
            calls = list(self._calls)
        totals: dict[str, dict[str, int]] = {}
        for c in calls:
            m = c["model"]
            if m not in totals:
                totals[m] = {"calls": 0, "prompt_tokens": 0,
                              "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
            totals[m]["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
                totals[m][k] += c[k]
        return totals

    def save_calls(self, path: str | Path) -> None:
        """Save per-call raw I/O + token log as JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            calls = list(self._calls)
        with open(path, "w") as f:
            for c in calls:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def save_summary(self, path: str | Path) -> None:
        """Save per-model token summary as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        for model, s in self.summary().items():
            print(
                f"[tokens] {model}  calls={s['calls']}"
                f"  prompt={s['prompt_tokens']}"
                f"  completion={s['completion_tokens']}"
                f"  reasoning={s['reasoning_tokens']}"
                f"  total={s['total_tokens']}"
            )


# Global singleton
tracker = TokenTracker()
