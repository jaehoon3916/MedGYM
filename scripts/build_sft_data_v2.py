"""Generate v2 SFT golden data by DISTILLING a remote teacher (ours_v2_teacher, e.g. deepseek-v4).

For each (case × persona) the env is rolled forward with the API teacher; per
turn we record the student's observation prompt (the ours_v2 messages) + the golden action the
student should emit — the JSON {"stage": <EXPAND|CHALLENGE|CONVERGE>, "action_guidance": <text>}.

The behavioral diversity axis is plugins.user_llm.persona (bayes_params.yaml's
c0/lambda/w/rho/kappa dials). Accepts a single persona string OR a list.

Differs from build_sft_data.py (which force-sets the rule-based `oracle` teacher — broken on v2:
BASE is keyed on McBurney stages so every v2 control scores 0 → degenerate always-EXPAND). Here we
use whatever policy the config specifies; point plugins.policy.type at "ours_v2_teacher".

Episodes run CONCURRENTLY (ThreadPoolExecutor, mirroring scripts/run_poc.py) — each worker builds
its own fresh user_llm + env (v3/v4 carry per-episode mutable belief state, so a shared instance
would race across threads); medical_llm/fact_validator_llm/policy are stateless and shared.
RESUMABLE: reruns skip (case, persona) pairs already fully written in --out
(the last pair in an interrupted prior run is treated as possibly-incomplete and re-collected).

Cost = env sim LLM calls (v4 user ≈5 calls/turn) + one teacher API call/turn (paid).
Run a small --limit pilot first, eyeball JSON validity + strategic sensibility, THEN the full set.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dialogue import load_dotenv
from scripts.run_poc import _as_list, _build_user_simulator
from core.config import load_yaml, build_plugins
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo
from core.token_tracker import tracker


def _collect_episode(
    case_info: CaseInfo, persona: str, user_llm_cfg: dict,
    medical_llm, fact_validator_llm, policy, final_judge, config, max_turns: int,
) -> list[dict]:
    """Runs one (case, persona) episode with a FRESH user_llm + env (per-episode
    mutable belief state, unsafe to share across threads); medical_llm/fact_validator_llm/policy
    are stateless/shared. user_llm_cfg is a per-job dict with `persona` already set to THIS job's
    persona (never mutate the shared base dict -- concurrent jobs would race on it)."""
    user_llm = _build_user_simulator(user_llm_cfg)
    env = MedicalHACEnvironment(
        user_llm, medical_llm, fact_validator_llm, policy, config, final_judge=final_judge,
    )
    obs = env.reset(case_info, None, max_turns=max_turns)
    records: list[dict] = []
    while not obs.done:
        vt = obs.verification if policy.needs_verification else None
        # The SFT prompt is the student's own ours_v2 prompt == the teacher's prompt.
        messages = policy.build_messages(vt, obs.current_user_utterance, obs.dialogue_history)
        kw = {"verification_template": obs.verification} if policy.needs_verification else {}
        po = policy.select_action(
            obs.case_info, obs.dialogue_history, obs.current_user_utterance, **kw,
        )
        # Golden target = exactly what the student PolicyOursV2 should generate/parse.
        target = json.dumps(
            {"stage": po.metadata.get("control", "EXPAND"), "action_guidance": po.action_prompt or ""},
            ensure_ascii=False,
        )
        records.append({
            "messages": messages, "action": target,
            "case_id": case_info.case_id,
            "persona": persona,                        # the real behavioral dial (bayes_params)
        })
        env.step(po)
        obs = env.observation
    return records


_Key = tuple[str, str]  # (case_id, persona)


def _load_resume_state(out_path: Path) -> tuple[list[dict], set[_Key]]:
    """Returns (lines to keep, (case_id, persona) pairs to SKIP re-collecting).

    The pair that appears LAST in the file (by write order) is treated as possibly interrupted
    mid-episode (a prior run could have been killed between turns) and is dropped from both the
    kept lines and the skip set, so it gets fully re-collected from scratch.
    """
    if not out_path.exists():
        return [], set()
    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    if not lines:
        return [], set()

    def key(d: dict) -> _Key:
        return (d["case_id"], d.get("persona", "?"))

    order: list[_Key] = []
    seen: set[_Key] = set()
    for d in lines:
        k = key(d)
        if k not in seen:
            seen.add(k)
            order.append(k)
    incomplete = order[-1]
    kept = [d for d in lines if key(d) != incomplete]
    skip = seen - {incomplete}
    return kept, skip

def main():
    load_dotenv()
    tracker.reset()  # per-call singleton; must be reset per script invocation (see run_poc.py)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap number of cases (pilot)")
    ap.add_argument("--concurrency", type=int, default=None, help="parallel episodes (default: config.concurrency or 8)")
    ap.add_argument("--out", default=None, help="override sft.data_out")
    args = ap.parse_args()

    config = load_yaml(args.config)
    # final_judge is not needed for SFT data (is_correct now comes from the persona anyway).
    config.setdefault("plugins", {}).setdefault("final_judge", {})["enabled"] = False

    user_llm_cfg = config["plugins"]["user_llm"]
    # build_plugins also constructs a throwaway shared user_llm we don't use (each episode
    # builds its OWN via _build_user_simulator below, since v3/v4 carry per-episode belief state).
    _, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    assert policy.name().startswith("ours-v2-api-teacher"), (
        f"plugins.policy.type must be 'ours_v2_teacher' for v2 distillation; got {policy.name()!r}"
    )
    policy.load()

    raw = json.loads(Path(config["experiment"]["data_path"]).read_text())
    cases = raw if isinstance(raw, list) else [raw]
    if args.limit:
        cases = cases[: args.limit]
    personas = _as_list(user_llm_cfg.get("persona", "veteran_attending"))

    max_turns = int(config.get("experiment", {}).get("max_turns", 6))
    out_path = Path(args.out or config.get("sft", {}).get("data_out", "outputs/sft_v2/sft_data.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concurrency = args.concurrency or int(config.get("concurrency", 8))

    kept_lines, skip_keys = _load_resume_state(out_path)
    if skip_keys:
        print(f"Resuming: {len(skip_keys)} (case, persona) pair(s) already "
              f"collected, skipping. 1 possibly-interrupted pair will be re-collected.")

    jobs = [
        (CaseInfo(**c), persona)
        for c in cases
        for persona in personas
        if (c["case_id"], persona) not in skip_keys
    ]
    print(f"  {len(cases)} case(s) × {len(personas)} persona(s) = {len(jobs)} episode(s) to collect "
          f"({len(skip_keys)} already done).")

    n = sum(1 for _ in kept_lines)
    with open(out_path, "w") as f:
        for rec in kept_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(
                    _collect_episode, case_info, persona,
                    {**user_llm_cfg, "persona": persona},  # per-job copy -- never mutate the shared dict
                    medical_llm, fact_validator_llm, policy, final_judge, config, max_turns,
                ): (case_info.case_id, persona)
                for case_info, persona in jobs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="episodes", unit="ep"):
                records = fut.result()
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += len(records)

    print(f"Wrote {n} v2 SFT (prompt, golden JSON action) pairs → {out_path}")

    print("  Token usage:")
    tracker.print_summary()
    calls_path = out_path.parent / "token_calls.jsonl"
    tracker.save_calls(calls_path)          # full per-call audit trail (raw prompts/responses)
    print(f"  Per-call token log saved to {calls_path}")
    tracker.accumulate_to_ledger(
        Path(config.get("experiment", {}).get("token_ledger", "token_usage_ledger.json")),
        run_meta={"script": "build_sft_data_v2", "n_jobs": len(jobs), "n_pairs": n, "config": args.config},
    )


if __name__ == "__main__":
    main()
