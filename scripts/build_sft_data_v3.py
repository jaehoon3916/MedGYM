"""Generate action_space_v3 (EXTEND/RECOMMEND) SFT golden data by distilling a teacher policy,
with two supported teacher tracks selected entirely by the config's plugins.policy.type:

  - "action_space_v3_llm" (name() prefix "action-space-v3-llm-policy") -- SAME-PROMPT teacher.
    Byte-identical prompt to the student (plugins/policy/policy_ours_v3.py's PolicyOursV3), just
    a bigger API model choosing the answer. Mirrors build_sft_data_v2.py's ours_v2_teacher track.

  - "oracle_a3" with reveal_quadrant_meta: true (name() suffix "-meta-oracle") -- QUADRANT-ORACLE
    teacher. Sees only AI-alone-correct / doctor-currently-correct (never the literal answer
    letter), so it can actually produce a disagreeing RECOMMEND when the doctor is wrong and the
    AI-alone read is right -- the category found to be near-empty auditing the v2 GRPO rollouts
    (see memory/project_medgym_action_space_v3_design.md): of 1337 RECOMMEND.move turns, 852
    (64%) were pure agreement language, only ~18 (1.3%) had any disagreement marker, because the
    v2-trained policy's RECOMMEND guidance was structurally "confirm the clinician's plan." A
    same-prompt-only teacher risks reproducing the same collapse; this track exists to fill the
    ai_only_correct quadrant (and the conflicting-RECOMMEND cell) with real disagreement.
    Any OTHER oracle mode (reveal_correct_option / reveal_user_state) is rejected by
    _teacher_track below: reveal_correct_option ("-true-oracle") leaks the literal answer letter
    into the guidance text (not a realistic distillation target); reveal_user_state
    ("-user-state-oracle") requires a per-episode plugins.policy.persona, but this script shares
    one policy instance across concurrent episodes of DIFFERENT personas (mirrors
    build_sft_data_v2.py's "policy is stateless/shared" design) -- wiring per-episode persona
    into a shared policy object is a real change this script does not make.

Regardless of which teacher track picked the action, the RECORDED SFT PROMPT (the `messages`
field of each output record) is ALWAYS the plain, unprivileged v3 prompt built by
_student_messages() below -- byte-identical to what PolicyOursV3/ActionSpaceV3LLMPolicy build.
An oracle-track episode's extra oracle_block text (plugins/policy/action_space_v3_oracle_policy.
py) is used only to help the teacher pick a better/more decisive action; it must never leak into
what the (unprivileged, deployment-time) student is trained to condition on.

Each collected turn is also labeled post-hoc with the correctness-relation quadrant r_t =
Rel(y_hat^A_t, y_hat^H_t, y*) (docs/action_space_v3_formulation.md) -- both_correct /
ai_only_correct / doctor_only_correct / both_wrong -- using data already produced this step, no
extra API calls:
  - y_hat^A_t : the medical LLM's own stated belief THIS turn (the just-appended medical
    DialogueTurn's metadata["belief"]), mapped to a lettered option via the same word-overlap
    heuristic scripts/analyze_action_space_v3_burden.py already uses (imported, not
    reimplemented: _map_belief_to_letter).
  - y_hat^H_t : the clinician's belief letter BEFORE this turn (obs.user_state captured prior to
    calling env.step -- the same state used to build this turn's prompt), read the same way
    plugins/policy/action_space_v3_oracle_policy.py's _doctor_ok() does.
  - y*        : case_info.correct_option.
D_t (belief-conflict, 1 - b^H_t(y_hat^A_t)) is also recorded for RECOMMEND turns, computed
exactly as scripts/analyze_action_space_v3_burden.py's H1 test does, so the auto-report below is
a directly comparable rerun of that audit on this newly distilled data -- did the misaligned-
RECOMMEND fraction actually move off ~1.3%?

Always persists a quadrant/coverage report (CSV + PNG) next to --out, in addition to the
terminal preview -- never terminal-only.

Episodes run CONCURRENTLY (ThreadPoolExecutor), one fresh user_llm + env per episode (v4 carries
per-episode mutable belief state, unsafe to share across threads); medical_llm/
fact_validator_llm/policy are stateless and shared. RESUMABLE: same (case_id, persona)
skip/resume semantics as build_sft_data_v2.py (the last-written pair is treated as possibly
interrupted and re-collected).

Cost = env sim LLM calls (v4 user ~5 calls/turn) + one teacher API call/turn (paid). Run a small
--limit pilot first and check the quadrant report -- does ai_only_correct actually show up now
(track B), unlike the v2 audit -- THEN run the full set.

Usage:
  # Track A -- same-prompt naive teacher
  python scripts/build_sft_data_v3.py --config configs/sft_v3_distill_naive.yaml --limit 15

  # Track B -- quadrant-oracle teacher (targets the ai_only_correct / conflicting-RECOMMEND gap)
  python scripts/build_sft_data_v3.py --config configs/sft_v3_distill_quadrant_oracle.yaml --limit 15

  # Merge both tracks into one SFT set, then re-report on the combined file:
  cat outputs/sft_v3/sft_data_naive.jsonl outputs/sft_v3/sft_data_quadrant_oracle.jsonl \\
      > outputs/sft_v3/sft_data_combined.jsonl
  python scripts/build_sft_data_v3.py --report-only outputs/sft_v3/sft_data_combined.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dialogue import load_dotenv
from scripts.run_poc import _as_list, _build_user_simulator
from scripts.analyze_action_space_v3_burden import _map_belief_to_letter
from core.config import load_yaml, load_action_space, build_plugins
from core.environment import MedicalHACEnvironment
from core.prompt_builder import _load
from core.schemas import CaseInfo
from core.token_tracker import tracker
from plugins.policy.policy_ours_v3 import _reachable_controls

_QUADRANTS = ["both_correct", "ai_only_correct", "doctor_only_correct", "both_wrong"]
_QUADRANT_COLORS = {
    "both_correct": "#54A24B", "ai_only_correct": "#4C78A8",
    "doctor_only_correct": "#F58518", "both_wrong": "#E45756",
}


def _teacher_track(policy_name: str) -> str:
    if policy_name.startswith("action-space-v3-llm-policy"):
        return "naive"
    if policy_name.startswith("action-space-v3-oracle-policy") and "-meta-oracle" in policy_name:
        return "quadrant_oracle"
    raise ValueError(
        "plugins.policy must be either action_space_v3_llm (same-prompt teacher) or oracle_a3 "
        "with reveal_quadrant_meta: true (quadrant-oracle teacher) for v3 SFT distillation; got "
        f"policy.name()={policy_name!r}. Other oracle modes (reveal_correct_option / "
        "reveal_user_state) are not supported by this script -- see its module docstring."
    )


def _stage_menu_text(action_space: dict) -> str:
    lines = []
    for cid in ("EXTEND", "RECOMMEND"):
        desc = " ".join(str(action_space["stages"][cid].get("description", "")).split())
        lines.append(f"- {cid}: {desc}")
    return "\n".join(lines)


def _student_messages(
    stage_menu: str, vt, current_user_utterance: str, dialogue_history, reachable: list[str],
) -> list[dict[str, str]]:
    """The UNPRIVILEGED v3 prompt PolicyOursV3/ActionSpaceV3LLMPolicy build at deployment time --
    used as the recorded SFT prompt regardless of which teacher track chose the golden action
    (see module docstring: an oracle-track episode's oracle_block must never leak in here)."""
    tmpl = _load("action_space_v3_llm_policy")
    fv_block = ""
    if vt is not None:
        fv_block = tmpl["fact_validation_block"].format(
            overall_relation=vt.overall_relation, confidence=vt.confidence, reasoning=vt.reasoning,
        )
    user = tmpl["user"].format(
        reachable_stages=", ".join(reachable),
        current_user_utterance=current_user_utterance,
        fact_validation_block=fv_block,
        dialogue=dialogue_history.to_prompt_with_actions(),
    )
    return [
        {"role": "system", "content": tmpl["system"].format(stage_menu=stage_menu)},
        {"role": "user", "content": user},
    ]


def _quadrant(ai_correct: bool | None, doctor_correct: bool | None) -> str | None:
    if ai_correct is None or doctor_correct is None:
        return None
    if ai_correct and doctor_correct:
        return "both_correct"
    if ai_correct and not doctor_correct:
        return "ai_only_correct"
    if not ai_correct and doctor_correct:
        return "doctor_only_correct"
    return "both_wrong"


def _collect_episode(
    case_info: CaseInfo, persona: str, user_llm_cfg: dict,
    medical_llm, fact_validator_llm, policy, final_judge, config, max_turns: int,
    transitions: dict, stage_menu: str, teacher_track: str,
) -> list[dict]:
    """Runs one (case, persona) episode with a FRESH user_llm + env (per-episode mutable belief
    state, unsafe to share across threads); medical_llm/fact_validator_llm/policy are
    stateless/shared. user_llm_cfg is a per-job dict with `persona` already set to THIS job's
    persona (never mutate the shared base dict -- concurrent jobs would race on it)."""
    user_llm = _build_user_simulator(user_llm_cfg)
    env = MedicalHACEnvironment(
        user_llm, medical_llm, fact_validator_llm, policy, config, final_judge=final_judge,
    )
    obs = env.reset(case_info, None, max_turns=max_turns)
    correct = str(case_info.correct_option).strip().upper()
    records: list[dict] = []
    while not obs.done:
        vt = obs.verification if policy.needs_verification else None
        pre_user_state = obs.user_state  # y_hat^H_t: doctor's belief BEFORE this turn's action
        reachable = _reachable_controls(obs.dialogue_history, transitions)
        messages = _student_messages(stage_menu, vt, obs.current_user_utterance, obs.dialogue_history, reachable)
        kw = {"verification_template": obs.verification} if policy.needs_verification else {}
        po = policy.select_action(
            obs.case_info, obs.dialogue_history, obs.current_user_utterance, **kw,
        )
        control = po.metadata.get("control", "EXTEND")
        target = json.dumps(
            {"stage": control, "action_guidance": po.action_prompt or ""},
            ensure_ascii=False,
        )
        env.step(po)
        obs = env.observation

        # y_hat^A_t: the medical turn just appended by env.step -- search backwards rather than
        # a fixed offset, since the terminal (max_turns) step appends no trailing user turn.
        medical_turn = next(t for t in reversed(obs.dialogue_history.turns) if t.speaker == "medical")
        ai_letter = _map_belief_to_letter(medical_turn.metadata.get("belief"), case_info.options)

        doctor_letter, prior_belief_dist = None, {}
        if pre_user_state is not None:
            state = pre_user_state.model_dump()
            doctor_letter = str(state.get("belief") or "").strip().upper() or None
            prior_belief_dist = state.get("belief_dist") or {}

        ai_correct = (ai_letter == correct) if ai_letter else None
        doctor_correct = (doctor_letter == correct) if doctor_letter else None

        d_t = None
        if control == "RECOMMEND" and ai_letter and prior_belief_dist:
            d_t = round(1.0 - float(prior_belief_dist.get(ai_letter, 0.0)), 4)

        records.append({
            "messages": messages, "action": target,
            "case_id": case_info.case_id,
            "persona": persona,                        # the real behavioral dial (bayes_params)
            "teacher_track": teacher_track,
            "control": control,
            "ai_letter": ai_letter, "doctor_letter": doctor_letter,
            "r_ai_correct": ai_correct, "r_doctor_correct": doctor_correct,
            "r_quadrant": _quadrant(ai_correct, doctor_correct),
            "d_t": d_t,
        })
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


def _write_report(records: list[dict], out_path: Path) -> None:
    """Rule: every experiment result is persisted as CSV + PNG, never terminal-only."""
    stem = str(out_path.with_suffix(""))
    csv_path = Path(f"{stem}_quadrant_report.csv")
    png_path = Path(f"{stem}_quadrant_report.png")

    fields = [
        "case_id", "persona", "teacher_track", "control", "ai_letter", "doctor_letter",
        "r_ai_correct", "r_doctor_correct", "r_quadrant", "d_t",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in records)

    _plot_quadrant_report(records, png_path)
    print(f"  Quadrant coverage report -> {csv_path}, {png_path}")

    quadrant_counts = {q: sum(1 for r in records if r.get("r_quadrant") == q) for q in _QUADRANTS}
    unknown_n = sum(1 for r in records if r.get("r_quadrant") is None)
    print("  r_t quadrant coverage:", quadrant_counts, f"(unknown={unknown_n})")
    recommend = [r for r in records if r.get("control") == "RECOMMEND" and r.get("d_t") is not None]
    misaligned_n = sum(1 for r in recommend if r["d_t"] >= 0.5)
    if recommend:
        print(f"  RECOMMEND misaligned (D_t>=0.5): {misaligned_n}/{len(recommend)} "
              f"({misaligned_n / len(recommend):.1%}) -- v2 audit baseline was ~1.3%.")


def _plot_quadrant_report(records: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quadrant_counts = {q: sum(1 for r in records if r.get("r_quadrant") == q) for q in _QUADRANTS}
    unknown_n = sum(1 for r in records if r.get("r_quadrant") is None)

    recommend = [r for r in records if r.get("control") == "RECOMMEND" and r.get("d_t") is not None]
    aligned_n = sum(1 for r in recommend if r["d_t"] < 0.5)
    misaligned_n = sum(1 for r in recommend if r["d_t"] >= 0.5)
    recommend_unmapped_n = sum(
        1 for r in records if r.get("control") == "RECOMMEND" and r.get("d_t") is None
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax = axes[0]
    labels = list(_QUADRANTS) + (["unknown"] if unknown_n else [])
    heights = [quadrant_counts[q] for q in _QUADRANTS] + ([unknown_n] if unknown_n else [])
    colors = [_QUADRANT_COLORS[q] for q in _QUADRANTS] + (["#B0B0B0"] if unknown_n else [])
    bars = ax.bar(labels, heights, color=colors)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h, str(h), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("turn count")
    ax.set_title("r_t quadrant coverage (all turns)")
    ax.tick_params(axis="x", labelrotation=20)

    ax = axes[1]
    labels2 = ["RECOMMEND\naligned (D<0.5)", "RECOMMEND\nmisaligned (D>=0.5)"]
    heights2 = [aligned_n, misaligned_n]
    bars2 = ax.bar(labels2, heights2, color=["#4C78A8", "#F58518"])
    for bar, h in zip(bars2, heights2):
        ax.text(bar.get_x() + bar.get_width() / 2, h, str(h), ha="center", va="bottom", fontsize=9)
    total_recommend = aligned_n + misaligned_n
    frac = f"{misaligned_n / total_recommend:.1%}" if total_recommend else "n/a"
    ax.set_ylabel("turn count")
    ax.set_title(f"RECOMMEND belief-conflict coverage\n(misaligned={frac}, unmapped n={recommend_unmapped_n})")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    load_dotenv()
    tracker.reset()  # per-call singleton; must be reset per script invocation (see run_poc.py)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="teacher config (required unless --report-only)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of cases (pilot)")
    ap.add_argument("--concurrency", type=int, default=None, help="parallel episodes (default: config.concurrency or 8)")
    ap.add_argument("--out", default=None, help="override sft.data_out")
    ap.add_argument(
        "--report-only", default=None,
        help="skip collection; (re)build the CSV/PNG quadrant report from an existing jsonl "
             "(e.g. a cat-combined multi-track file)",
    )
    args = ap.parse_args()

    if args.report_only:
        records = [json.loads(l) for l in Path(args.report_only).read_text().splitlines() if l.strip()]
        _write_report(records, Path(args.report_only))
        return

    if not args.config:
        ap.error("--config is required unless --report-only is given")

    config = load_yaml(args.config)
    # final_judge is not needed for SFT data (is_correct now comes from the persona anyway).
    config.setdefault("plugins", {}).setdefault("final_judge", {})["enabled"] = False

    user_llm_cfg = config["plugins"]["user_llm"]
    _, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    teacher_track = _teacher_track(policy.name())
    policy.load()

    action_space = load_action_space(config.get("action_space_path"))
    stage_menu = _stage_menu_text(action_space)
    transitions: dict = action_space.get("transitions", {})

    raw = json.loads(Path(config["experiment"]["data_path"]).read_text())
    cases = raw if isinstance(raw, list) else [raw]
    if args.limit:
        cases = cases[: args.limit]
    personas = _as_list(user_llm_cfg.get("persona", "veteran_attending"))

    max_turns = int(config.get("experiment", {}).get("max_turns", 8))
    default_out = f"outputs/sft_v3/sft_data_{teacher_track}.jsonl"
    out_path = Path(args.out or config.get("sft", {}).get("data_out", default_out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concurrency = args.concurrency or int(config.get("concurrency", 8))

    kept_lines, skip_keys = _load_resume_state(out_path)
    if skip_keys:
        print(f"Resuming [{teacher_track}]: {len(skip_keys)} (case, persona) pair(s) already "
              f"collected, skipping. 1 possibly-interrupted pair will be re-collected.")

    jobs = [
        (CaseInfo(**c), persona)
        for c in cases
        for persona in personas
        if (c["case_id"], persona) not in skip_keys
    ]
    print(f"[{teacher_track}] {len(cases)} case(s) x {len(personas)} persona(s) = {len(jobs)} "
          f"episode(s) to collect ({len(skip_keys)} already done).")

    all_records: list[dict] = list(kept_lines)
    with open(out_path, "w") as f:
        for rec in kept_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(
                    _collect_episode, case_info, persona,
                    {**user_llm_cfg, "persona": persona},  # per-job copy -- never mutate the shared dict
                    medical_llm, fact_validator_llm, policy, final_judge, config, max_turns,
                    transitions, stage_menu, teacher_track,
                ): (case_info.case_id, persona)
                for case_info, persona in jobs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"episodes[{teacher_track}]", unit="ep"):
                records = fut.result()
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    all_records.append(rec)
                f.flush()

    print(f"Wrote {len(all_records)} v3 SFT (prompt, golden JSON action) pairs -> {out_path}")

    print("  Token usage:")
    tracker.print_summary()
    calls_path = out_path.parent / f"token_calls_{teacher_track}.jsonl"
    tracker.save_calls(calls_path)          # full per-call audit trail (raw prompts/responses)
    print(f"  Per-call token log saved to {calls_path}")
    tracker.accumulate_to_ledger(
        Path(config.get("experiment", {}).get("token_ledger", "token_usage_ledger.json")),
        run_meta={
            "script": "build_sft_data_v3", "teacher_track": teacher_track,
            "n_jobs": len(jobs), "n_pairs": len(all_records), "config": args.config,
        },
    )

    _write_report(all_records, out_path)


if __name__ == "__main__":
    main()
