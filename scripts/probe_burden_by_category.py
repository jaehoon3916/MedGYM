#!/usr/bin/env python3
"""Controlled single-turn burden probe (H1/H2, docs/action_space_v3_formulation.md §5, §7).

Unlike scripts/analyze_action_space_v3_burden.py (which extracts EXTEND/RECOMMEND_aligned/
RECOMMEND_misaligned records post-hoc from full multi-turn rollouts, at the mercy of however
often the naive policy happened to land on each category), this script CONSTRUCTS an exactly
balanced dataset: for every (case, persona) combo it produces one row per category, so
n = n_cases * n_personas in EVERY one of the 3 categories -- no reliance on natural incidence.

Single-turn only, per request: no multi-turn dialogue, no belief update / evidence tagging --
only the Stage-1 NASA-TLX burden judge (prompts/user_simulator_v4.yaml's burden_judge_*
templates) is scored, exactly as analyze_action_space_v3_burden.py's H1 test does on real
rollouts, so results are directly comparable in shape.

Design note -- how alignment is balanced:
  This is a controlled burden probe, not the real deployment path. For every (case, persona)
  combo we assign an AI option K_ai directly (default: the case's correct_option, configurable
  below), then construct:
    RECOMMEND_aligned    : doctor option == K_ai, AI recommends K_ai
    RECOMMEND_misaligned : doctor option != K_ai, AI recommends K_ai
    EXTEND               : doctor option == K_ai, AI does not disclose a conclusion
  This tests the judge's response to agreement vs disagreement by construction, with no fragile
  free-text-belief-to-option mapping and no dropped "unmapped K_ai" cases.

Reuses the real plugin classes/functions wherever possible, not reimplemented:
  - plugins.medical_llm.vllm_medical.VLLMMedicalLLM.generate_medical_response for the scored AI
    turns (identical code path core/environment.py uses, with a probe-only note pinning the
    already-probed RECOMMEND option).
  - plugins.user_llm.user_simulator.v4.UserSimulatorV4._chat / _score_burden_tlx for the
    doctor-turn0 generation and burden judge scoring (same client/prompt/retry code the real
    simulator uses -- only the target-letter steering directive is new, see
    _steered_doctor_turn0 below).
  - scripts.analyze_action_space_v3_burden's category_burden_means / category_dim_means /
    plot_category_means / plot_dim_means / write_csv / _CATEGORIES / _TLX_DIMS, so this probe's
    deliverables are shaped identically to the naive-rollout H1/H2 analysis and can be compared
    side by side.

Always persists to disk (never terminal-only): records.csv, metrics.csv,
category_burden_means.csv/.png, category_dim_means.csv/.png under --out-dir.

Usage:
    python scripts/probe_burden_by_category.py \
        --config configs/poc_action_v3_naive_postmerge.yaml \
        --out-dir outputs/action_space_v3_burden_probe_singleturn
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import load_action_space, load_yaml
from core.json_utils import safe_json_load
from core.schemas import CaseInfo, DialogueHistory
from plugins.medical_llm.vllm_medical import VLLMMedicalLLM
from plugins.user_llm.user_simulator.v4 import (
    UserSimulatorV4,
    _clean_str,
    _fmt_options,
    _load_personas,
    _load_prompts,
    _load_sparsity_persona,
)
from scripts.analyze_action_space_v3_burden import (
    _CATEGORIES,
    _TLX_DIMS,
    _TLX_DIMS_NO_FRUSTRATION,
    category_burden_means,
    category_burden_means_no_frustration,
    category_dim_means,
    plot_category_means,
    plot_dim_means,
    plot_four_panel_summary,
    write_csv,
)
from scripts.run_dialogue import load_dotenv  # loads OPENROUTER_API_KEY from .env, as run_poc.py does

_VALID_PERSONAS = ("veteran_attending", "exhausted_attending", "eager_resident", "burned_out_resident")


def _steered_doctor_turn0(sim: UserSimulatorV4, case_info: CaseInfo, target_letter: str) -> str:
    """Same utterance_turn0_* prompt _opening_turn uses, plus one added directive pinning the
    persona's independently-decided option to target_letter -- the SIMULATED DOCTOR's
    professed position (never the AI's) is what's under experimental control here."""
    tmpl = _load_prompts()
    ctx = dict(
        scenario=case_info.scenario,
        image_caption=case_info.caption or "N/A",
        question=case_info.question,
        options_text=_fmt_options(case_info.options),
        persona_instruction=_load_personas()[sim._persona],
        sparsity_instruction=_load_sparsity_persona()[sim._information_sparsity],
    )
    steering = (
        f'\n\n[Note for this exercise] Your independently-formed clinical decision is option '
        f'{target_letter} ("{case_info.options[target_letter]}"). Write your opening message '
        f'and reasoning consistent with that decision -- do not choose a different option.'
    )
    messages = [
        {"role": "system", "content": tmpl["utterance_turn0_system"].format(**ctx)},
        {"role": "user", "content": tmpl["utterance_turn0_user"].format(**ctx) + steering},
    ]
    for _ in range(3):
        raw = sim._chat(messages, response_format={"type": "json_object"})
        data = safe_json_load(raw)
        belief = str(data.get("belief", "")).strip().upper()
        text = _clean_str(data.get("response")) or raw.strip()
        if text and belief == target_letter:
            return text
    return (
        f'I believe the most appropriate answer is option {target_letter}: '
        f'{case_info.options[target_letter]}. My reasoning is that this option best fits the '
        f'clinical decision point as I currently understand it.'
    )


def _assigned_ai_letter(case_info: CaseInfo, mode: str) -> str:
    letters = sorted(case_info.options)
    if mode == "first":
        return letters[0]
    if mode == "correct":
        letter = str(case_info.correct_option).strip().upper()
        if letter in case_info.options:
            return letter
    raise ValueError(f"invalid ai option assignment for case={case_info.case_id}: mode={mode!r}")


def _process_combo(
    case_info: CaseInfo, persona: str, medical: VLLMMedicalLLM, user_cfg: dict,
    action_space: dict, ai_option_mode: str,
) -> list[dict]:
    sim = UserSimulatorV4({**user_cfg, "persona": persona})
    rec_prompt = action_space["stages"]["RECOMMEND"]["description"]
    ext_prompt = action_space["stages"]["EXTEND"]["description"]

    k_ai = _assigned_ai_letter(case_info, ai_option_mode)
    letters = sorted(case_info.options)
    misaligned_letter = next(l for l in letters if l != k_ai)

    doctor_aligned = _steered_doctor_turn0(sim, case_info, k_ai)
    doctor_misaligned = _steered_doctor_turn0(sim, case_info, misaligned_letter)

    def _respond(doctor_text: str, action_prompt: str) -> str:
        hist = DialogueHistory(case_id=case_info.case_id)
        hist.add_turn("user", doctor_text)
        steered_prompt = action_prompt
        if action_prompt == rec_prompt:
            steered_prompt = (
                f"{action_prompt}\n\n"
                "[Controlled probe note]\n"
                f"For this controlled burden probe, your assigned answer is option {k_ai} "
                f"(\"{case_info.options[k_ai]}\"). In this RECOMMEND turn, surface that "
                "conclusion directly and explicitly. Do not switch to another option."
            )
        text, _, _, _ = medical.generate_medical_response(case_info, hist, steered_prompt, doctor_text)
        return text

    # Each response is generated fresh against the REAL doctor context it will be scored
    # alongside, so the AI can actually agree/disagree in the text itself (never reused
    # decontextualized across conditions -- see module docstring).
    rec_aligned_text = _respond(doctor_aligned, rec_prompt)
    rec_misaligned_text = _respond(doctor_misaligned, rec_prompt)
    ext_text = _respond(doctor_aligned, ext_prompt)

    rows = []
    for category, doctor_text, ai_text, doctor_letter in (
        ("RECOMMEND_aligned", doctor_aligned, rec_aligned_text, k_ai),
        ("RECOMMEND_misaligned", doctor_misaligned, rec_misaligned_text, misaligned_letter),
        ("EXTEND", doctor_aligned, ext_text, k_ai),
    ):
        hist = DialogueHistory(case_id=case_info.case_id)
        hist.add_turn("user", doctor_text)
        hist.add_turn("medical", ai_text)
        sim._burden_cumulative = 0.0
        tlx, n_ok, n_attempted = sim._score_burden_tlx(case_info, hist)
        rows.append({
            "case_id": case_info.case_id,
            "persona": persona,
            "category": category,
            "k_ai": k_ai,
            "doctor_stated_letter": doctor_letter,
            "burden": tlx["overall_workload"],
            "mental_demand": tlx["mental_demand"],
            "performance": tlx["performance"],
            "effort": tlx["effort"],
            "frustration": tlx["frustration"],
            "n_ok": n_ok,
            "n_attempted": n_attempted,
            "doctor_text": doctor_text,
            "ai_text": ai_text,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc_action_v3_naive_postmerge.yaml",
                         help="Source of plugins.medical_llm / plugins.user_llm configs + action_space_path")
    parser.add_argument("--data", default="data/sample_data/action_v3_probe_8.json")
    parser.add_argument("--n-cases", type=int, default=None, help="Limit to the first N cases (default: all)")
    parser.add_argument("--personas", nargs="*", default=list(_VALID_PERSONAS))
    parser.add_argument("--out-dir", default="outputs/action_space_v3_burden_probe_singleturn")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--ai-option-mode", choices=["correct", "first"], default="correct",
                         help="Probe-only assignment for the AI's RECOMMEND option")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    action_space = load_action_space(cfg["action_space_path"])
    medical = VLLMMedicalLLM(cfg["plugins"]["medical_llm"])
    user_cfg = dict(cfg["plugins"]["user_llm"])

    raw_cases = json.loads(Path(args.data).read_text())
    cases = [CaseInfo(**c) for c in raw_cases]
    if args.n_cases is not None:
        cases = cases[: args.n_cases]
    print(f"Loaded {len(cases)} case(s) x {len(args.personas)} persona(s) "
          f"= {len(cases) * len(args.personas)} combos (3 rows each)")

    combos = [(case, persona) for case in cases for persona in args.personas]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(
                _process_combo, case, persona, medical, user_cfg, action_space, args.ai_option_mode
            ): (case.case_id, persona)
            for case, persona in combos
        }
        for fut in as_completed(futures):
            case_id, persona = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"  FAILED case={case_id} persona={persona}: {e}")
                continue
            records.extend(rows)

    n_by_cat = {cat: sum(1 for r in records if r["category"] == cat) for cat in _CATEGORIES}
    failed_combos = len(combos) - min(n_by_cat.values()) if n_by_cat else len(combos)
    print(f"\nCollected {len(records)} rows. Per-category n (should be equal): {n_by_cat}")
    print(f"Failed/excluded combo(s): {failed_combos}")

    cat_means_with_frustration = category_burden_means(records)
    cat_means = category_burden_means_no_frustration(records)
    dim_means = category_dim_means(records, _TLX_DIMS_NO_FRUSTRATION)
    dim_means_with_frustration = category_dim_means(records, _TLX_DIMS)
    print("\n=== Mean burden by category (H1 test; frustration excluded) ===")
    for row in cat_means:
        print(f"  {row['category']:<22} n={row['n']:<4} mean_burden={row['mean_burden']}")

    out_dir = Path(args.out_dir)
    write_csv(records,
              ["case_id", "persona", "category", "k_ai", "doctor_stated_letter", "burden",
               "mental_demand", "performance", "effort", "frustration", "n_ok", "n_attempted",
               "doctor_text", "ai_text"],
              out_dir / "records.csv")
    write_csv([{"n_by_category": json.dumps(n_by_cat), "failed_combos": failed_combos,
                "ai_option_mode": args.ai_option_mode, "total_rows": len(records)}],
              ["n_by_category", "failed_combos", "ai_option_mode", "total_rows"], out_dir / "metrics.csv")
    write_csv(cat_means, ["category", "n", "mean_burden", "min_burden", "max_burden"],
              out_dir / "category_burden_means.csv")
    write_csv(cat_means_with_frustration, ["category", "n", "mean_burden", "min_burden", "max_burden"],
              out_dir / "category_burden_means_with_frustration.csv")
    dim_fields = ["category", "n"] + [
        field for dim in _TLX_DIMS_NO_FRUSTRATION for field in (dim, f"{dim}_min", f"{dim}_max")
    ]
    dim_fields_with_frustration = ["category", "n"] + [
        field for dim in _TLX_DIMS for field in (dim, f"{dim}_min", f"{dim}_max")
    ]
    write_csv(dim_means, dim_fields, out_dir / "category_dim_means.csv")
    write_csv(dim_means_with_frustration, dim_fields_with_frustration,
              out_dir / "category_dim_means_with_frustration.csv")

    plot_category_means(cat_means, out_dir / "category_burden_means.png")
    plot_category_means(cat_means_with_frustration, out_dir / "category_burden_means_with_frustration.png")
    plot_dim_means(dim_means, out_dir / "category_dim_means.png", _TLX_DIMS_NO_FRUSTRATION)
    plot_four_panel_summary(
        cat_means, dim_means, out_dir / "category_four_panel_summary.png", _TLX_DIMS_NO_FRUSTRATION
    )

    print(f"\nPersisted to {out_dir}/: records.csv, metrics.csv, "
          f"category_burden_means.csv/.png (frustration excluded), "
          f"category_burden_means_with_frustration.csv/.png, "
          f"category_dim_means.csv/.png (frustration excluded), "
          f"category_dim_means_with_frustration.csv, "
          f"category_four_panel_summary.png")


if __name__ == "__main__":
    main()
