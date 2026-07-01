#!/usr/bin/env python3
"""Offline case-sharding — clinical-vignette adaptation of Microsoft's lost_in_conversation
lazification pipeline (arXiv:2505.06120).

Two LLM stages per case (prompts/shard_case.yaml):
  1. segment:        scenario+caption -> atomic clinical facts
  2. conversational: facts -> one short initial shard (presenting complaint) + ordered shards

Output: one JSON keyed by case_id, each value = {initial_shard, shards:[{shard_id, shard,
segment}]}. shard_id 0 is always the initial shard (the presenting complaint); the doctor
simulator reveals shard_id 0 on turn 0 and at most one further shard per turn thereafter, only
when the AI's question makes it relevant.

Usage (smoke test on the same cases a run config points at):
    python scripts/shard_cases.py --config configs/poc_0630_deliberation_llm_sparse.yaml \
        --out data/sharded/poc_0630.json
Or standalone:
    python scripts/shard_cases.py --n-cases 3 --data-dir data/sample_data --out data/sharded/smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from openai import OpenAI

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.run_scaling_poc import _load_balanced_cases  # noqa: E402
from scripts.run_dialogue import load_dotenv  # noqa: E402  (loads .env so OPENROUTER_API_KEY is set)
from core.json_utils import safe_json_load  # noqa: E402

_PROMPTS = yaml.safe_load((_ROOT / "prompts" / "shard_case.yaml").read_text())


def _chat(client: OpenAI, model: str, system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return safe_json_load(resp.choices[0].message.content or "")


def shard_case(client: OpenAI, model: str, case: dict) -> dict:
    scenario = case.get("scenario", "")
    caption = case.get("caption") or "N/A"

    # Stage 1: atomic clinical facts
    seg_out = _chat(
        client, model,
        _PROMPTS["segment_system"],
        _PROMPTS["segment_user"].format(scenario=scenario, caption=caption),
    )
    segments = seg_out.get("segments", [])

    # Stage 2: initial shard + ordered conversational shards
    conv_out = _chat(
        client, model,
        _PROMPTS["conversational_system"],
        _PROMPTS["conversational_user"].format(segments=json.dumps(segments, ensure_ascii=False, indent=1)),
    )

    initial_shard = str(conv_out.get("initial_shard", "")).strip()
    raw_shards = conv_out.get("shards", []) or []
    # shard_id 0 = initial (presenting complaint); 1..N = findings in importance order
    shards = [{"shard_id": 0, "shard": initial_shard,
               "segment": str(conv_out.get("initial_segment", "")).strip()}]
    for i, s in enumerate(raw_shards, start=1):
        shards.append({"shard_id": i, "shard": str(s.get("shard", "")).strip(),
                       "segment": str(s.get("segment", "")).strip()})

    return {
        "case_id": case.get("case_id"),
        "specialty": case.get("metadata", {}).get("specialty") or case.get("specialty"),
        "n_shards": len(shards),
        "initial_shard": initial_shard,
        "shards": shards,
        "_raw_segments": segments,  # kept for eyeballing shard quality; ignored by consumers
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Reuse a run config's data_dir + n_cases (so shards match a run's case set)")
    p.add_argument("--data-dir", default="data/sample_data")
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--model", default="deepseek/deepseek-v3.2")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    load_dotenv()
    data_dir, n_cases, model, base_url = args.data_dir, args.n_cases, args.model, args.base_url
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        data_dir = cfg.get("experiment", {}).get("data_dir", data_dir)
        n_cases = int(cfg.get("experiment", {}).get("n_cases", n_cases))
        # match the run's policy/validator model unless overridden on the CLI
        if "--model" not in sys.argv:
            model = cfg.get("plugins", {}).get("fact_validator_llm", {}).get("model", model)
        if "--base-url" not in sys.argv:
            base_url = cfg.get("plugins", {}).get("fact_validator_llm", {}).get("base_url", base_url)

    raw_cases, _ = _load_balanced_cases(_ROOT / data_dir, n_cases)
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENROUTER_API_KEY", "EMPTY"))

    out: dict[str, dict] = {}
    for i, case in enumerate(raw_cases, 1):
        cid = str(case.get("case_id"))
        print(f"[{i}/{len(raw_cases)}] sharding case {cid} ...", flush=True)
        try:
            out[cid] = shard_case(client, model, case)
            print(f"    -> {out[cid]['n_shards']} shards", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    !! failed: {e}", flush=True)

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(out)} sharded cases to {out_path}  (model={model})")


if __name__ == "__main__":
    main()
