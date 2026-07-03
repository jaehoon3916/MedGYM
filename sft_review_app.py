"""SFT golden-data quality viewer -- separate Flask app from app.py (rollout/transcript viewer).

app.py expects RolloutLogger-format jsonl (dialogue_history, model_name, timestamp per line) and
would render SFT jsonl (build_sft_data.py / build_sft_data_v2.py output: {messages, action,
case_id, persona} per TURN, not per rollout) uselessly -- wrong
case_id/policy/num_turns, empty history on click-through. This app is purpose-built for that
SFT-pair format instead.

Handles both:
  v1 (build_sft_data.py)    -- action is a plain "STAGE.locution" string
  v2 (build_sft_data_v2.py) -- action is a JSON string '{"stage": ..., "action_guidance": ...}';
                               "persona" = the real behavioral dial (bayes_params.yaml).

Usage:
    python sft_review_app.py            # scans outputs/**/sft_data*.jsonl, port 5050
"""
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

from flask import Flask, render_template, abort

sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, template_folder="templates_sft")
OUTPUTS_DIR = Path(__file__).parent / "outputs"

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_action(action_raw) -> dict:
    """Normalizes v1 (plain 'STAGE.locution' string) and v2 (JSON '{"stage":..,"action_guidance":..}')
    action fields into one shape: {"stage": str, "guidance": str|None, "raw": original}."""
    if isinstance(action_raw, str):
        try:
            data = json.loads(action_raw)
            if isinstance(data, dict) and "stage" in data:
                return {"stage": data.get("stage", "?"), "guidance": data.get("action_guidance"), "raw": action_raw}
        except (json.JSONDecodeError, TypeError):
            pass
        # v1 plain "STAGE.locution" (or bare stage)
        stage = action_raw.split(".")[0] if action_raw else "?"
        return {"stage": stage, "guidance": None, "raw": action_raw}
    return {"stage": "?", "guidance": None, "raw": str(action_raw)}


def _last_user_content(messages: list[dict]) -> str:
    """The rendered prompt text shown to the student (dialogue-so-far + current utterance are
    already interpolated into this by the policy's prompt builder) -- what a reviewer should read."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _episode_key(r: dict) -> tuple[str, str]:
    """Groups turn-level SFT rows into a single case/persona episode."""
    return (r.get("case_id") or "?", r.get("persona") or "?")


def find_datasets() -> list[dict]:
    out = []
    for jsonl_file in sorted(OUTPUTS_DIR.rglob("sft_data*.jsonl")):
        try:
            recs = _read_jsonl(jsonl_file)
        except (json.JSONDecodeError, OSError):
            continue
        if not recs:
            continue
        episodes = defaultdict(list)
        for r in recs:
            episodes[_episode_key(r)].append(r)
        stage_counts = Counter(_parse_action(r["action"])["stage"] for r in recs)
        lens = [len(v) for v in episodes.values()]
        personas = sorted({k[1] for k in episodes})
        out.append({
            "path": str(jsonl_file.relative_to(OUTPUTS_DIR)),
            "n_pairs": len(recs),
            "n_episodes": len(episodes),
            "personas": personas,
            "stage_counts": dict(stage_counts.most_common()),
            "min_turns": min(lens) if lens else 0,
            "max_turns": max(lens) if lens else 0,
            "avg_turns": round(sum(lens) / len(lens), 1) if lens else 0,
        })
    return out


def load_dataset(rel_path: str) -> dict[tuple, list[dict]]:
    fp = OUTPUTS_DIR / rel_path
    if not fp.exists():
        abort(404)
    recs = _read_jsonl(fp)
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        episodes[_episode_key(r)].append(r)
    return dict(episodes)


@app.route("/")
def index():
    return render_template("sft_index.html", datasets=find_datasets())


@app.route("/dataset/<path:rel_path>")
def dataset(rel_path):
    episodes = load_dataset(rel_path)
    rows = []
    for (case_id, persona), turns in sorted(episodes.items()):
        stages = [_parse_action(t["action"])["stage"] for t in turns]
        rows.append({
            "case_id": case_id, "persona": persona, "n_turns": len(turns),
            "stage_seq": " → ".join(stages),
            "has_challenge": "CHALLENGE" in stages, "has_converge": "CONVERGE" in stages,
        })
    return render_template("sft_dataset.html", path=rel_path, rows=rows)


@app.route("/dataset/<path:rel_path>/episode/<case_id>/<persona>")
def episode(rel_path, case_id, persona):
    episodes = load_dataset(rel_path)
    key = (case_id, persona)
    turns = episodes.get(key)
    if turns is None:
        abort(404)
    rendered = []
    for i, t in enumerate(turns):
        a = _parse_action(t["action"])
        rendered.append({
            "turn": i,
            "prompt": _last_user_content(t["messages"]),
            "stage": a["stage"],
            "guidance": a["guidance"],
            "raw_action": a["raw"],
        })

    # prev/next among all episodes in this dataset, sorted the same way dataset() lists them
    keys = sorted(episodes.keys())
    idx = keys.index(key)
    prev_key = keys[idx - 1] if idx > 0 else None
    next_key = keys[idx + 1] if idx < len(keys) - 1 else None

    return render_template(
        "sft_episode.html", path=rel_path, case_id=case_id, persona=persona,
        turns=rendered, prev_key=prev_key, next_key=next_key, position=(idx + 1, len(keys)),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
