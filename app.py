import json
import sys
from pathlib import Path

from flask import Flask, render_template, abort

sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_user_state_schema

_USER_STATE_FIELDS = load_user_state_schema()
# Episode-level persona conditions (set once per episode via EpisodeConfig); used for the case-card summary.
_EPISODE_FIELD_NAMES = {"initial_fact", "certainty", "authority_push", "information_sparcity", "safety_push"}
_EPISODE_FIELDS = [f for f in _USER_STATE_FIELDS if f["name"] in _EPISODE_FIELD_NAMES]

app = Flask(__name__)
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def _read_jsonl(path: Path) -> list:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _read_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_rollouts() -> list:
    rollouts = []
    for jsonl_file in sorted(OUTPUTS_DIR.rglob("*.jsonl")):
        if jsonl_file.stem.endswith("_calls"):
            continue
        turns = _read_jsonl(jsonl_file)
        if not turns:
            continue
        first = turns[0]
        summary_path = jsonl_file.parent / "tokens" / (jsonl_file.stem + "_token_summary.json")
        token_summary = _read_json(summary_path)
        total_tokens = sum(v.get("total_tokens", 0) for v in token_summary.values())
        rollouts.append({
            "path": str(jsonl_file.relative_to(OUTPUTS_DIR)),
            "exp_name": jsonl_file.parent.name,
            "case_id": first.get("case_id", "?"),
            "num_turns": len(turns),
            "timestamp": first.get("timestamp", ""),
            "policy": first.get("model_name", {}).get("policy", "?"),
            "total_tokens": total_tokens,
        })
    return rollouts


def load_rollout(rel_path: str):
    rollout_file = OUTPUTS_DIR / rel_path
    turns = _read_jsonl(rollout_file)
    if not turns:
        return None, {}, []
    token_dir = rollout_file.parent / "tokens"
    token_summary = _read_json(token_dir / (rollout_file.stem + "_token_summary.json"))
    calls = _read_jsonl(token_dir / (rollout_file.stem + "_calls.jsonl"))
    return turns, token_summary, calls


@app.route("/")
def index():
    rollouts = load_rollouts()
    return render_template("index.html", rollouts=rollouts)


@app.route("/rollout/<path:rel_path>")
def rollout(rel_path):
    turns, token_summary, calls = load_rollout(rel_path)
    if turns is None:
        abort(404)
    case_info = turns[0].get("case_info", {})
    episode_config = turns[0].get("episode_config", {})

    # Sibling rollouts in the same experiment folder, for prev/next navigation.
    parent = str(Path(rel_path).parent)
    sibs = sorted(
        (r for r in load_rollouts() if str(Path(r["path"]).parent) == parent),
        key=lambda r: r["path"],
    )
    idx = next((i for i, r in enumerate(sibs) if r["path"] == rel_path), None)
    prev_path = sibs[idx - 1]["path"] if idx not in (None, 0) else None
    next_path = sibs[idx + 1]["path"] if idx is not None and idx < len(sibs) - 1 else None
    position = (idx + 1, len(sibs)) if idx is not None else None

    return render_template(
        "rollout.html",
        turns=turns,
        case_info=case_info,
        episode_config=episode_config,
        path=rel_path,
        user_state_fields=_USER_STATE_FIELDS,
        episode_fields=_EPISODE_FIELDS,
        episode_field_names=list(_EPISODE_FIELD_NAMES),
        token_summary=token_summary,
        calls=calls,
        prev_path=prev_path,
        next_path=next_path,
        position=position,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
