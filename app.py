import json
import sys
from pathlib import Path

from flask import Flask, render_template, abort

sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_user_state_schema

_USER_STATE_FIELDS = load_user_state_schema()

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
        summary_path = jsonl_file.with_name(jsonl_file.stem + "_token_summary.json")
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
    base = (OUTPUTS_DIR / rel_path).with_suffix("")
    turns = _read_jsonl(OUTPUTS_DIR / rel_path)
    if not turns:
        return None, {}, []
    token_summary = _read_json(Path(str(base) + "_token_summary.json"))
    calls = _read_jsonl(Path(str(base) + "_calls.jsonl"))
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
    return render_template(
        "rollout.html",
        turns=turns,
        case_info=case_info,
        path=rel_path,
        user_state_fields=_USER_STATE_FIELDS,
        token_summary=token_summary,
        calls=calls,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
