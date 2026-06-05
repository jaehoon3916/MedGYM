import json
import sys
from pathlib import Path

from flask import Flask, render_template, abort

sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_user_state_schema

_USER_STATE_FIELDS = load_user_state_schema()

app = Flask(__name__)
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def load_rollouts():
    """Scan outputs/ and return all rollout metadata."""
    rollouts = []
    for jsonl_file in sorted(OUTPUTS_DIR.rglob("*.jsonl")):
        turns = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))
        if not turns:
            continue
        first = turns[0]
        rollouts.append({
            "path": str(jsonl_file.relative_to(OUTPUTS_DIR)),
            "exp_name": jsonl_file.parent.name,
            "case_id": first.get("case_id", "?"),
            "num_turns": len(turns),
            "timestamp": first.get("timestamp", ""),
            "policy": first.get("model_name", {}).get("policy", "?"),
        })
    return rollouts


def load_rollout(rel_path: str):
    path = OUTPUTS_DIR / rel_path
    if not path.exists():
        return None
    turns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


@app.route("/")
def index():
    rollouts = load_rollouts()
    return render_template("index.html", rollouts=rollouts)


@app.route("/rollout/<path:rel_path>")
def rollout(rel_path):
    turns = load_rollout(rel_path)
    if turns is None:
        abort(404)
    case_info = turns[0].get("case_info", {})
    return render_template(
        "rollout.html",
        turns=turns,
        case_info=case_info,
        path=rel_path,
        user_state_fields=_USER_STATE_FIELDS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
