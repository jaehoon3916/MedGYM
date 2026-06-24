"""Standalone annotation-only Flask app — same NASA-TLX human-annotation route as
app.py's /annotate, but without the rollout-viewer pages. Run separately:

    python annotate_app.py  # serves on port 5001
"""
import json
import sys
from pathlib import Path

from flask import Flask, render_template, abort, request, redirect, url_for

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from plugins.user_llm.user_simulator.v3_burden import _TLX_DIMS
import rejudge_rollout as _rejudge

# Same short 1-5 anchor sentences as prompts/user_simulator_v3.yaml's burden_judge_system, so a
# human annotator scores against the SAME calibration the LLM judge was given -- copied, not
# loaded from the yaml, since the human-facing wording is intentionally terser (full official
# NASA-TLX meaning paragraphs aren't needed for a quick per-turn annotation pass).
_NASA_TLX_ANCHORS = {
    "mental_demand": {
        1: "Simple, almost no extra thinking.", 2: "Light reasoning over a small amount of evidence.",
        3: "Moderate reasoning, connecting several facts/options.", 4: "Substantial reasoning, comparing diagnoses/uncertainties.",
        5: "Very complex reasoning, deeply revising the differential.",
    },
    "performance": {
        1: "Feel successful, clear, confident.", 2: "Mostly successful, minor uncertainty.",
        3: "Moderately uncertain, need more checking.", 4: "Judgment feels weak/incomplete/possibly wrong.",
        5: "Feel unable to proceed confidently / failing.",
    },
    "effort": {
        1: "Almost no effort, passive acknowledgment.", 2: "Short confirmation or simple recall.",
        3: "Must explain/organize/provide some reasoning.", 4: "Must compare evidence, revise reasoning, justify a choice.",
        5: "Must defend judgment / reconstruct reasoning in detail.",
    },
    "frustration": {
        1: "Calm, respectful, no pressure.", 2: "Mild discomfort, still easy to continue.",
        3: "AI raises concerns/disagreement, somewhat pressuring.", 4: "AI strongly challenges, may trigger defensiveness.",
        5: "Feel undermined, dismissed, attacked, or pressured.",
    },
}

# Official NASA-TLX meaning + this project's first-person interpretation, copied from the
# [Dimension: ...] blocks in prompts/user_simulator_v3.yaml's burden_judge_*_system prompts
# (same wording the LLM judge is given), so a human annotator applies the same construct.
_DIM_DESCRIPTIONS = {
    "mental_demand": (
        "Official NASA-TLX: how much mental/perceptual activity is required -- thinking, "
        "deciding, calculating, remembering, searching. Here: how much mental work YOU must "
        "do to understand and clinically process the AI's message."
    ),
    "performance": (
        "Official NASA-TLX: how successful you feel you were at the task (anchored Good to "
        "Poor). Here: how successful/confident/capable YOU feel after the AI's message. "
        "POLARITY: LOWER score = you feel you performed BETTER; HIGHER score = WORSE."
    ),
    "effort": (
        "Official NASA-TLX: how hard you had to work mentally/physically to achieve your "
        "level of performance. Here: how hard YOU must work to continue the interaction "
        "appropriately after the AI's message."
    ),
    "frustration": (
        "Official NASA-TLX: how insecure, discouraged, irritated, stressed, or annoyed you "
        "felt, vs. secure, gratified, content, relaxed. Here: how much irritation, stress, "
        "pressure, or loss of autonomy the AI's message imposes on YOU."
    ),
}

app = Flask(__name__)
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def _annotations_path(rel_path: str) -> Path:
    """Mirrors app.py / scripts/rejudge_rollout.py's out_dir convention (rollout_path.parent.parent --
    the experiment-condition folder, one level above rollouts/), so annotation files from either
    app stay in the same place and don't fork into two storage layouts."""
    rollout_file = OUTPUTS_DIR / rel_path
    return rollout_file.parent.parent / f"human_annotations_{rollout_file.stem}.json"


def _load_annotations(rel_path: str) -> dict:
    path = _annotations_path(rel_path)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _read_jsonl(path: Path) -> list:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_rollouts() -> list:
    rollouts = []
    for jsonl_file in sorted(OUTPUTS_DIR.rglob("*.jsonl")):
        if jsonl_file.stem.endswith("_calls"):
            continue
        turns = _read_jsonl(jsonl_file)
        if not turns:
            continue
        first = turns[0]
        rel_path = str(jsonl_file.relative_to(OUTPUTS_DIR))
        saved = _load_annotations(rel_path)
        rollouts.append({
            "path": rel_path,
            "exp_name": jsonl_file.parent.name,
            "case_id": first.get("case_id", "?"),
            "num_turns": len(turns),
            "timestamp": first.get("timestamp", ""),
            "num_annotated": len(saved),
        })
    return rollouts


@app.route("/")
def index():
    rollouts = load_rollouts()
    return render_template("annotate_index.html", rollouts=rollouts)


@app.route("/annotate/<path:rel_path>")
def annotate(rel_path):
    turns = _read_jsonl(OUTPUTS_DIR / rel_path)
    if not turns:
        abort(404)
    case_info = turns[0].get("case_info", {})
    history = turns[-1].get("dialogue_history", [])
    ai_turns = _rejudge.extract_ai_turns(history)
    saved = _load_annotations(rel_path)  # {"<ai_turn_index>": {dim: score, ...}}

    return render_template(
        "annotate.html",
        path=rel_path,
        case_info=case_info,
        ai_turns=ai_turns,
        tlx_dims=_TLX_DIMS,
        anchors=_NASA_TLX_ANCHORS,
        descriptions=_DIM_DESCRIPTIONS,
        saved=saved,
    )


@app.route("/annotate/<path:rel_path>/save", methods=["POST"])
def annotate_save(rel_path):
    turn_index = request.form.get("ai_turn_index")
    if turn_index is None:
        abort(400)
    scores = {}
    for dim in _TLX_DIMS:
        raw = request.form.get(dim)
        if raw is None or not raw.strip():
            abort(400, f"missing score for {dim}")
        v = int(raw)
        if not 1 <= v <= 5:
            abort(400, f"{dim} score out of range: {v}")
        scores[dim] = v

    saved = _load_annotations(rel_path)
    saved[turn_index] = scores
    out_path = _annotations_path(rel_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False))

    return redirect(url_for("annotate", rel_path=rel_path) + f"#turn-{turn_index}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
