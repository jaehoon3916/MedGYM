import json
import sys
from pathlib import Path

from flask import Flask, render_template, abort, request, redirect, url_for

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from core.config import load_user_state_schema
from plugins.user_llm.user_simulator.v3_burden import _TLX_DIMS
import rejudge_rollout as _rejudge

_USER_STATE_FIELDS = load_user_state_schema()
# Episode-level persona conditions (set once per episode via EpisodeConfig); used for the case-card summary.
_EPISODE_FIELD_NAMES = {"initial_fact", "certainty", "authority_push", "information_sparcity", "safety_push"}
_EPISODE_FIELDS = [f for f in _USER_STATE_FIELDS if f["name"] in _EPISODE_FIELD_NAMES]

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
    """Mirrors scripts/rejudge_rollout.py's out_dir convention (rollout_path.parent.parent --
    the experiment-condition folder, one level above rollouts/), so a human_annotations file
    and a rejudge_<case_id>.json file from the same episode always land side by side."""
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
        rel = jsonl_file.relative_to(OUTPUTS_DIR)
        parts = rel.parts
        exp_name = parts[0]
        condition = parts[1] if len(parts) >= 3 else ""
        rollouts.append({
            "path": str(rel),
            "exp_name": exp_name,
            "condition": condition,
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
    from collections import defaultdict
    rollouts = load_rollouts()
    grouped: dict[str, list] = defaultdict(list)
    for r in rollouts:
        grouped[r["exp_name"]].append(r)
    groups = dict(sorted(grouped.items()))
    return render_template("index.html", groups=groups, rollouts=rollouts)


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


@app.route("/annotate/<path:rel_path>")
def annotate(rel_path):
    turns, _token_summary, _calls = load_rollout(rel_path)
    if turns is None:
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
    app.run(debug=True, host="0.0.0.0", port=5000)
