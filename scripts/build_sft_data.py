"""Generate SFT golden data by driving the env with the reward-oracle teacher (argmax r_align).

For each (case × persona), roll the env forward with the oracle policy and record, per turn, the
student's observation prompt (messages) + the golden action it should emit ("STAGE.locution").
No policy model is loaded here — the teacher is rule-based; cost is the env's sim LLM calls.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dialogue import load_dotenv
from core.config import load_yaml, build_plugins, load_episode_configs
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo
from plugins.policy.qwen_policy import _build_messages, _last_medical_action


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    config.setdefault("plugins", {})["policy"] = {"type": "oracle"}     # teacher; no model load
    config["plugins"].setdefault("final_judge", {})["enabled"] = False  # not needed for SFT data

    user_llm, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    env = MedicalHACEnvironment(
        user_llm, medical_llm, fact_validator_llm, policy, config, final_judge=final_judge,
    )
    action_space = policy.action_space

    raw = json.loads(Path(config["experiment"]["data_path"]).read_text())
    cases = raw if isinstance(raw, list) else [raw]
    personas = load_episode_configs(config.get("experiment", {}).get("initial_user_state"))
    max_turns = int(config.get("experiment", {}).get("max_turns", 6))
    out_path = Path(config.get("sft", {}).get("data_out", "outputs/sft/sft_data.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w") as f:
        for c in cases:
            case_info = CaseInfo(**c)
            for name, ep in personas:
                obs = env.reset(case_info, ep, max_turns=max_turns)
                while not obs.done:
                    kw = {"verification_template": obs.verification} if policy.needs_verification else {}
                    po = policy.select_action(
                        obs.case_info, obs.dialogue_history, obs.current_user_utterance, **kw,
                    )
                    messages = _build_messages(
                        obs.verification, obs.current_user_utterance, obs.dialogue_history,
                        action_space, _last_medical_action(obs.dialogue_history),
                    )
                    f.write(json.dumps(
                        {"messages": messages, "action": po.action_id,
                         "case_id": case_info.case_id, "persona": name},
                        ensure_ascii=False,
                    ) + "\n")
                    n += 1
                    env.step(po)
                    obs = env.observation
                print(f"  {case_info.case_id} / {name}: collected (total {n})")

    print(f"Wrote {n} SFT (prompt, golden_action) pairs → {out_path}")


if __name__ == "__main__":
    main()
