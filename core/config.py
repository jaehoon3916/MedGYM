from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_user_state_schema(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "user_state.yaml"
    return load_yaml(path)["fields"]


def load_action_space(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "action_space.yaml"
    data = load_yaml(path)
    return {
        "stages": {s["id"]: s for s in data["stages"]},
        "locutions": {loc["id"]: loc for loc in data["locutions"]},
    }


def build_plugins(config: dict[str, Any]):
    """Instantiate plugins from YAML config. Returns (user_llm, medical_llm, fact_validator_llm, policy)."""
    from plugins.user_llm.mock_user import MockUserLLM
    from plugins.user_llm.vllm_user import VLLMUserLLM
    from plugins.medical_llm.mock_medical import MockMedicalLLM
    from plugins.medical_llm.vllm_medical import VLLMMedicalLLM
    from plugins.fact_validator_llm.mock_fact_validator import MockFactValidatorLLM
    from plugins.fact_validator_llm.vllm_fact_validator import VLLMFactValidatorLLM
    from plugins.policy.rule_policy import RulePolicy
    from plugins.policy.naive_policy import NaivePolicy
    from plugins.policy.prompt_policy import PromptPolicy
    from plugins.policy.qwen_policy import QwenPolicy

    plugin_cfg = config.get("plugins", {})

    _user_llm_map = {"mock": MockUserLLM, "vllm": VLLMUserLLM}
    _medical_llm_map = {"mock": MockMedicalLLM, "vllm": VLLMMedicalLLM}
    _fact_validator_map = {"mock": MockFactValidatorLLM, "vllm": VLLMFactValidatorLLM}
    _policy_map = {
        "rule":             RulePolicy,
        "naive":            NaivePolicy,
        "prompt":           PromptPolicy,
        "baseline_policy":  QwenPolicy,
        "sft_policy":       QwenPolicy,
        "full_policy":      QwenPolicy,
    }

    user_type = plugin_cfg.get("user_llm", {}).get("type", "mock")
    medical_type = plugin_cfg.get("medical_llm", {}).get("type", "mock")
    fact_validator_type = plugin_cfg.get("fact_validator_llm", {}).get("type", "mock")
    ablation_mode = config.get("experiment", {}).get("ablation_mode", "rule")
    policy_type = plugin_cfg.get("policy", {}).get("type", ablation_mode)

    action_space = load_action_space()

    user_llm = _user_llm_map[user_type](plugin_cfg.get("user_llm", {}))
    medical_llm = _medical_llm_map[medical_type](plugin_cfg.get("medical_llm", {}))
    fact_validator_llm = _fact_validator_map[fact_validator_type](
        plugin_cfg.get("fact_validator_llm", {})
    )
    policy_cfg = {**plugin_cfg.get("policy", {}), "mode": policy_type}
    policy = _policy_map[policy_type](policy_cfg, action_space=action_space)

    for p in (user_llm, medical_llm, fact_validator_llm, policy):
        p.load()

    return user_llm, medical_llm, fact_validator_llm, policy
