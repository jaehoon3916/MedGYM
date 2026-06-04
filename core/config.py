from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_action_space(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "action_space.yaml"
    data = load_yaml(path)
    return {item["id"]: {"description": item["description"], "prompt": item["prompt"]}
            for item in data["action_space"]}


def build_plugins(config: dict[str, Any]):
    """Instantiate plugins from YAML config. Returns (user_llm, medical_llm, extractor_llm, policy)."""
    from plugins.user_llm.mock_user import MockUserLLM
    from plugins.medical_llm.mock_medical import MockMedicalLLM
    from plugins.extractor_llm.mock_extractor import MockExtractorLLM
    from plugins.extractor_llm.rule_extractor import RuleExtractorLLM
    from plugins.policy.rule_policy import RulePolicy

    plugin_cfg = config.get("plugins", {})

    _user_llm_map = {"mock": MockUserLLM}
    _medical_llm_map = {"mock": MockMedicalLLM}
    _extractor_map = {"mock": MockExtractorLLM, "rule": RuleExtractorLLM}
    _policy_map = {"rule": RulePolicy}

    user_type = plugin_cfg.get("user_llm", {}).get("type", "mock")
    medical_type = plugin_cfg.get("medical_llm", {}).get("type", "mock")
    extractor_type = plugin_cfg.get("extractor_llm", {}).get("type", "rule")
    policy_type = plugin_cfg.get("policy", {}).get("type", "rule")

    action_space = load_action_space()

    user_llm = _user_llm_map[user_type](plugin_cfg.get("user_llm", {}))
    medical_llm = _medical_llm_map[medical_type](plugin_cfg.get("medical_llm", {}))
    extractor_llm = _extractor_map[extractor_type](plugin_cfg.get("extractor_llm", {}))
    policy = _policy_map[policy_type](plugin_cfg.get("policy", {}), action_space=action_space)

    for p in (user_llm, medical_llm, extractor_llm, policy):
        p.load()

    return user_llm, medical_llm, extractor_llm, policy
