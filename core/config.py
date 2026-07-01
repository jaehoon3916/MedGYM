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


def load_episode_config(name: str | None, base_dir: str | Path | None = None):
    """Load an initial-user-state preset (EpisodeConfig) by name, like loading a persona.

    `name` is the preset stem in initial_user_state/, e.g. "user_state_5". Returns None when
    `name` is falsy, so the caller falls back to the default EpisodeConfig.
    """
    from core.schemas import EpisodeConfig

    if not name:
        return None
    base = Path(base_dir) if base_dir else Path(__file__).parent.parent / "initial_user_state"
    fname = name if str(name).endswith(".yaml") else f"{name}.yaml"
    return EpisodeConfig(**load_yaml(base / fname))


def load_episode_configs(spec, base_dir: str | Path | None = None):
    """Resolve the experiment.initial_user_state spec into a list of (name, EpisodeConfig|None).

    spec may be:
      None / falsy        → [(None, None)]   (single run with the default EpisodeConfig)
      "all"               → every preset in the folder, in numeric order
      a single name (str) → [(name, cfg)]
      a list of names     → one (name, cfg) per entry
    """
    base = Path(base_dir) if base_dir else Path(__file__).parent.parent / "initial_user_state"
    if not spec:
        return [(None, None)]
    if isinstance(spec, str) and spec.strip().lower() == "all":
        names = sorted(
            (p.stem for p in base.glob("user_state_*.yaml")),
            key=lambda s: int(s.rsplit("_", 1)[-1]),
        )
    elif isinstance(spec, str):
        names = [spec]
    else:
        names = list(spec)
    return [(n, load_episode_config(n, base)) for n in names]


def load_action_space(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "action_space.yaml"
    data = load_yaml(path)
    return {
        "stages": {s["id"]: s for s in data["stages"]},
        "locutions": {loc["id"]: loc for loc in data["locutions"]},
    }


def build_plugins(config: dict[str, Any]):
    """Instantiate plugins from YAML config.

    Returns (user_llm, medical_llm, fact_validator_llm, policy, final_judge).
    Also attaches agenda_planner and resolution_tracker to the returned objects
    as .agenda_planner / .resolution_tracker attributes when configured (agenda arm).
    """
    from plugins.user_llm.user_simulator.v1 import UserSimulatorV1
    from plugins.user_llm.user_simulator.v2 import UserSimulatorV2
    from plugins.user_llm.user_simulator.v3 import UserSimulatorV3
    from plugins.medical_llm.vllm_medical import VLLMMedicalLLM
    from plugins.fact_validator_llm.vllm_fact_validator import VLLMFactValidatorLLM
    from plugins.fact_validator_llm.null_fact_validator import NullFactValidatorLLM
    from plugins.final_judge_llm.vllm_final_judge import VLLMFinalJudgeLLM
    from plugins.policy.rule_policy import RulePolicy
    from plugins.policy.naive_policy import NaivePolicy
    from plugins.policy.prompt_policy import PromptPolicy
    from plugins.policy.qwen_policy import QwenPolicy, LocalQwenPolicy
    from plugins.policy.oracle_policy import RewardOraclePolicy
    from plugins.policy.medcobe_naive_policy import MedCobeNaivePolicy
    from plugins.policy.react_policy import ReactPolicy
    from plugins.policy.medcobe_feedback_policy import MedCobeFeedbackPolicy
    from plugins.policy.deliberation_llm_policy import DeliberationLLMPolicy
    from plugins.policy.routing_policy import RoutingPolicy
    from plugins.policy.agenda_action_policy import AgendaActionPolicy

    plugin_cfg = config.get("plugins", {})

    _user_llm_map = {"v1": UserSimulatorV1, "v2": UserSimulatorV2, "v3": UserSimulatorV3}
    _medical_llm_map = {"vllm": VLLMMedicalLLM}
    _POLICIES_WITHOUT_FACT_VALIDATOR = {"naive", "react", "medcobe_feedback", "medcobe_naive", "routing"}

    _fact_validator_map = {"vllm": VLLMFactValidatorLLM, "null": NullFactValidatorLLM}
    _final_judge_map = {"vllm": VLLMFinalJudgeLLM}
    _policy_map = {
        "rule":                         RulePolicy,
        "naive":                        NaivePolicy,
        "prompt":                       PromptPolicy,
        "baseline_policy":              QwenPolicy,
        "sft_policy":                   QwenPolicy,
        "full_policy":                  QwenPolicy,
        "local_baseline":               LocalQwenPolicy,
        "local_sft":                    LocalQwenPolicy,
        "local_full":                   LocalQwenPolicy,
        "oracle":                       RewardOraclePolicy,
        "medcobe_naive":                MedCobeNaivePolicy,
        "react":                        ReactPolicy,
        "medcobe_feedback":             MedCobeFeedbackPolicy,
        "deliberation_llm":             DeliberationLLMPolicy,  # validator on/off via policy.use_fact_validator
        "routing":                      RoutingPolicy,
        "agenda_action":                AgendaActionPolicy,
    }

    user_type = plugin_cfg.get("user_llm", {}).get("type", "v2")
    medical_type = plugin_cfg.get("medical_llm", {}).get("type", "vllm")
    final_judge_cfg = plugin_cfg.get("final_judge", {})
    final_judge_type = final_judge_cfg.get("type", "vllm")
    ablation_mode = config.get("experiment", {}).get("ablation_mode", "rule")
    policy_type = plugin_cfg.get("policy", {}).get("type", ablation_mode)
    # Fact validator is only meaningful for deliberation_llm (and other needs_verification policies).
    # Force null unconditionally for all others regardless of what the config says. deliberation_llm
    # additionally exposes policy.use_fact_validator=false (ablation) -> also force null, no API calls.
    _delib_fv_off = (policy_type == "deliberation_llm"
                     and not plugin_cfg.get("policy", {}).get("use_fact_validator", True))
    _force_null_fv = policy_type in _POLICIES_WITHOUT_FACT_VALIDATOR or _delib_fv_off
    fact_validator_type = "null" if _force_null_fv else plugin_cfg.get("fact_validator_llm", {}).get("type", "vllm")

    action_space = load_action_space()

    user_llm = _user_llm_map[user_type](plugin_cfg.get("user_llm", {}))
    medical_llm = _medical_llm_map[medical_type](plugin_cfg.get("medical_llm", {}))
    # Keep the validator blind to exactly what the AI is blind to: if the config didn't set
    # fact_validator_llm.see_case_info explicitly, default it to medical_llm.show_case_info
    # (already resolved from info_condition by _resolve_info_condition before build_plugins).
    # So info_condition dense/sparse -> AI blind -> validator blind, automatically. An explicit
    # see_case_info in the config always wins.
    _fv_cfg = dict(plugin_cfg.get("fact_validator_llm", {}))
    if "see_case_info" not in _fv_cfg:
        _fv_cfg["see_case_info"] = bool(plugin_cfg.get("medical_llm", {}).get("show_case_info", True))
    fact_validator_llm = _fact_validator_map[fact_validator_type](_fv_cfg)
    policy_cfg = {**plugin_cfg.get("policy", {}), "mode": policy_type}
    # medcobe_feedback resolves its per-model calibration by target_model; default it to the
    # medical LLM being driven so the user only needs `policy: {type: medcobe_feedback}` in the
    # config (mirrors the original MedCOBE_EMNLP, where the guideline is auto-scoped to the model
    # under test). An explicit policy.target_model still wins.
    if policy_type == "medcobe_feedback" and not policy_cfg.get("target_model"):
        policy_cfg["target_model"] = plugin_cfg.get("medical_llm", {}).get("model")
    policy = _policy_map[policy_type](policy_cfg, action_space=action_space)
    # final_judge is optional — disabled via plugins.final_judge.enabled: false (e.g. during experiments)
    final_judge = (
        _final_judge_map[final_judge_type](final_judge_cfg)
        if final_judge_cfg.get("enabled", True) else None
    )

    to_load = [user_llm, medical_llm, fact_validator_llm, policy]
    if final_judge is not None:
        to_load.append(final_judge)
    for p in to_load:
        p.load()

    return user_llm, medical_llm, fact_validator_llm, policy, final_judge


def build_agenda_plugins(config: dict[str, Any]):
    """Build agenda-arm-specific plugins (DisagreementAnalyzer + ResolutionTracker).

    Called separately from build_plugins() so existing callers are unaffected.
    Returns (agenda_planner, resolution_tracker); either may be None if not configured.
    """
    from plugins.agenda_planner.vllm_agenda_planner import VLLMDisagreementAnalyzer
    from plugins.resolution_tracker.vllm_resolution_tracker import VLLMResolutionTracker

    plugin_cfg = config.get("plugins", {})
    ap_cfg = plugin_cfg.get("agenda_planner", {})
    rt_cfg = plugin_cfg.get("resolution_tracker", {})

    if not ap_cfg:
        return None, None

    _ap_map: dict[str, Any] = {"vllm": VLLMDisagreementAnalyzer}
    _rt_map: dict[str, Any] = {"vllm": VLLMResolutionTracker}

    agenda_planner = _ap_map[ap_cfg.get("type", "vllm")](ap_cfg)
    resolution_tracker = _rt_map[rt_cfg.get("type", "vllm")](rt_cfg) if rt_cfg else None

    agenda_planner.load()
    if resolution_tracker is not None:
        resolution_tracker.load()

    return agenda_planner, resolution_tracker
