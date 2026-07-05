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
        # absent entirely in specs with no McBurney locution/type sublayer (e.g. the current
        # configs/action_space_v3.yaml -- see its header and core/reward.py's docstring)
        "locutions": {loc["id"]: loc for loc in data.get("locutions", [])},
        "transitions": data.get("transitions", {}),   # v2 control-layer gate (absent in v1)
    }


def build_plugins(config: dict[str, Any], reuse_policy: Any = None):
    """Instantiate plugins from YAML config.

    Returns (user_llm, medical_llm, fact_validator_llm, policy, final_judge).
    Also attaches agenda_planner and resolution_tracker to the returned objects
    as .agenda_planner / .resolution_tracker attributes when configured (agenda arm).

    reuse_policy: when given, that already-built policy is returned as-is instead of
    constructing (and loading) a new one -- used to build extra rollout envs that share the one
    GPU-resident policy while getting their own fresh (stateful) user_llm + (stateless) API
    plugins for parallel GRPO rollouts (see training/grpo/trainer.py).
    """
    from plugins.user_llm.user_simulator.v1 import UserSimulatorV1
    from plugins.user_llm.user_simulator.v2 import UserSimulatorV2
    from plugins.user_llm.user_simulator.v3 import UserSimulatorV3
    from plugins.user_llm.user_simulator.v4 import UserSimulatorV4
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
    from plugins.policy.react_control_policy import ReactControlPolicy
    from plugins.policy.reflexion_policy import ReflexionPolicy
    from plugins.policy.routing_policy import RoutingPolicy
    from plugins.policy.agenda_action_policy import AgendaActionPolicy
    from plugins.policy.policy_ours_v2 import PolicyOursV2
    from plugins.policy.policy_ours_v3 import PolicyOursV3
    from plugins.policy.react_control_v3_policy import ReactControlV3Policy
    from plugins.policy.reflexion_v3_policy import ReflexionV3Policy
    from plugins.policy.ours_v2_api_teacher import OursV2APITeacher
    from plugins.policy.action_space_llm_policy import ActionSpaceLLMPolicy
    from plugins.policy.action_space_v3_llm_policy import ActionSpaceV3LLMPolicy
    from plugins.policy.action_space_v3_oracle_policy import ActionSpaceV3OraclePolicy

    plugin_cfg = config.get("plugins", {})

    _user_llm_map = {"v1": UserSimulatorV1, "v2": UserSimulatorV2, "v3": UserSimulatorV3,
                     "v4": UserSimulatorV4}  # v4 = Bayesian belief core + narrow LLM surface
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
        "react_control":                ReactControlPolicy,     # ReAct baseline on the 3 control stages (vs ours_v2)
        "reflexion":                    ReflexionPolicy,        # Reflexion baseline (reflect->act) on the 3 control stages
        "routing":                      RoutingPolicy,
        "agenda_action":                AgendaActionPolicy,
        "ours_v2":                      PolicyOursV2,  # hybrid 3-way control; validator via policy.use_fact_validator
        "ours_v3":                      PolicyOursV3,  # local trainable EXTEND/RECOMMEND control; validator via policy.use_fact_validator
        "react_control_v3":             ReactControlV3Policy,  # ReAct baseline on the A3 EXTEND/RECOMMEND control (vs ours_v3)
        "reflexion_v3":                 ReflexionV3Policy,     # Reflexion baseline (reflect->act) on the A3 EXTEND/RECOMMEND control (vs ours_v3)
        "ours_v2_teacher":              OursV2APITeacher,  # API-backed (deepseek-v4) teacher for SFT distillation
        "action_space_llm":             ActionSpaceLLMPolicy,  # generic control-layer prompt, driven by action_space_path
        "action_space_v3_llm":          ActionSpaceV3LLMPolicy,  # v3-specific EXTEND/RECOMMEND prompt
        "naive_a3":                     ActionSpaceV3LLMPolicy,  # alias for the v3 naive prompt policy
        "oracle_a3":                    ActionSpaceV3OraclePolicy,  # A3 oracle / meta-oracle policy
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
    _delib_fv_off = (policy_type in ("deliberation_llm", "ours_v2")
                     and not plugin_cfg.get("policy", {}).get("use_fact_validator", True))
    _force_null_fv = policy_type in _POLICIES_WITHOUT_FACT_VALIDATOR or _delib_fv_off
    fact_validator_type = "null" if _force_null_fv else plugin_cfg.get("fact_validator_llm", {}).get("type", "vllm")

    action_space = load_action_space(config.get("action_space_path"))

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
    policy = reuse_policy if reuse_policy is not None else _policy_map[policy_type](policy_cfg, action_space=action_space)
    # final_judge is optional — disabled via plugins.final_judge.enabled: false (e.g. during experiments)
    final_judge = (
        _final_judge_map[final_judge_type](final_judge_cfg)
        if final_judge_cfg.get("enabled", True) else None
    )

    to_load = [user_llm, medical_llm, fact_validator_llm]
    if reuse_policy is None:            # a reused policy is already loaded (GPU-resident); don't reload
        to_load.append(policy)
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
