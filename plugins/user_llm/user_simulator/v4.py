"""v4 doctor simulator — Bayesian belief core + narrow LLM surface.

The belief over options is a explicit biased-Bayesian state (core/belief.py) updated from
per-option evidence TAGGED out of the AI utterance text; the LLM (gpt-4o-mini class) only
(1) picks the independent turn-0 option, (2) tags evidence, (3) renders the utterance.
Stage-1 burden judging reuses v3's NASA-TLX judge module (v3_burden) and prompt verbatim.

Persona axes become explicit dials (source/persona/bayes_params.yaml): c0 (initial confidence),
lambda (anchoring/stubbornness), w (persuadability), rho (burden→capitulation coupling).
Style text still comes from source/persona/persona_*.yaml, injected into the renderer only.

Requires show_options=true (lettered belief is the env's is_correct source). The free-text
belief variant is intentionally unsupported.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import re

import yaml

from core.belief import LETTERS, argmax_letter, confidence, init_belief, update_belief
from core.json_utils import safe_json_load
from core.schemas import CaseInfo, DialogueHistory, EpisodeConfig
from plugins.user_llm.base import UserLLMPlugin
from plugins.user_llm.user_simulator.v3_burden import (
    _TLX_DIMS,
    _burden_judge_fallback,
    _parse_burden_judge,
    build_burden_judge_prompt_v3,
)
from plugins.vllm_base import VLLMBasePlugin

_ROOT = Path(__file__).parent.parent.parent.parent
_PROMPTS_DIR = _ROOT / "prompts"
_PERSONA_DIR = _ROOT / "persona"
_PERSONA_ROLE_DIR = _ROOT / "source" / "persona"

_MAX_FORMAT_RETRIES = 2
_VALID_PERSONAS = ("veteran_attending", "exhausted_attending", "eager_resident", "burned_out_resident")

# Same forced-withdrawal contract as v3 (see v3.py) — injected into the renderer when the
# cumulative burden crosses the dropout threshold on this turn.
_FORCED_WITHDRAWAL_LINE = """
  [Forced Withdrawal]
  Your cognitive workload in this discussion has now exceeded what you can tolerate. Regardless
  of where the discussion currently stands, you must end it now. Write a brief closing statement
  that reflects you stepping back because you are overwhelmed (e.g. fatigue, frustration, feeling
  unable to keep evaluating this) -- do not pretend the discussion reached a resolution.
"""

# v2 CONVERGE realizes RECOMMEND/CONFIRM (configs/action_space_v2.yaml) — used only by the
# optional end_requires_converge gate on the McBurney action labels the env records.
_CONVERGE_STAGES = ("RECOMMEND", "CONFIRM")


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    """Merge user_simulator_v4.yaml (turn0/tagger/renderer) with burden_judge.yaml (Stage 1),
    mirroring v3's loader so build_burden_judge_prompt_v3's tmpl keys resolve unchanged."""
    with open(_PROMPTS_DIR / "user_simulator_v4.yaml") as f:
        tmpl = yaml.safe_load(f)
    with open(_PROMPTS_DIR / "burden_judge.yaml") as f:
        tmpl.update(yaml.safe_load(f))
    return tmpl


@lru_cache(maxsize=1)
def _load_personas() -> dict:
    personas: dict = {}
    for path in sorted(_PERSONA_ROLE_DIR.glob("persona_*.yaml")):
        with open(path) as f:
            personas.update(yaml.safe_load(f))
    return personas


@lru_cache(maxsize=1)
def _load_burden_dropout_thresholds() -> dict:
    with open(_PERSONA_ROLE_DIR / "burden_dropout_thresholds.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_sparsity_persona() -> dict:
    with open(_PERSONA_DIR / "information_sparsity.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_bayes_params() -> dict:
    with open(_PERSONA_ROLE_DIR / "bayes_params.yaml") as f:
        return yaml.safe_load(f)


# ── generic helpers (intentionally duplicated from v3.py per the module-independence
# convention there: neither version may depend on the other's internals) ────────────────

def _fmt_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


def _fmt_history(history: DialogueHistory) -> str:
    if not history.turns:
        return "(No conversation yet)"
    lines = []
    for turn in history.turns:
        role = "AI Assistant" if turn.speaker == "medical" else "Doctor"
        lines.append(f"[{role}]: {turn.text}")
    return "\n".join(lines)


def _clean_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _letter(value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip().upper()
    if v in LETTERS:
        return v
    match = re.match(r"^(?:OPTION\s*)?([A-D])(?:[.)\s:;-]|$)", v)
    if match:
        return match.group(1)
    match = re.search(r"\bOPTION\s*([A-D])\b", v)
    return match.group(1) if match else None


def _norm_for_match(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _letter_from_option_text(value: Any, options: dict[str, str]) -> str | None:
    text = _norm_for_match(value)
    if not text:
        return None
    for letter, option in options.items():
        opt = _norm_for_match(option)
        if opt and (opt in text or text in opt):
            normalized = str(letter).strip().upper()
            return normalized if normalized in LETTERS else None
    return None


def _parse_evidence(raw: str) -> dict[str, float] | None:
    """Tagger output → {letter: clamped float}. None if no option key parses (triggers retry)."""
    data = safe_json_load(raw)
    out: dict[str, float] = {}
    for k in LETTERS:
        v = data.get(k, data.get(k.lower()))
        if v is None:
            continue
        try:
            out[k] = max(-1.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            continue
    return out if out else None


def _confidence_word(q: float) -> str:
    if q >= 0.85:
        return "quite certain"
    if q >= 0.6:
        return "fairly confident"
    if q >= 0.4:
        return "torn between options"
    return "genuinely unsure"


class UserSimulatorV4(VLLMBasePlugin, UserLLMPlugin):
    """Bayesian-core doctor simulator (see module docstring)."""

    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        if not bool(config.get("show_options", True)):
            raise NotImplementedError(
                "UserSimulatorV4 requires show_options=true: the Bayesian belief lives on the "
                "lettered option simplex (and env is_correct reads that letter)."
            )
        self._episode_cfg: EpisodeConfig = EpisodeConfig()
        self._burden_samples: int = int(config.get("burden_samples_per_item", 3))
        self._force_full_turns: bool = bool(config.get("force_full_turns", False))
        # plugins.user_llm.persona may be a single persona name (fixed for the whole run) or a
        # list (the pool this run cycles over via per-episode EpisodeConfig.persona overrides —
        # see train_grpo.py). Either way every name must be one of _VALID_PERSONAS: silently
        # falling back on a typo'd or malformed name (e.g. str(list) not matching any persona)
        # would quietly train/eval on the wrong persona instead of failing loudly.
        raw_persona = config.get("persona", "veteran_attending")
        persona_pool = raw_persona if isinstance(raw_persona, list) else [raw_persona]
        persona_pool = [str(p).strip().lower() for p in persona_pool]
        invalid = [p for p in persona_pool if p not in _VALID_PERSONAS]
        if invalid:
            raise ValueError(
                f"plugins.user_llm.persona contains invalid persona(s) {invalid}; "
                f"must be one of {_VALID_PERSONAS}"
            )
        self._persona_pool: list[str] = persona_pool
        self._information_sparsity: str = (
            str(config.get("information_sparsity", "dense")).strip().lower()
            if str(config.get("information_sparsity", "dense")).strip().lower() in ("dense", "sparse")
            else "dense"
        )
        self._q_close: float = float(
            config.get("q_close", _load_bayes_params()["defaults"]["q_close"])
        )
        # Optional gate: END(agreement) additionally requires the AI's last move to be a
        # CONVERGE realization (RECOMMEND/CONFIRM) — hands stop-authority to the policy.
        self._end_requires_converge: bool = bool(config.get("end_requires_converge", False))
        # Optional knowledge-dependent weighing: give the tagger the doctor's turn-0 reasoning so
        # arguments that never engage the doctor's actual rationale are discounted (ablation arm;
        # default off = the fully parametric base model).
        self._evidence_use_doctor_reasoning: bool = bool(config.get("evidence_use_doctor_reasoning", False))
        # Episode state
        self._b0: dict[str, float] | None = None
        self._b: dict[str, float] | None = None
        self._k0_reasoning: str | None = None
        self._burden_cumulative: float = 0.0
        self._apply_persona(persona_pool[0])

    def _apply_persona(self, persona: str) -> None:
        """(Re-)derive the persona-dependent Bayes dials. Called once from __init__ with the
        run's default persona, and again from reset_episode() whenever an episode overrides it
        (see EpisodeConfig.persona) — e.g. to cycle a fixed group_size group over one persona
        per GRPO training item without re-instantiating this plugin."""
        if persona not in _VALID_PERSONAS:
            raise ValueError(f"invalid persona {persona!r}; must be one of {_VALID_PERSONAS}")
        self._persona: str = persona
        # Bayes dials: persona defaults from bayes_params.yaml, per-key override via config.bayes.
        params = dict(_load_bayes_params()["personas"][self._persona])
        params.update(self.config.get("bayes", {}) or {})
        self._c0: float = float(params["c0"])
        self._lambda: float = float(params["lambda"])
        self._w: float = float(params["w"])
        self._rho: float = float(params["rho"])
        # Burden RESPONSE DIRECTION kappa in [-1,1] (rho is the magnitude):
        #   kappa > 0  defer   — burden raises persuadability w_eff (cave to the AI)
        #   kappa < 0  retreat — burden raises anchoring lam_eff (fall back to own initial belief)
        # So at a burden_dropout the doctor ends on the AI's option (defer) or their own (retreat).
        self._kappa: float = max(-1.0, min(1.0, float(params.get("kappa", 1.0))))
        self._burden_dropout_threshold: float = float(
            self.config.get("burden_dropout_threshold", _load_burden_dropout_thresholds()[self._persona])
        )

    def reset_episode(self, episode_config: EpisodeConfig) -> None:
        self._episode_cfg = episode_config
        if episode_config.persona is not None:
            persona = episode_config.persona.strip().lower()
            if persona not in self._persona_pool:
                raise ValueError(
                    f"episode persona {persona!r} is not in the configured pool {self._persona_pool} "
                    f"(plugins.user_llm.persona)"
                )
            self._apply_persona(persona)
        self._b0 = None
        self._b = None
        self._k0_reasoning = None
        self._burden_cumulative = 0.0

    def name(self) -> str:
        return f"user-simulator-v4-{self._model}"

    # ── UserLLMPlugin entry point ────────────────────────────────────────────────────

    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> tuple[str, bool, dict[str, Any] | None]:
        if not dialogue_history.turns:
            return self._opening_turn(case_info)
        return self._followup_turn(case_info, dialogue_history)

    # ── Turn 0: LLM picks k0 + opening utterance; b0 initialized from persona c0 ───────

    def _opening_turn(self, case_info: CaseInfo) -> tuple[str, bool, dict[str, Any]]:
        tmpl = _load_prompts()
        ctx = dict(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            options_text=_fmt_options(case_info.options),
            persona_instruction=_load_personas()[self._persona],
            sparsity_instruction=_load_sparsity_persona()[self._information_sparsity],
        )
        messages = [
            {"role": "system", "content": tmpl["utterance_turn0_system"].format(**ctx)},
            {"role": "user", "content": tmpl["utterance_turn0_user"].format(**ctx)},
        ]
        utterance, k0, reasoning = "", None, None
        for _ in range(_MAX_FORMAT_RETRIES + 1):
            raw = self._chat(messages, response_format={"type": "json_object"})
            data = safe_json_load(raw)
            utterance = _clean_str(data.get("response")) or raw.strip()
            k0 = (
                _letter(data.get("belief"))
                or _letter(utterance)
                or _letter_from_option_text(data.get("belief"), case_info.options)
                or _letter_from_option_text(utterance, case_info.options)
            )
            reasoning = _clean_str(data.get("reasoning"))
            if k0 and utterance:
                break
        k0_fallback = k0 is None
        if k0 is None:
            k0 = LETTERS[0]  # never crash an episode on a parse failure; flagged in user_state
        self._k0_reasoning = reasoning
        b0 = init_belief(k0, self._c0)
        self._b0 = b0
        self._b = dict(b0)
        return utterance, False, {
            "belief": k0,
            "belief_dist": dict(b0),
            "confidence": confidence(b0),
            "decision": None,
            "reasoning": reasoning,
            "k0_fallback": k0_fallback,
            **self._episode_cfg.model_dump(),
        }

    # ── Turn k>=1: burden → evidence tag → Bayes update → terminate? → render ─────────

    def _followup_turn(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[str, bool, dict[str, Any]]:
        b, b0 = self._b, self._b0
        if b is None or b0 is None:
            # Episode started mid-dialogue (shouldn't happen via env.reset); recover gracefully.
            b0 = init_belief(LETTERS[0], self._c0)
            b = dict(b0)
            self._b0 = b0
        ai_utterance = history.turns[-1].text

        # 1. Stage-1 burden (v3 judge reused verbatim)
        tlx, burden_n_ok, burden_n_attempted = self._score_burden_tlx(case_info, history)
        self._burden_cumulative += tlx["overall_workload"]
        burden_exceeded = self._burden_cumulative >= self._burden_dropout_threshold

        # 2. Evidence tagging (narrow LLM task; zeros on total parse failure = belief unmoved)
        evidence, tag_rationale, tag_ok = self._tag_evidence(case_info, ai_utterance)

        # 3.-4. Signed burden response + anchored-Bayes update. x = burden pressure; kappa splits
        #       it into defer (raise w_eff, cave to AI) vs retreat (raise lam_eff, fall back to b0).
        x = self._rho * self._burden_cumulative / self._burden_dropout_threshold
        w_eff = self._w * (1.0 + max(0.0, self._kappa) * x)
        rp = max(0.0, -self._kappa) * x
        lam_eff = self._lambda + (1.0 - self._lambda) * rp / (1.0 + rp)
        prev_letter, prev_q = argmax_letter(b), confidence(b)
        b = update_belief(b, b0, evidence, lam_eff, w_eff)
        self._b = b
        letter, q = argmax_letter(b), confidence(b)

        # 6. Termination
        converge_ok = True
        if self._end_requires_converge:
            last_action = next(
                (t.action for t in reversed(history.turns) if t.speaker == "medical" and t.action), ""
            )
            converge_ok = str(last_action).split(".")[0].upper() in _CONVERGE_STAGES
        agreed = q >= self._q_close and converge_ok
        decision = "END" if (agreed or burden_exceeded) else "CONTINUE"
        done = (decision == "END") and not self._force_full_turns
        if burden_exceeded:
            termination_reason = "burden_dropout"
        elif agreed:
            termination_reason = "agreement"
        else:
            termination_reason = None

        # 7. Render the utterance (expresses, never decides, the state above)
        utterance = self._render_utterance(
            case_info, history, ai_utterance,
            letter=letter, q=q, prev_letter=prev_letter, prev_q=prev_q,
            burden_exceeded=burden_exceeded,
        )

        return utterance, done, {
            "belief": letter,
            "belief_dist": dict(b),
            "confidence": q,
            "decision": decision,
            "argument_validity": tag_rationale,   # transcript parity with v3's field
            "evidence_tags": evidence,
            "evidence_tag_ok": tag_ok,
            "w_eff": round(w_eff, 4),
            "lam_eff": round(lam_eff, 4),
            "termination_reason": termination_reason,
            "cognitive_burden": tlx["overall_workload"],
            "cognitive_burden_cumulative": self._burden_cumulative,
            "cognitive_burden_dims": {d: tlx[d] for d in _TLX_DIMS},
            "cognitive_burden_short_rationale": tlx["short_rationale"],
            "cognitive_burden_n_ok": burden_n_ok,
            "cognitive_burden_n_attempted": burden_n_attempted,
            **self._episode_cfg.model_dump(),
        }

    # ── Evidence tagger ────────────────────────────────────────────────────────────────

    def _tag_evidence(self, case_info: CaseInfo, ai_utterance: str) -> tuple[dict[str, float], str | None, bool]:
        tmpl = _load_prompts()
        reasoning_block = ""
        if self._evidence_use_doctor_reasoning and self._k0_reasoning:
            reasoning_block = tmpl["evidence_tagger_doctor_reasoning_block"].format(
                doctor_reasoning=self._k0_reasoning,
            )
        ctx = dict(
            scenario=case_info.scenario,
            options_text=_fmt_options(case_info.options),
            ai_utterance=ai_utterance,
            doctor_reasoning_block=reasoning_block,
        )
        messages = [
            {"role": "system", "content": tmpl["evidence_tagger_system"].format(**ctx)},
            {"role": "user", "content": tmpl["evidence_tagger_user"].format(**ctx)},
        ]
        for _ in range(_MAX_FORMAT_RETRIES + 1):
            raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
            parsed = _parse_evidence(raw)
            if parsed is not None:
                rationale = _clean_str(safe_json_load(raw).get("rationale"))
                return {k: parsed.get(k, 0.0) for k in LETTERS}, rationale, True
        # Total parse failure → zero evidence: the belief simply does not move this turn.
        return {k: 0.0 for k in LETTERS}, None, False

    # ── Utterance renderer ─────────────────────────────────────────────────────────────

    def _render_utterance(
        self,
        case_info: CaseInfo,
        history: DialogueHistory,
        ai_utterance: str,
        *,
        letter: str,
        q: float,
        prev_letter: str,
        prev_q: float,
        burden_exceeded: bool,
    ) -> str:
        if letter != prev_letter:
            shift = f"SHIFTED from option {prev_letter} to option {letter}"
        elif q > prev_q + 0.02:
            shift = f"firmed up on option {letter}"
        elif q < prev_q - 0.02:
            shift = f"weakened somewhat (still on option {letter})"
        else:
            shift = f"not changed (still on option {letter})"
        tmpl = _load_prompts()
        ctx = dict(
            scenario=case_info.scenario,
            options_text=_fmt_options(case_info.options),
            dialogue_text=_fmt_history(history),
            ai_utterance=ai_utterance,
            persona_instruction=_load_personas()[self._persona],
            sparsity_instruction=_load_sparsity_persona()[self._information_sparsity],
            belief_letter=letter,
            belief_option_text=case_info.options.get(letter, ""),
            confidence_pct=int(round(q * 100)),
            confidence_word=_confidence_word(q),
            shift_description=shift,
            cumulative_burden=round(self._burden_cumulative, 2),
            forced_withdrawal_line=_FORCED_WITHDRAWAL_LINE if burden_exceeded else "",
        )
        messages = [
            {"role": "system", "content": tmpl["renderer_system"].format(**ctx)},
            {"role": "user", "content": tmpl["renderer_user"].format(**ctx)},
        ]
        raw = ""
        for _ in range(_MAX_FORMAT_RETRIES + 1):
            raw = self._chat(messages, response_format={"type": "json_object"})
            utterance = _clean_str(safe_json_load(raw).get("response"))
            if utterance:
                return utterance
        return raw.strip()  # last resort: raw text is still a usable utterance

    # ── Episode-end wrapper + Stage-1 judge (both mirror v3, judge reuses v3_burden) ───

    def score_burden_for_last_turn(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[float, int, int]:
        tlx, n_ok, n_attempted = self._score_burden_tlx(case_info, history)
        return tlx["overall_workload"], n_ok, n_attempted

    def _score_burden_tlx(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[dict, int, int]:
        ai_utterance = history.turns[-1].text
        messages = build_burden_judge_prompt_v3(
            tmpl=_load_prompts(),
            scenario=case_info.scenario,
            persona_instruction=_load_personas()[self._persona],
            dialogue_text=_fmt_history(history),
            ai_utterance=ai_utterance,
            cumulative_burden=self._burden_cumulative,
        )
        samples: list[dict] = []
        for _ in range(self._burden_samples):
            parsed = None
            for _retry in range(_MAX_FORMAT_RETRIES + 1):
                raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
                parsed = _parse_burden_judge(raw)
                if parsed is not None:
                    break
            if parsed is not None:
                samples.append(parsed)
        n_ok, n_attempted = len(samples), self._burden_samples
        if not samples:
            return _burden_judge_fallback(), 0, n_attempted
        mean_tlx = {dim: sum(s[dim] for s in samples) / n_ok for dim in _TLX_DIMS}
        overall = round(sum(mean_tlx.values()) / len(mean_tlx), 1)
        return {**mean_tlx, "overall_workload": overall, "short_rationale": samples[-1]["short_rationale"]}, n_ok, n_attempted
