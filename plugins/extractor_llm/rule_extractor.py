from typing import Any

from plugins.extractor_llm.base import ExtractorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState


class RuleExtractorLLM(ExtractorLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._schema: list[dict[str, Any]] = config.get("_user_state_schema", [])

    def name(self) -> str:
        return "rule-extractor"

    def load(self) -> None:
        pass

    def extract_user_state(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> UserState:
        if not dialogue_history.turns:
            return UserState(summary="No dialogue yet.")

        last_user_turns = [t for t in dialogue_history.turns if t.speaker == "user"]
        if not last_user_turns:
            return UserState(summary="No user utterance yet.")

        text = last_user_turns[-1].text.lower()
        extracted: dict[str, Any] = {}

        for field in self._schema:
            name = field["name"]
            ftype = field["type"]
            default = field.get("default", "" if ftype == "text" else [])
            extraction = field.get("extraction", {})

            if ftype == "categorical":
                value = default
                if extraction.get("method") == "case_match":
                    correct_kws = [case_info.correct_option.lower(), *case_info.answer.lower().split()]
                    distractor_kws = [w for d in case_info.distractors for w in d.lower().split()]
                    if any(kw in text for kw in correct_kws if len(kw) > 3):
                        value = "true"
                    elif distractor_kws and any(kw in text for kw in distractor_kws if len(kw) > 3):
                        value = "false"
                    else:
                        value = "unknown"
                else:
                    for val, keywords in extraction.get("keyword_map", {}).items():
                        if any(kw in text for kw in keywords):
                            value = val
                            break
                extracted[name] = value

            elif ftype == "list":
                items = []
                for trigger in extraction.get("triggers", []):
                    if "keywords" in trigger:
                        if any(kw in text for kw in trigger["keywords"]):
                            items.append(trigger["value"])
                    elif "from_field" in trigger:
                        src = extracted.get(trigger["from_field"], None)
                        when_val = trigger.get("when_value")
                        if when_val is not None:
                            # trigger when src field equals a specific value
                            if src == when_val:
                                items.append(trigger["value"])
                        else:
                            # trigger when src field is non-empty
                            if src:
                                items.append(trigger["value"])
                extracted[name] = items

            elif ftype == "text":
                template = extraction.get("template", "")
                try:
                    extracted[name] = template.format(**extracted)
                except KeyError:
                    extracted[name] = default

        return UserState(**extracted)
