from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from plugins.base import BasePlugin
from core.token_tracker import tracker as _tracker


class VLLMBasePlugin(BasePlugin):
    """Shared base for plugins that call a vLLM or OpenAI-compatible server."""

    def __init__(self, config: dict[str, Any]):
        # Call BasePlugin directly (not via super()) so multiple-inheritance subclasses like
        # PromptPolicy/ReactPolicy(VLLMBasePlugin, PolicyPlugin) don't route this into
        # PolicyPlugin.__init__, which requires an action_space arg this call doesn't have.
        BasePlugin.__init__(self, config)
        base_url = config.get("base_url", "http://localhost:8001/v1")
        api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model: str = config.get("model", "")
        self._max_tokens: int = config.get("max_tokens", 512)
        self._temperature: float = config.get("temperature", 0.7)
        self._extra_body: dict[str, Any] | None = config.get("extra_body") or None

    def load(self) -> None:
        pass

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )
        merged_extra = {**(self._extra_body or {}), **(extra_body or {})} or None
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        if response_format:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError(
                f"API returned no choices for model {self._model!r}. "
                f"Possible cause: rate limit, context overflow, or provider error. "
                f"Response: {response}"
            )
        content = response.choices[0].message.content or ""
        _tracker.record(self._model, messages, content, response.usage)
        return content
