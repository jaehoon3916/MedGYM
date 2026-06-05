from __future__ import annotations

from typing import Any

from openai import OpenAI

from plugins.base import BasePlugin


class VLLMBasePlugin(BasePlugin):
    """Shared base for plugins that call a vLLM OpenAI-compatible server."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        base_url = config.get("base_url", "http://localhost:8001/v1")
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")
        self._model: str = config.get("model", "")
        self._max_tokens: int = config.get("max_tokens", 512)
        self._temperature: float = config.get("temperature", 0.7)

    def load(self) -> None:
        pass

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            extra_body=extra_body or {},
        )
        return response.choices[0].message.content or ""
