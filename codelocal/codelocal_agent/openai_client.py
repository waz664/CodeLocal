from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import AgentConfig


class OpenAIClientError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        return self._post_json(self.config.chat_completions_url, body)

    def list_models(self) -> dict[str, Any]:
        req = urllib.request.Request(self.config.models_url, method="GET")
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise OpenAIClientError(str(exc)) from exc

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAIClientError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise OpenAIClientError(str(exc)) from exc
