"""Small OpenAI-compatible HTTP client for the externally managed vLLM server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import GenerationConfig, VllmConfig


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Parsed subset of a chat completion response."""

    message: dict[str, Any]
    usage: dict[str, Any]
    raw: dict[str, Any]


class VllmChatClient:
    """HTTP client for vLLM's OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        generation_config: GenerationConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.generation_config = generation_config

    def readiness_check(self) -> dict[str, Any]:
        """Verify that the server is reachable and exposes the expected model."""

        models = self._request_json("GET", self.vllm_config.models_url)
        model_ids = [str(item.get("id")) for item in models.get("data", []) if isinstance(item, dict)]
        if model_ids and self.vllm_config.model not in model_ids:
            raise RuntimeError(
                f"vLLM server is reachable, but model {self.vllm_config.model!r} "
                f"is not listed. Available models: {model_ids}"
            )
        return {"ok": True, "models": model_ids}

    def reset_server_cache(self, *, include_mm_cache: bool = True) -> dict[str, Any]:
        """Call dev-mode cache reset endpoints when explicitly requested."""

        results: dict[str, Any] = {}
        results["reset_prefix_cache"] = self._request_json(
            "POST",
            f"{self.vllm_config.root_url}/reset_prefix_cache",
        )
        if include_mm_cache:
            results["reset_mm_cache"] = self._request_json(
                "POST",
                f"{self.vllm_config.root_url}/reset_mm_cache",
            )
        return results

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        final_step: bool = False,
        generation_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send one chat completion request."""

        payload = {
            "model": self.vllm_config.model,
            "messages": messages,
            **self.generation_config.to_request_payload(
                final_step=final_step,
                overrides=generation_overrides,
            ),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        raw = self._request_json("POST", self.vllm_config.chat_url, payload=payload)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"Chat completion response has no choices: {raw}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Chat completion choice has no message object: {choices[0]}")
        usage = raw.get("usage") or {}
        return ChatResponse(message=message, usage=usage if isinstance(usage, dict) else {}, raw=raw)

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.vllm_config.timeout_seconds) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code} for {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach vLLM server at {url}: {exc}") from exc
        if not text.strip():
            return {}
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected JSON object from {url}, got {type(payload).__name__}.")
        return payload
