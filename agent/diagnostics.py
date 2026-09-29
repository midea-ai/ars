"""Terminal diagnostics for reward-agent runs."""

from __future__ import annotations

import json
import sys
from typing import Any


def emit_terminal_warning(message: str, payload: dict[str, Any]) -> None:
    """Print model-protocol warnings that should be visible during runs."""

    print(f"[reward-agent warning] {message}: {terminal_payload(payload)}", file=sys.stderr)


def emit_terminal_error(message: str, payload: dict[str, Any]) -> None:
    """Print harness/tool errors that need immediate terminal visibility."""

    print(f"[reward-agent error] {message}: {terminal_payload(payload)}", file=sys.stderr)


def emit_terminal_blocker(message: str, payload: dict[str, Any]) -> None:
    """Print deterministic gate blockers separately from Python exceptions."""

    print(f"[reward-agent blocker] {message}: {terminal_payload(payload)}", file=sys.stderr)


def tool_blocker_payload(
    *,
    stage_id: str,
    tool_call_id: str,
    name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return a compact payload for tool results that explicitly report ok=false."""

    payload: dict[str, Any] = {
        "stage_id": stage_id,
        "tool_call_id": tool_call_id,
        "name": name,
        "ok": bool(result.get("ok")),
        "status": str(result.get("status") or "unknown"),
    }
    for key in ("blocked_items", "next_required_actions", "warnings"):
        if key in result:
            payload[key] = result[key]

    data = result.get("data")
    if not isinstance(data, dict):
        return payload

    for key in (
        "invalid_observation_ids",
        "progress",
        "confirmed_observation_coverage",
    ):
        if key in data:
            payload[key] = data[key]

    validation = data.get("validation")
    if isinstance(validation, dict):
        payload["validation"] = {
            key: validation[key]
            for key in ("ok", "status", "blocked_items", "warnings")
            if key in validation
        }
    return payload


def terminal_payload(payload: dict[str, Any]) -> str:
    """Return a compact one-line payload for terminal diagnostics."""

    display = dict(payload)
    if "content" in display:
        display["content"] = summarize_text(display["content"])
    if "raw_arguments" in display:
        display["raw_arguments"] = summarize_raw_arguments(display["raw_arguments"])
    return json.dumps(display, ensure_ascii=False, sort_keys=True)


def summarize_raw_arguments(value: Any, *, limit: int = 500) -> Any:
    """Keep invalid or oversized tool arguments readable in stderr output."""

    if isinstance(value, str):
        return summarize_text(value, limit=limit)
    return value


def summarize_text(value: Any, *, limit: int = 500) -> str | None:
    """Return a single-line preview of model text for terminal diagnostics."""

    if value is None:
        return None
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
