"""Shared result envelope for reward-agent tool calls."""

from __future__ import annotations

from typing import Any


ToolResult = dict[str, Any]


def tool_success(data: dict[str, Any] | None = None, *, warnings: list[str] | None = None) -> ToolResult:
    """Return the standard success envelope for agent-facing tools."""

    return {
        "ok": True,
        "status": "success",
        "data": dict(data or {}),
        "warnings": list(warnings or []),
    }


def tool_blocked(
    *,
    blocked_items: list[str],
    next_required_actions: list[str] | None = None,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> ToolResult:
    """Return the standard deterministic-gate blocker envelope."""

    return {
        "ok": False,
        "status": "blocked",
        "data": dict(data or {}),
        "blocked_items": list(blocked_items),
        "next_required_actions": list(next_required_actions or []),
        "warnings": list(warnings or []),
    }


def tool_error(
    *,
    error_type: str,
    message: str,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> ToolResult:
    """Return the standard exception/error envelope for failed tool calls."""

    return {
        "ok": False,
        "status": "error",
        "data": data,
        "error": {
            "type": str(error_type),
            "message": str(message),
        },
        "warnings": list(warnings or []),
    }


def tool_result_data(result: dict[str, Any]) -> dict[str, Any]:
    """Return the data object from a standard tool result envelope."""

    data = result.get("data")
    return data if isinstance(data, dict) else {}


def tool_result_ok(result: dict[str, Any]) -> bool:
    """Return whether a standard tool result succeeded."""

    return bool(result.get("ok") is True and result.get("status") == "success")


def tool_result_blocked_items(result: dict[str, Any]) -> list[str]:
    """Return blocker strings from a standard blocked result."""

    items = result.get("blocked_items")
    return [str(item) for item in items] if isinstance(items, list) else []


def tool_result_warnings(result: dict[str, Any]) -> list[str]:
    """Return warning strings from a standard tool result."""

    warnings = result.get("warnings")
    return [str(item) for item in warnings] if isinstance(warnings, list) else []
