"""Workflow runner for the reward-agent harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .client import ChatResponse, VllmChatClient
from .config import GenerationConfig, RewardAgentConfig, StageSpec, load_prompt_text
from .diagnostics import (
    emit_terminal_blocker,
    emit_terminal_error,
    emit_terminal_warning,
    summarize_raw_arguments,
    tool_blocker_payload,
)
from .schemas import PendingMediaAttachment, UsageRecord
from .session import RewardAgentSession
from .tool_protocol import tool_error, tool_result_ok


_MAX_CONSECUTIVE_NO_TOOL_RECOVERIES = 3
_REFERENCE_EXAMPLES_PROMPT = "prompts/reference_examples.md"
_TRAJECTORY_OUTCOME_GROUND_TRUTH_PROMPT = "prompts/trajectory_outcome_ground_truth.md"
_TRAJECTORY_OUTCOME_PLACEHOLDER = "{{trajectory_outcome}}"


class ChatClientProtocol(Protocol):
    """Protocol used by tests and alternate clients."""

    def readiness_check(self) -> dict[str, Any]:
        ...

    def reset_server_cache(self, *, include_mm_cache: bool = True) -> dict[str, Any]:
        ...

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        final_step: bool = False,
        generation_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        ...


class RewardAgentRunner:
    """Run a workflow over one reward-agent session."""

    def __init__(
        self,
        *,
        session: RewardAgentSession,
        config: RewardAgentConfig | None = None,
        client: ChatClientProtocol | None = None,
        print_context: bool = True,
        check_server_ready: bool = True,
    ) -> None:
        self.session = session
        self.config = config or session.config
        self.client = client or VllmChatClient(
            vllm_config=self.config.vllm,
            generation_config=self.config.generation,
        )
        self.print_context = bool(print_context)
        self.check_server_ready = bool(check_server_ready)

    def run(self) -> dict[str, Any]:
        """Execute all workflow stages and return the final manifest-like status."""

        workflow = self.config.workflow
        if workflow is None:
            raise ValueError("RewardAgentConfig.workflow is required.")
        if self.check_server_ready:
            ready = self.client.readiness_check()
            self.session.append_trace("vllm_readiness_check", ready)
        if self.config.vllm.reset_server_cache == "before-run":
            reset = self.client.reset_server_cache(include_mm_cache=True)
            self.session.append_trace("vllm_cache_reset", reset)

        # Reference examples are part of conversation bootstrap, not a stage
        # input. They are therefore present at run start and automatically
        # reattached whenever a fresh conversation is created.
        reference_messages = self.session.reference_example_messages()
        common_prompt = self._load_prompt_text(workflow.common_prompt)
        if self.config.is_success_ground_truth.enabled:
            if self.session.is_success_ground_truth is None:
                raise RuntimeError(
                    "is_success_ground_truth.enabled=true but the session has no ground-truth label."
                )
            outcome_prompt = _render_trajectory_outcome_prompt(
                self._load_prompt_text(_TRAJECTORY_OUTCOME_GROUND_TRUTH_PROMPT),
                is_success=self.session.is_success_ground_truth,
            )
            common_prompt = common_prompt.rstrip() + "\n\n" + outcome_prompt.lstrip()
        if reference_messages:
            common_prompt = (
                common_prompt.rstrip()
                + "\n\n"
                + self._load_prompt_text(_REFERENCE_EXAMPLES_PROMPT).lstrip()
            )
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": common_prompt},
            *reference_messages,
        ]
        self.session.append_trace(
            "conversation_bootstrap_created",
            {
                "reason": "run_start",
                "num_reference_examples": len(reference_messages),
                "reference_prompt_path": (
                    _REFERENCE_EXAMPLES_PROMPT if reference_messages else None
                ),
            },
        )
        messages = list(base_messages)
        run_status = "running"
        blocked_stage_id: str | None = None
        blocked_detail: dict[str, Any] | None = None
        for stage in workflow.stages:
            if stage.fresh_session:
                messages = list(base_messages)
                self.session.append_trace(
                    "stage_fresh_session_started",
                    {
                        "stage_id": stage.stage_id,
                        "num_reference_examples": len(reference_messages),
                    },
                )
            stage_status = self._run_stage(
                stage,
                messages,
                common_prompt_path=workflow.common_prompt,
            )
            if stage_status["status"] != "completed":
                run_status = str(stage_status["status"])
                blocked_stage_id = stage.stage_id
                blocked_detail = stage_status
                break
            if self.session.final_result is not None:
                run_status = "completed"
                break

        if self.session.final_result is None:
            if run_status == "running":
                run_status = "blocked_not_finalized_by_model"
            self.session.save_result(
                summary=(
                    "# Reward Agent Blocked\n\n"
                    "The workflow ended without a successful `complete_stage` finalization. "
                    "This is a harness-level fallback, not a completed progress audit. "
                    "Review trace.jsonl for the last model response and tool history."
                ),
                # A blocked workflow is not a completed assessment. Preserve a
                # maximal review signal in its per-attempt result artifact even
                # though batch status handling keeps it out of normal outputs.
                external_review_need=5,
                external_review_focus=(
                    "Review the unresolved progress assessment and the final model/tool trace."
                ),
                payload={
                    "status": run_status,
                    "blocked_stage_id": blocked_stage_id,
                    "blocked_detail": blocked_detail,
                    "progress_validation_ok": False,
                },
            )
        elif run_status == "running":
            run_status = "completed"

        return {
            "case_id": self.session.case_id,
            "run_dir": str(self.session.run_dir),
            "finalized": self.session.final_result is not None,
            "completed": run_status == "completed",
            "status": run_status,
            "blocked_stage_id": blocked_stage_id,
            "num_visual_assets": len(self.session.visual_assets),
            "num_observations": len(self.session.observations),
        }

    def _run_stage(
        self,
        stage: StageSpec,
        messages: list[dict[str, Any]],
        *,
        common_prompt_path: str,
    ) -> dict[str, Any]:
        stage_prompt = self._load_prompt_text(stage.prompt)
        stage_context_payload = self.session.build_case_context_payload(stage_id=stage.stage_id)
        stage_context = json.dumps(stage_context_payload, indent=2, ensure_ascii=False)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{stage_prompt}\n\n"
                    f"Current stage id: {stage.stage_id}\n\n"
                    "Case context:\n"
                    f"```json\n{stage_context}\n```"
                ),
            }
        )
        initial_messages, initial_media_trace, initial_context_trace = self._stage_initial_messages(stage)
        if initial_messages:
            messages.extend(initial_messages)
        tools = self.session.tool_manifest(
            allowed_tools=stage.allowed_tools,
            stage_id=stage.stage_id,
            tool_config=stage.tool_config,
        )
        self.session.append_trace(
            "stage_inputs_attached",
            {
                "stage_id": stage.stage_id,
                "prompt": {
                    "common_prompt_path": common_prompt_path,
                    "stage_prompt_path": stage.prompt,
                },
                "case_context": stage_context_payload,
                "initial_media": initial_media_trace,
                "initial_context": initial_context_trace,
            },
        )
        self.session.append_trace(
            "stage_started",
            {
                "stage_id": stage.stage_id,
                "max_steps": stage.max_steps,
                "fresh_session": stage.fresh_session,
                "allowed_tools": list(stage.allowed_tools),
                "initial_media_config": dict(stage.initial_media),
                "initial_context_config": dict(stage.initial_context),
                "tool_config": dict(stage.tool_config),
                "generation_overrides": dict(stage.generation),
                "promote_observations_to_confirmed_on_complete": (
                    stage.promote_observations_to_confirmed_on_complete
                ),
                "effective_thinking_token_budget": _effective_generation_value(
                    self.config.generation,
                    dict(stage.generation),
                    "thinking_token_budget",
                ),
            },
        )

        pending_media: list[PendingMediaAttachment] = []
        consecutive_no_tool_calls = 0
        stage_generation_overrides = dict(stage.generation)
        effective_thinking_token_budget = _effective_generation_value(
            self.config.generation,
            stage_generation_overrides,
            "thinking_token_budget",
        )
        for step_index in range(1, int(stage.max_steps) + 1):
            if pending_media:
                messages.extend(self._media_messages(pending_media))
                self.session.append_trace(
                    "media_attached",
                    {
                        "stage_id": stage.stage_id,
                        "step_index": step_index,
                        "asset_ids": [media.asset_id for media in pending_media],
                    },
                )
                pending_media = []

            chat_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": tools,
                "final_step": False,
            }
            if stage_generation_overrides:
                chat_kwargs["generation_overrides"] = stage_generation_overrides
            response = self.client.chat(**chat_kwargs)
            thinking_diagnostics = _build_thinking_diagnostics(
                message=response.message,
                usage=response.usage,
                thinking_token_budget=effective_thinking_token_budget,
                forced_end_phrase=self.config.generation.thinking_budget_forced_end_phrase,
            )
            model_response_path = self._save_model_response(
                response=response,
                stage_id=stage.stage_id,
                step_index=step_index,
                thinking_diagnostics=thinking_diagnostics,
            )
            thinking_diagnostics["model_response_path"] = model_response_path
            self._emit_thinking_budget_warning(
                stage_id=stage.stage_id,
                step_index=step_index,
                diagnostics=thinking_diagnostics,
            )
            messages.append(response.message)
            self._record_usage(
                stage_id=stage.stage_id,
                step_index=step_index,
                usage=response.usage,
                thinking_diagnostics=thinking_diagnostics,
            )

            tool_calls = response.message.get("tool_calls") or []
            if not tool_calls:
                consecutive_no_tool_calls += 1
                no_tool_payload = {
                    "stage_id": stage.stage_id,
                    "step_index": step_index,
                    "content": response.message.get("content"),
                    "finish_reason": self._finish_reason(response),
                    "consecutive_no_tool_calls": consecutive_no_tool_calls,
                    "max_consecutive_recoveries": _MAX_CONSECUTIVE_NO_TOOL_RECOVERIES,
                    **self._no_tool_usage_diagnostics(
                        response,
                        thinking_token_budget=effective_thinking_token_budget,
                    ),
                }
                if consecutive_no_tool_calls <= _MAX_CONSECUTIVE_NO_TOOL_RECOVERIES:
                    self.session.append_trace(
                        "stage_no_tool_calls_recovering",
                        no_tool_payload,
                    )
                    emit_terminal_warning(
                        "model response had no tool calls",
                        no_tool_payload,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._no_tool_recovery_message(
                                stage=stage,
                                attempt=consecutive_no_tool_calls,
                            ),
                        }
                    )
                    continue
                self.session.append_trace(
                    "stage_blocked_no_tool_calls",
                    no_tool_payload,
                )
                emit_terminal_error(
                    "stage blocked after repeated no-tool model responses",
                    no_tool_payload,
                )
                return {
                    "status": "blocked_no_tool_calls",
                    "stage_id": stage.stage_id,
                    "step_index": step_index,
                    "consecutive_no_tool_calls": consecutive_no_tool_calls,
                }

            consecutive_no_tool_calls = 0
            tool_results = self._execute_tool_calls(
                tool_calls,
                messages=messages,
                stage_id=stage.stage_id,
                allowed_tools=stage.allowed_tools,
            )
            pending_media = self.session.pending_media_for_tool_results(tool_results)
            if _has_successful_stage_completion(tool_calls, tool_results):
                self.session.append_trace(
                    "stage_completed",
                    {"stage_id": stage.stage_id, "step_index": step_index},
                )
                return {"status": "completed", "stage_id": stage.stage_id, "step_index": step_index}
        else:
            self.session.append_trace(
                "stage_max_steps_reached",
                {"stage_id": stage.stage_id, "max_steps": stage.max_steps},
            )
            return {
                "status": "blocked_max_steps",
                "stage_id": stage.stage_id,
                "max_steps": stage.max_steps,
            }

        return {"status": "blocked_stage_exited", "stage_id": stage.stage_id}

    def _load_prompt_text(self, path: str) -> str:
        """Load a prompt and inject resolved runtime config values."""

        return _render_prompt_template(load_prompt_text(path), config=self.config)

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        messages: list[dict[str, Any]],
        stage_id: str,
        allowed_tools: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        allowed = {str(tool) for tool in allowed_tools} if allowed_tools else None
        for tool_call in tool_calls:
            tool_call_id = str(tool_call.get("id") or f"tool_call_{len(self.session.tool_history) + 1:04d}")
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments")
            arguments: dict[str, Any] = {}
            tool_error_payload: dict[str, Any] | None = None
            if allowed is not None and name not in allowed:
                message = f"Tool {name!r} is not available in stage {stage_id!r}."
                result = tool_error(
                    error_type="ToolNotAllowed",
                    message=message,
                    data={"allowed_tools": sorted(allowed)},
                )
                tool_error_payload = {
                    "stage_id": stage_id,
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "error": "ToolNotAllowed",
                    "message": message,
                    "allowed_tools": sorted(allowed),
                }
            else:
                try:
                    arguments = _parse_tool_arguments(raw_arguments)
                    result = self.session.dispatch_tool(name, arguments, stage_id=stage_id)
                except Exception as exc:  # noqa: BLE001 - tool errors should be returned to the model.
                    result = tool_error(error_type=type(exc).__name__, message=str(exc))
                    tool_error_payload = {
                        "stage_id": stage_id,
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "raw_arguments": summarize_raw_arguments(raw_arguments),
                    }
            self.session.record_tool_call(
                tool_call_id=tool_call_id,
                name=name,
                arguments=arguments,
                result=result,
            )
            self.session.append_trace(
                "tool_call_executed",
                {
                    "tool_call_id": tool_call_id,
                    "stage_id": stage_id,
                    "name": name,
                    "arguments": arguments,
                    "raw_arguments": raw_arguments if arguments == {} and tool_error_payload else None,
                    "result": result,
                },
            )
            if tool_error_payload is not None:
                emit_terminal_error("tool call failed", tool_error_payload)
            elif result.get("ok") is False:
                emit_terminal_blocker(
                    "tool returned ok=false",
                    tool_blocker_payload(
                        stage_id=stage_id,
                        tool_call_id=tool_call_id,
                        name=name,
                        result=result,
                    ),
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            results.append(result)
        return results

    def _record_usage(
        self,
        *,
        stage_id: str,
        step_index: int,
        usage: dict[str, Any],
        thinking_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        record = UsageRecord.from_response_usage(
            usage,
            context_window=self.config.vllm.context_window,
            reserved_output_tokens=self.config.generation.reserved_output_tokens,
        )
        data = record.to_dict()
        data["stage_id"] = stage_id
        data["step_index"] = int(step_index)
        if thinking_diagnostics is not None:
            data["thinking_diagnostics"] = _compact_thinking_diagnostics(thinking_diagnostics)
        self.session.append_trace("context_usage", data)
        if self.print_context:
            print(format_usage_line(record))

    def _save_model_response(
        self,
        *,
        response: ChatResponse,
        stage_id: str,
        step_index: int,
        thinking_diagnostics: dict[str, Any],
    ) -> str:
        """Persist the complete model response outside the model conversation."""

        relative_path = Path("model_responses") / f"{_safe_artifact_stem(stage_id)}_step_{step_index:04d}.json"
        output_path = self.session.run_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "stage_id": stage_id,
            "step_index": int(step_index),
            "finish_reason": self._finish_reason(response),
            "usage": response.usage,
            "message": response.message,
            "thinking_diagnostics": dict(thinking_diagnostics),
            "raw_response_metadata": _raw_response_metadata(response.raw),
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trace_payload = {
            "stage_id": stage_id,
            "step_index": int(step_index),
            "path": str(relative_path),
            "finish_reason": self._finish_reason(response),
            "thinking_diagnostics": _compact_thinking_diagnostics(thinking_diagnostics),
        }
        self.session.append_trace("model_response_saved", trace_payload)
        return str(relative_path)

    @staticmethod
    def _emit_thinking_budget_warning(
        *,
        stage_id: str,
        step_index: int,
        diagnostics: dict[str, Any],
    ) -> None:
        """Print a compact warning only when the thinking budget appears exhausted."""

        status = diagnostics.get("thinking_budget_status")
        if status not in {"reached", "likely_reached"}:
            return
        message = "thinking budget reached" if status == "reached" else "thinking budget likely reached"
        emit_terminal_warning(
            message,
            {
                "stage_id": stage_id,
                "step_index": int(step_index),
                "thinking_budget_status": status,
                "thinking_budget_status_source": diagnostics.get("thinking_budget_status_source"),
                "thinking_token_budget": diagnostics.get("thinking_token_budget"),
                "reported_reasoning_tokens": diagnostics.get("reported_reasoning_tokens"),
                "reported_reasoning_tokens_source": diagnostics.get("reported_reasoning_tokens_source"),
                "completion_tokens": diagnostics.get("completion_tokens"),
                "forced_end_phrase_seen": diagnostics.get("forced_end_phrase_seen"),
                "model_response_path": diagnostics.get("model_response_path"),
            },
        )

    @staticmethod
    def _media_messages(attachments: list[PendingMediaAttachment]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": attachment.to_message_content(),
            }
            for attachment in attachments
        ]

    def _stage_initial_messages(
        self,
        stage: StageSpec,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Build runner-provided stage inputs such as full video or progress data."""

        messages: list[dict[str, Any]] = []
        initial_media_trace: dict[str, Any] = {}
        if stage.initial_media.get("full_video", False):
            try:
                attachment = self.session.full_video_attachment()
            except Exception as exc:  # noqa: BLE001 - report fallback instructions to the model.
                payload = {
                    "stage_id": stage.stage_id,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
                initial_media_trace["full_video"] = {
                    "status": "unavailable",
                    "error": payload["error"],
                    "message": payload["message"],
                }
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Initial full-video attachment is unavailable for this stage.\n\n"
                            f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n\n"
                            "Use `inspect_visual` with smaller closed frame ranges, a larger "
                            "`sample_every_n`, or multiple segmented requests before recording observations."
                        ),
                    }
                )
            else:
                asset = next(
                    (
                        item
                        for item in self.session.visual_assets
                        if item.asset_id == attachment.asset_id
                    ),
                    None,
                )
                metadata_summary = self.session.metadata_resolution_summary()
                initial_media_trace["full_video"] = {
                    "status": "attached",
                    "asset_id": attachment.asset_id,
                    "num_frames": len(attachment.frame_indices),
                    "frame_shape_hwc": metadata_summary.get("frame_shape_hwc"),
                    "resolution": metadata_summary.get("resolution"),
                    "asset": self.session.compact_visual_asset_summary(asset)
                    if asset is not None
                    else None,
                }
                messages.extend(self._media_messages([attachment]))

        initial_context_trace: dict[str, Any] = {}
        if stage.initial_context.get("progress_snapshot", False):
            initial_context_trace["progress_snapshot"] = self.session.read_progress(
                view="both",
                trace_source="runner_initial_context",
            )
        if stage.initial_context.get("progress_summary", False):
            initial_context_trace["progress_summary"] = self.session.summarize_progress(
                trace_source="runner_initial_context",
            )
        if initial_context_trace:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Runner-provided initial context for this stage:\n"
                        f"```json\n{json.dumps(initial_context_trace, ensure_ascii=False)}\n```"
                    ),
                }
            )
        return messages, initial_media_trace, initial_context_trace

    @staticmethod
    def _finish_reason(response: ChatResponse) -> str | None:
        choices = response.raw.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        reason = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None

    def _no_tool_recovery_message(self, *, stage: StageSpec, attempt: int) -> str:
        """Tell the model that stage progress must be expressed through tools."""

        allowed_tools = ", ".join(f"`{tool}`" for tool in stage.allowed_tools)
        return (
            "Your previous response ended without a tool call. Natural language cannot complete "
            "or advance this stage.\n\n"
            f"Current stage: `{stage.stage_id}`.\n"
            f"Allowed tools: {allowed_tools}.\n"
            f"Recovery attempt: {attempt}/{_MAX_CONSECUTIVE_NO_TOOL_RECOVERIES}.\n\n"
            "Continue by calling exactly one allowed tool for the current stage. If you believe "
            "the stage is complete or blocked, call `complete_stage()` so the harness can run "
            "the deterministic gate and return blockers if needed. Do not write a long analysis "
            "before the tool call."
        )

    def _no_tool_usage_diagnostics(
        self,
        response: ChatResponse,
        *,
        thinking_token_budget: int | None,
    ) -> dict[str, Any]:
        """Return trace-only diagnostics for no-tool responses."""

        usage = response.usage or {}
        return {
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "thinking_token_budget": thinking_token_budget,
            "max_tokens_per_step": self.config.generation.max_tokens_per_step,
        }


def format_usage_line(record: UsageRecord) -> str:
    """Return the terminal-friendly context budget line."""

    return (
        "ctx "
        f"prompt={record.prompt_tokens} "
        f"completion={record.completion_tokens} "
        f"total={record.total_tokens} "
        f"limit={record.context_window} "
        f"remaining~={record.remaining_tokens}"
    )


def _build_thinking_diagnostics(
    *,
    message: dict[str, Any],
    usage: dict[str, Any],
    thinking_token_budget: int | None,
    forced_end_phrase: str | None,
) -> dict[str, Any]:
    """Return harness-only diagnostics for model reasoning output."""

    reasoning_text, reasoning_field = _extract_reasoning_text(message)
    completion_tokens = _safe_int(usage.get("completion_tokens"), default=0)
    reported_tokens, reported_source = _extract_reported_reasoning_tokens(usage)
    forced_phrase = (forced_end_phrase or "").strip()
    forced_end_phrase_seen = bool(forced_phrase and reasoning_text is not None and forced_phrase in reasoning_text)

    budget = None if thinking_token_budget is None else int(thinking_token_budget)
    if budget is None:
        budget_status = "not_configured"
        budget_status_source = "no_thinking_token_budget"
    elif reported_tokens is not None:
        budget_status = "reached" if reported_tokens >= budget else "not_reached"
        budget_status_source = reported_source
    elif completion_tokens < budget:
        budget_status = "not_reached"
        budget_status_source = "completion_tokens_below_budget"
    elif forced_end_phrase_seen:
        budget_status = "likely_reached"
        budget_status_source = "forced_end_phrase"
    else:
        budget_status = "unknown"
        budget_status_source = "no_reported_reasoning_tokens"

    return {
        "thinking_token_budget": budget,
        "completion_tokens": completion_tokens,
        "reported_reasoning_tokens": reported_tokens,
        "reported_reasoning_tokens_source": reported_source,
        "reasoning_field": reasoning_field,
        "reasoning_char_count": 0 if reasoning_text is None else len(reasoning_text),
        "forced_end_phrase_configured": bool(forced_phrase),
        "forced_end_phrase_seen": forced_end_phrase_seen,
        "thinking_budget_status": budget_status,
        "thinking_budget_status_source": budget_status_source,
    }


def _compact_thinking_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return trace-safe thinking diagnostics without full reasoning content."""

    keys = (
        "thinking_token_budget",
        "completion_tokens",
        "reported_reasoning_tokens",
        "reported_reasoning_tokens_source",
        "reasoning_field",
        "reasoning_char_count",
        "forced_end_phrase_configured",
        "forced_end_phrase_seen",
        "thinking_budget_status",
        "thinking_budget_status_source",
        "model_response_path",
    )
    return {key: diagnostics.get(key) for key in keys if key in diagnostics}


def _extract_reasoning_text(message: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return reasoning text from vLLM's current or older field names."""

    for field in ("reasoning", "reasoning_content"):
        value = message.get(field)
        if value is not None:
            return str(value), field
    return None, None


def _extract_reported_reasoning_tokens(usage: dict[str, Any]) -> tuple[int | None, str | None]:
    """Read only explicitly reported OpenAI-style reasoning token details."""

    for detail_key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(detail_key)
        if not isinstance(details, dict):
            continue
        value = _optional_int(details.get("reasoning_tokens"))
        if value is not None:
            return value, f"usage.{detail_key}.reasoning_tokens"
    return None, None


def _optional_int(value: Any) -> int | None:
    """Return an int for real numeric fields, preserving missing/invalid values."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, *, default: int) -> int:
    """Return an int for token counters that may be missing from server output."""

    converted = _optional_int(value)
    return int(default) if converted is None else converted


def _safe_artifact_stem(value: str) -> str:
    """Return a filename-safe stem while keeping stage ids readable."""

    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    stem = stem.strip("_")
    return stem or "stage"


def _effective_generation_value(
    config: GenerationConfig,
    overrides: dict[str, Any],
    key: str,
) -> Any:
    """Return a generation field after applying the current stage override."""

    if key in overrides:
        return overrides[key]
    return getattr(config, key)


def _raw_response_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only the response metadata that is still useful for protocol debugging."""

    metadata = {
        key: raw.get(key)
        for key in ("id", "object", "created", "model", "service_tier", "system_fingerprint")
        if key in raw
    }
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        metadata["choice"] = {
            key: choice.get(key)
            for key in ("index", "finish_reason", "stop_reason")
            if key in choice
        }
    return metadata


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"Tool arguments must decode to a JSON object, got {type(parsed).__name__}.")
        return parsed
    raise TypeError(f"Unsupported tool arguments type: {type(value).__name__}.")


def _render_prompt_template(text: str, *, config: RewardAgentConfig) -> str:
    """Replace the small set of config placeholders used by package prompts."""

    replacements = {
        "{{progress_value_min_points}}": "0",
        "{{progress_value_max_points}}": str(int(config.progress.value_max_points)),
        "{{progress_default_jump_threshold_points}}": str(
            int(config.progress.default_jump_threshold_points)
        ),
    }
    rendered = text
    for needle, replacement in replacements.items():
        rendered = rendered.replace(needle, replacement)
    return rendered


def _render_trajectory_outcome_prompt(text: str, *, is_success: bool) -> str:
    """Render exactly one natural-language trajectory outcome placeholder."""

    if not isinstance(is_success, bool):
        raise TypeError("is_success must be a boolean when rendering the outcome prompt.")
    placeholder_count = text.count(_TRAJECTORY_OUTCOME_PLACEHOLDER)
    if placeholder_count != 1:
        raise ValueError(
            f"{_TRAJECTORY_OUTCOME_GROUND_TRUTH_PROMPT} must contain exactly one "
            f"{_TRAJECTORY_OUTCOME_PLACEHOLDER!r} placeholder, found {placeholder_count}."
        )
    outcome = "success" if is_success else "failure"
    rendered = text.replace(_TRAJECTORY_OUTCOME_PLACEHOLDER, outcome)
    if _TRAJECTORY_OUTCOME_PLACEHOLDER in rendered:
        raise ValueError("Trajectory outcome prompt rendering left an unresolved placeholder.")
    return rendered


def _has_successful_stage_completion(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> bool:
    """Return whether any complete_stage call in this response passed the gate."""

    for tool_call, result in zip(tool_calls, tool_results, strict=False):
        function = tool_call.get("function") or {}
        if str(function.get("name") or "") != "complete_stage":
            continue
        if tool_result_ok(result):
            return True
    return False
