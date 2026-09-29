"""Stateful reward-agent session and deterministic tool layer."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from .episode_source import CasePackEpisodeSource, RewardAgentEpisodeSource
from .progress import (
    ProgressCalibrationError,
    calibrate_progress_curve as apply_curve_calibration,
    edit_progress_segment as apply_segment_edit,
    export_corrected_progress_csv,
    render_progress_plot as write_progress_plot,
    summarize_progress as summarize_progress_values,
    validate_progress_curve,
)
from .config import RewardAgentConfig
from .reference_examples import (
    MaterializedReferenceExample,
    materialize_reference_examples,
)
from .schemas import (
    FinalResult,
    ObservationRecord,
    ObservationRevisionRecord,
    PendingMediaAttachment,
    ProgressCalibrationRecord,
    ProgressEditRecord,
    ProgressView,
    PropagationMode,
    ToolCallRecord,
    VisualAsset,
)
from .tool_protocol import (
    ToolResult,
    tool_blocked,
    tool_result_blocked_items,
    tool_result_data,
    tool_result_ok,
    tool_result_warnings,
    tool_success,
)
from .task_context import ResolvedTaskContext, TaskContextResolver
from .visual import (
    inspect_visual_frames,
    sample_frame_indices,
    write_progress_review_video,
    write_video_clip,
)


_PUBLIC_METADATA_KEYS = (
    "num_frames",
    "camera_key",
    "frame_shape_hwc",
    "agent_video_fps",
    "frame_indexing_rule",
)
_CONFIRMATION_BOUNDARY_SEARCH_FRAMES = 10
_ONE_FRAME_VIDEO_TO_TILE_WARNING = (
    "inspect_visual requested a one-frame video; the harness converted it to "
    "a tile image for vLLM compatibility."
)


class RewardAgentSession:
    """Mutable session state for one reward-agent case run.

    The session owns private filesystem details such as the raw video path and
    generated media paths. Tool results returned to the model intentionally
    redact those paths so the model can ground conclusions in visual evidence
    rather than directory names or episode filenames.
    """

    def __init__(
        self,
        case_dir: str | Path | None = None,
        *,
        episode_source: RewardAgentEpisodeSource | None = None,
        run_dir: str | Path | None = None,
        config: RewardAgentConfig | None = None,
        run_id: str | None = None,
        resume: bool = False,
        base_robot_instruction: str | None = None,
        task_context: ResolvedTaskContext | None = None,
    ) -> None:
        self.config = config or RewardAgentConfig.default()
        if episode_source is None:
            if case_dir is None:
                raise ValueError("case_dir is required when episode_source is not provided.")
            episode_source = CasePackEpisodeSource(
                case_dir,
                value_max_points=self.config.progress.value_max_points,
            )
        self.episode_source = episode_source
        self.case_dir = Path(getattr(episode_source, "source_root", case_dir or ".")).expanduser().resolve()
        self.case_id = str(episode_source.case_id)
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        self.run_dir = self._resolve_run_dir(run_dir=run_dir)
        self.visual_assets_dir = self.run_dir / "visual_assets"
        self.trace_path = self.run_dir / "trace.jsonl"
        self.state_path = self.run_dir / "session_state.json"
        self.manifest_path = self.run_dir / "result_manifest.json"

        if base_robot_instruction is None:
            base_robot_instruction = self.episode_source.read_task()
        base_instruction = str(base_robot_instruction).strip()
        if task_context is None:
            task_context = TaskContextResolver.from_config(self.config.task_context).resolve(
                base_instruction
            )
        elif task_context.base_instruction != base_instruction:
            raise ValueError(
                "Resolved task context base instruction does not match the session instruction: "
                f"{task_context.base_instruction!r} != {base_instruction!r}."
            )
        self.resolved_task_context = task_context
        self.base_robot_instruction = task_context.base_instruction
        self.robot_instruction = task_context.final_instruction
        self.task_context = task_context.to_manifest_dict()
        self.metadata = self.episode_source.read_metadata()
        self.is_success_ground_truth = (
            self.episode_source.read_is_success()
            if self.config.is_success_ground_truth.enabled
            else None
        )
        self.video_path = self.episode_source.video_path
        self._video_metadata_cache: dict[str, Any] | None = None
        self.progress_rows = self.episode_source.read_progress_rows()
        self.has_progress_proposal = self.progress_rows is not None
        if self.progress_rows is None:
            self.frame_indices = self._frame_indices_without_progress_proposal()
            self.proposal_progress_points: np.ndarray | None = None
            self.current_progress_points = np.zeros(len(self.frame_indices), dtype=np.int64)
            self.assigned_mask = np.zeros(len(self.frame_indices), dtype=bool)
            self.edit_reasons = ["" for _ in self.frame_indices]
        else:
            self.frame_indices = np.asarray([row.frame_index for row in self.progress_rows], dtype=np.int64)
            self.proposal_progress_points = np.asarray(
                [row.progress_points for row in self.progress_rows],
                dtype=np.int64,
            )
            self.current_progress_points = self.proposal_progress_points.copy()
            self.assigned_mask = np.ones(len(self.frame_indices), dtype=bool)
            self.edit_reasons = ["unchanged: progress proposal retained" for _ in self.frame_indices]

        self.visual_assets: list[VisualAsset] = []
        self.observations: list[ObservationRecord] = []
        self.observation_revision_history: list[ObservationRevisionRecord] = []
        self.tool_history: list[ToolCallRecord] = []
        self.progress_edit_history: list[ProgressEditRecord] = []
        self.progress_calibration_history: list[ProgressCalibrationRecord] = []
        self.exported_progress_path: Path | None = None
        self.progress_plot_path: Path | None = None
        self.progress_review_video_path: Path | None = None
        self.nonfatal_artifact_warnings: list[dict[str, Any]] = []
        self.final_summary: str | None = None
        self.external_review_need: int | None = None
        self.external_review_focus: str | None = None
        self.progress_revision = 0
        self.validated_progress_revision: int | None = None
        self.summary_progress_revision: int | None = None
        self.final_result: FinalResult | None = None

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.visual_assets_dir.mkdir(parents=True, exist_ok=True)
        self.reference_examples: tuple[MaterializedReferenceExample, ...] = ()
        if self.resolved_task_context.reference_examples:
            self.reference_examples = materialize_reference_examples(
                self.resolved_task_context.reference_examples,
                run_dir=self.run_dir,
                value_max_points=self.config.progress.value_max_points,
                video_fps=self.config.visual.agent_video_fps,
                video_codec=self.config.visual.video_codec,
            )
        if resume and self.state_path.is_file():
            self._load_state()
        self.append_trace(
            "session_resumed" if resume and self.state_path.is_file() else "session_initialized",
            {
                "case_id": self.case_id,
                "case_dir": str(self.case_dir),
                "run_dir": str(self.run_dir),
                "has_video": self.video_path is not None,
                "has_progress_proposal": self.has_progress_proposal,
                "num_progress_rows": len(self.frame_indices),
                "task_context_matched": bool(self.resolved_task_context.matched),
                "num_reference_examples": len(self.reference_examples),
            },
        )
        self._save_state()

    def read_robot_instruction(self) -> ToolResult:
        """Return the reward-agent instruction text without treating it as visual evidence."""

        result = tool_success(
            {
                "robot_instruction": self.robot_instruction,
                "usage_rule": (
                    "Use this instruction and any appended context only to identify relevant "
                    "visible entities, controls, and goal relations. It is not evidence that "
                    "any action or state transition occurred."
                ),
            }
        )
        self.append_trace("read_robot_instruction", result)
        return result

    def read_metadata(self) -> ToolResult:
        """Return public metadata from the case pack."""

        data = self._public_metadata()
        result = tool_success(data)
        self.append_trace("read_metadata", result)
        return result

    def read_progress(
        self,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        sample_every_n: int = 1,
        max_rows: int | None = None,
        view: ProgressView = "both",
        include_edit_spans: bool = False,
        trace_source: str = "tool",
    ) -> ToolResult:
        """Return candidate and current progress values through the tool interface."""

        view = _normalize_enum_string(view)
        if view not in {"proposal", "current", "both"}:
            raise ValueError(f"Unsupported progress view: {view!r}")
        effective_view = view
        message: str | None = None
        if not self.has_progress_proposal:
            if view == "both":
                effective_view = "current"
            elif view == "proposal":
                effective_view = "frame_axis"
                message = (
                    "No candidate progress proposal is available through read_progress. "
                    "Generate current progress from visual evidence instead of auditing existing values."
                )

        start = int(self.frame_indices[0] if start_frame is None else start_frame)
        end = int(self.frame_indices[-1] if end_frame is None else end_frame)
        positions = self._sample_progress_positions(
            start_frame=start,
            end_frame=end,
            sample_every_n=sample_every_n,
            max_rows=max_rows,
        )

        columns = ["frame_index"]
        if effective_view in {"proposal", "both"}:
            columns.append("proposal_progress_points")
        if effective_view in {"current", "both"}:
            columns.append("current_progress_points")

        rows: list[list[int | None]] = []
        for idx in positions:
            row = [int(self.frame_indices[idx])]
            if effective_view in {"proposal", "both"}:
                row.append(
                    None
                    if self.proposal_progress_points is None
                    else int(self.proposal_progress_points[idx])
                )
            if effective_view in {"current", "both"}:
                row.append(
                    int(self.current_progress_points[idx])
                    if bool(self.assigned_mask[idx])
                    else None
                )
            rows.append(row)

        data = {
            "has_progress_proposal": self.has_progress_proposal,
            "mode": self._progress_mode(),
            "curve_status": self._progress_curve_status(),
            "value_range": [0, int(self.config.progress.value_max_points)],
            "assigned_count": int(np.sum(self.assigned_mask)),
            "total_count": int(len(self.frame_indices)),
            "view": view,
            "effective_view": effective_view,
            "frame_range": [start, end],
            "sample_every_n": max(1, int(sample_every_n)),
            "max_rows": max_rows,
            "columns": columns,
            "num_rows": len(rows),
            "rows": rows,
            "source": str(trace_source),
        }
        if message is not None:
            data["message"] = message
        if include_edit_spans:
            data["edit_spans"] = self._progress_edit_spans(start_frame=start, end_frame=end)
        result = tool_success(data)
        self.append_trace("read_progress", result)
        return result

    def write_summary(
        self,
        *,
        summary: Any,
        external_review_need: Any,
        external_review_focus: Any = None,
    ) -> ToolResult:
        """Store the final narrative and external-review routing assessment."""

        clean_summary = _required_str_tool_arg("summary", summary, tool_name="write_summary")
        parsed_external_review_need = _required_external_review_need_tool_arg(
            "external_review_need",
            external_review_need,
            tool_name="write_summary",
        )
        clean_external_review_focus = _optional_external_review_focus_tool_arg(
            "external_review_focus",
            external_review_focus,
            external_review_need=parsed_external_review_need,
            tool_name="write_summary",
        )

        validation, _trace_validation = self._progress_validation_result(csv_path=None)
        if not tool_result_ok(validation):
            result = tool_blocked(
                blocked_items=tool_result_blocked_items(validation) or ["Progress validation failed."],
                next_required_actions=[
                    "Fix the progress curve with an appropriate progress-writing tool, "
                    "then call validate_progress again before write_summary."
                ],
                data={
                    "validation": validation,
                    "progress_revision": int(self.progress_revision),
                    "validated_progress_revision": self.validated_progress_revision,
                    "summary_progress_revision": self.summary_progress_revision,
                },
                warnings=tool_result_warnings(validation),
            )
            self.append_trace("write_summary", result)
            return result

        self.final_summary = clean_summary
        self.external_review_need = parsed_external_review_need
        self.external_review_focus = clean_external_review_focus
        # A successful summary is bound to the exact validated curve version it
        # describes. Later progress edits clear this binding.
        self.validated_progress_revision = int(self.progress_revision)
        self.summary_progress_revision = int(self.progress_revision)
        data = {
            "summary": self.final_summary,
            "external_review_need": int(self.external_review_need),
            "external_review_focus": self.external_review_focus,
            "progress_revision": int(self.progress_revision),
            "validated_progress_revision": int(self.validated_progress_revision),
            "summary_progress_revision": int(self.summary_progress_revision),
        }
        result = tool_success(data)
        self.append_trace("write_summary", result)
        self._save_state()
        return result

    def summarize_progress(
        self,
        *,
        top_k: int = 8,
        jump_threshold: int | None = None,
        trace_source: str = "tool",
    ) -> ToolResult:
        """Summarize current progress values and jump windows for inspection planning."""

        effective_jump_threshold = (
            self.config.progress.default_jump_threshold_points
            if jump_threshold is None
            else int(jump_threshold)
        )
        if not bool(np.all(self.assigned_mask)):
            data = {
                "has_progress_proposal": self.has_progress_proposal,
                "mode": self._progress_mode(),
                "curve_status": self._progress_curve_status(),
                "value_range": [0, int(self.config.progress.value_max_points)],
                "num_rows": int(len(self.frame_indices)),
                "start_frame": int(self.frame_indices[0]),
                "end_frame": int(self.frame_indices[-1]),
                "assigned_count": int(np.sum(self.assigned_mask)),
                "unassigned_count": int(np.sum(~self.assigned_mask)),
                "jump_threshold": effective_jump_threshold,
                "top_delta_windows": [],
                "jump_points": [],
                "source": str(trace_source),
            }
            result = tool_success(data)
            self.append_trace("summarize_progress", result)
            return result

        summary = summarize_progress_values(
            self.frame_indices,
            self.current_progress_points,
            top_k=top_k,
            jump_threshold=effective_jump_threshold,
        )
        data = summary.to_dict()
        data["has_progress_proposal"] = self.has_progress_proposal
        data["mode"] = self._progress_mode()
        data["curve_status"] = self._progress_curve_status()
        data["value_range"] = [0, int(self.config.progress.value_max_points)]
        data["jump_threshold"] = effective_jump_threshold
        data["assigned_count"] = int(np.sum(self.assigned_mask))
        data["total_count"] = int(len(self.frame_indices))
        data["source"] = str(trace_source)
        result = tool_success(data)
        self.append_trace("summarize_progress", result)
        return result

    def edit_progress_segments(self, *, segments: list[dict[str, Any]]) -> ToolResult:
        """Edit or generate progress by applying one or more local segment edits."""

        if not segments:
            raise ValueError("segments must be non-empty.")

        max_points = int(self.config.progress.value_max_points)
        normalized_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise TypeError(f"segments[{index}] must be an object.")

            start_frame = _required_int_tool_arg(
                f"segments[{index}].start_frame",
                segment.get("start_frame"),
                tool_name="edit_progress_segments",
            )
            end_frame = _required_int_tool_arg(
                f"segments[{index}].end_frame",
                segment.get("end_frame"),
                tool_name="edit_progress_segments",
            )
            if start_frame > end_frame:
                raise ValueError(
                    f"segments[{index}].start_frame must be <= end_frame, got {start_frame} > {end_frame}."
                )

            start_value = _required_progress_point_tool_arg(
                f"segments[{index}].start_progress_points",
                segment.get("start_progress_points"),
                tool_name="edit_progress_segments",
                value_max_points=max_points,
            )
            end_value = _required_progress_point_tool_arg(
                f"segments[{index}].end_progress_points",
                segment.get("end_progress_points"),
                tool_name="edit_progress_segments",
                value_max_points=max_points,
            )
            raw_propagation = segment.get("propagation")
            propagation = "none" if _is_blank_tool_arg(raw_propagation) else _normalize_enum_string(raw_propagation)
            if propagation != "none":
                raise ValueError(
                    f"segments[{index}].propagation must be 'none' for edit_progress_segments batches, "
                    f"got {propagation!r}."
                )
            reason = _required_str_tool_arg(
                f"segments[{index}].reason",
                segment.get("reason"),
                tool_name="edit_progress_segments",
            )
            confidence = _required_confidence_tool_arg(
                f"segments[{index}].confidence",
                segment.get("confidence"),
                tool_name="edit_progress_segments",
            )

            segment_mask = (self.frame_indices >= int(start_frame)) & (self.frame_indices <= int(end_frame))
            if len(np.flatnonzero(segment_mask)) == 0:
                raise ValueError(f"No progress rows fall inside frame range [{start_frame}, {end_frame}].")

            normalized_segment = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_progress_points": start_value,
                "end_progress_points": end_value,
                "propagation": propagation,
                "reason": reason,
                "confidence": confidence,
            }
            if normalized_segments:
                previous = normalized_segments[-1]
                previous_key = (int(previous["start_frame"]), int(previous["end_frame"]))
                current_key = (start_frame, end_frame)
                if current_key < previous_key:
                    raise ValueError(
                        "segments must be ordered by start_frame then end_frame; "
                        f"segments[{index - 1}]={previous_key} appears before segments[{index}]={current_key}."
                    )
                previous_end = int(previous["end_frame"])
                if start_frame < previous_end:
                    raise ValueError(
                        f"segments[{index}] overlaps segments[{index - 1}]; only one shared boundary frame "
                        "is allowed."
                    )
                if start_frame == previous_end and int(previous["end_progress_points"]) != start_value:
                    raise ValueError(
                        f"segments[{index}] shares boundary frame {start_frame} with segments[{index - 1}], "
                        "so the previous end_progress_points must equal the next start_progress_points."
                    )
            normalized_segments.append(normalized_segment)

        edit_records: list[ProgressEditRecord] = []
        for segment in normalized_segments:
            edit_id = f"edit_{len(self.progress_edit_history) + len(edit_records) + 1:04d}"
            edit_record = apply_segment_edit(
                frame_indices=self.frame_indices,
                current_progress_points=self.current_progress_points,
                edit_reasons=self.edit_reasons,
                edit_id=edit_id,
                start_frame=int(segment["start_frame"]),
                end_frame=int(segment["end_frame"]),
                start_progress_points=int(segment["start_progress_points"]),
                end_progress_points=int(segment["end_progress_points"]),
                propagation=segment["propagation"],
                reason=str(segment["reason"]),
                confidence=int(segment["confidence"]),
                value_max_points=max_points,
            )
            self._mark_progress_assigned(
                start_frame=int(segment["start_frame"]),
                end_frame=int(segment["end_frame"]),
                propagation=segment["propagation"],
            )
            edit_records.append(edit_record)

        self.progress_edit_history.extend(edit_records)
        self.progress_revision += 1
        self.validated_progress_revision = None
        self.summary_progress_revision = None
        self.exported_progress_path = None
        self.progress_plot_path = None
        self.progress_review_video_path = None
        self.nonfatal_artifact_warnings = []
        data = {
            "num_segments": len(edit_records),
            "segments": [record.to_dict() for record in edit_records],
            "value_range": [0, max_points],
            "progress_revision": int(self.progress_revision),
            "validated_progress_revision": self.validated_progress_revision,
            "summary_progress_revision": self.summary_progress_revision,
        }
        result = tool_success(data)
        self.append_trace("edit_progress_segments", result)
        self._save_state()
        return result

    def calibrate_progress_curve(
        self,
        *,
        anchor_a_frame: Any,
        anchor_a_progress_points: Any,
        anchor_b_frame: Any,
        anchor_b_progress_points: Any,
        reason: Any,
        confidence: Any,
    ) -> ToolResult:
        """Affine-calibrate assigned progress rows from two explicit anchors."""

        max_points = int(self.config.progress.value_max_points)
        calibration_id = f"calibration_{len(self.progress_calibration_history) + 1:04d}"
        try:
            parsed_anchor_a_frame = _required_int_tool_arg(
                "anchor_a_frame",
                anchor_a_frame,
                tool_name="calibrate_progress_curve",
            )
            parsed_anchor_b_frame = _required_int_tool_arg(
                "anchor_b_frame",
                anchor_b_frame,
                tool_name="calibrate_progress_curve",
            )
            parsed_anchor_a_progress = _required_progress_point_tool_arg(
                "anchor_a_progress_points",
                anchor_a_progress_points,
                tool_name="calibrate_progress_curve",
                value_max_points=max_points,
            )
            parsed_anchor_b_progress = _required_progress_point_tool_arg(
                "anchor_b_progress_points",
                anchor_b_progress_points,
                tool_name="calibrate_progress_curve",
                value_max_points=max_points,
            )
            parsed_reason = _required_str_tool_arg(
                "reason",
                reason,
                tool_name="calibrate_progress_curve",
            )
            parsed_confidence = _required_confidence_tool_arg(
                "confidence",
                confidence,
                tool_name="calibrate_progress_curve",
            )
            record = apply_curve_calibration(
                frame_indices=self.frame_indices,
                current_progress_points=self.current_progress_points,
                assigned_mask=self.assigned_mask,
                edit_reasons=self.edit_reasons,
                calibration_id=calibration_id,
                anchor_a_frame=parsed_anchor_a_frame,
                anchor_a_progress_points=parsed_anchor_a_progress,
                anchor_b_frame=parsed_anchor_b_frame,
                anchor_b_progress_points=parsed_anchor_b_progress,
                reason=parsed_reason,
                confidence=parsed_confidence,
                value_max_points=max_points,
            )
        except (ProgressCalibrationError, TypeError, ValueError) as exc:
            data = {
                "calibration_id": calibration_id,
                "value_range": [0, max_points],
                "assigned_count": int(np.sum(self.assigned_mask)),
                "total_count": int(len(self.frame_indices)),
                "progress_revision": int(self.progress_revision),
                "validated_progress_revision": self.validated_progress_revision,
                "summary_progress_revision": self.summary_progress_revision,
            }
            if isinstance(exc, ProgressCalibrationError):
                data.update(exc.data)
            result = tool_blocked(
                blocked_items=[str(exc)],
                next_required_actions=[
                    "Choose assigned anchor frames with distinct current progress values, "
                    "adjust local progress segments first if needed, then retry calibrate_progress_curve."
                ],
                data=data,
            )
            self.append_trace("calibrate_progress_curve", result)
            return result

        self.progress_calibration_history.append(record)
        self.progress_revision += 1
        self.validated_progress_revision = None
        self.summary_progress_revision = None
        self.exported_progress_path = None
        self.progress_plot_path = None
        self.progress_review_video_path = None
        self.nonfatal_artifact_warnings = []
        data = {
            "calibration": record.to_dict(),
            "value_range": [0, max_points],
            "assigned_count": int(np.sum(self.assigned_mask)),
            "total_count": int(len(self.frame_indices)),
            "progress_revision": int(self.progress_revision),
            "validated_progress_revision": self.validated_progress_revision,
            "summary_progress_revision": self.summary_progress_revision,
        }
        result = tool_success(data)
        self.append_trace("calibrate_progress_curve", result)
        self._save_state()
        return result

    def export_corrected_progress(self, *, filename: str = "corrected_progress.csv") -> ToolResult:
        """Export the current corrected or generated progress curve."""

        path = self.run_dir / filename
        self.exported_progress_path = export_corrected_progress_csv(
            path=path,
            frame_indices=self.frame_indices,
            proposal_progress_points=self.proposal_progress_points,
            current_progress_points=self.current_progress_points,
            assigned_mask=self.assigned_mask,
            edit_reasons=self.edit_reasons,
            value_max_points=self.config.progress.value_max_points,
        )
        data = {
            "exported": True,
            "has_progress_proposal": self.has_progress_proposal,
            "mode": self._progress_mode(),
            "num_rows": int(len(self.frame_indices)),
            "value_range": [0, int(self.config.progress.value_max_points)],
        }
        result = tool_success(data)
        self.append_trace(
            "export_corrected_progress",
            tool_success({**data, "path": str(self.exported_progress_path)}),
        )
        self._save_state()
        return result

    def _progress_validation_result(self, *, csv_path: Path | None) -> tuple[ToolResult, ToolResult]:
        """Build exposed and trace validation envelopes for the current progress state."""

        result = validate_progress_curve(
            frame_indices=self.frame_indices,
            proposal_progress_points=self.proposal_progress_points,
            current_progress_points=self.current_progress_points,
            assigned_mask=self.assigned_mask,
            edit_reasons=self.edit_reasons,
            value_max_points=self.config.progress.value_max_points,
            is_success_ground_truth=self.is_success_ground_truth,
            csv_path=csv_path,
        )
        data = result.to_dict()
        errors = [str(item) for item in data.pop("errors", [])]
        warnings = [str(item) for item in data.pop("warnings", [])]
        data.pop("ok", None)
        data["has_progress_proposal"] = self.has_progress_proposal
        data["mode"] = self._progress_mode()
        data["value_range"] = [0, int(self.config.progress.value_max_points)]
        data["progress_revision"] = int(self.progress_revision)
        data["validated_progress_revision"] = self.validated_progress_revision
        trace_data = dict(data)
        if csv_path is not None:
            trace_data["csv_path"] = str(csv_path)
        # Keep the internal trace path for debugging, but do not expose a path
        # field to the agent. Mid-stage validation is intentionally memory-only.
        data.pop("csv_path", None)
        if errors:
            envelope = tool_blocked(
                blocked_items=errors,
                next_required_actions=["Fix the progress curve and validate again."],
                data=data,
                warnings=warnings,
            )
            trace_envelope = tool_blocked(
                blocked_items=errors,
                next_required_actions=["Fix the progress curve and validate again."],
                data=trace_data,
                warnings=warnings,
            )
        else:
            envelope = tool_success(data, warnings=warnings)
            trace_envelope = tool_success(trace_data, warnings=warnings)
        return envelope, trace_envelope

    def validate_progress(self) -> ToolResult:
        """Validate the current in-memory progress curve without consulting exported artifacts."""

        envelope, trace_envelope = self._progress_validation_result(csv_path=None)
        if tool_result_ok(envelope):
            self.validated_progress_revision = int(self.progress_revision)
            tool_result_data(envelope)["validated_progress_revision"] = int(self.validated_progress_revision)
            tool_result_data(trace_envelope)["validated_progress_revision"] = int(self.validated_progress_revision)
        self.append_trace("validate_progress", trace_envelope)
        if tool_result_ok(envelope):
            self._save_state()
        return envelope

    def _write_progress_plot(self) -> Path:
        """Render the current progress curve as a hidden human-review artifact."""

        return write_progress_plot(
            path=self.run_dir / "corrected_progress_plot.png",
            frame_indices=self.frame_indices,
            proposal_progress_points=self.proposal_progress_points,
            current_progress_points=self.current_progress_points,
            value_max_points=self.config.progress.value_max_points,
            assigned_mask=self.assigned_mask,
            title="Reward Agent Progress Curve",
        )

    def _write_progress_review_video(self) -> Path | None:
        """Render a human-only video aligning episode frames with the full curve."""

        visual_frame_indices = self._review_visual_frame_indices()
        frames = self.episode_source.read_selected_frames(visual_frame_indices)
        return write_progress_review_video(
            frames=frames,
            visual_frame_indices=visual_frame_indices,
            output_path=self.run_dir / "corrected_progress_review.mp4",
            progress_frame_indices=self.frame_indices,
            proposal_progress_points=self.proposal_progress_points,
            current_progress_points=self.current_progress_points,
            assigned_mask=self.assigned_mask,
            fps=self._review_video_fps(),
            codec=self.config.visual.video_codec,
            title="Reward Agent Progress Curve",
            value_max_points=self.config.progress.value_max_points,
        )

    def _write_optional_progress_review_video(self, *, max_attempts: int = 2) -> Path | None:
        """Best-effort review video rendering with final warnings for batch summaries."""

        artifact = "corrected_progress_review_video"
        attempts = max(1, int(max_attempts))
        last_error: Exception | None = None
        for attempt_index in range(1, attempts + 1):
            try:
                return self._write_progress_review_video()
            except Exception as exc:  # noqa: BLE001 - this artifact is intentionally nonfatal.
                last_error = exc
                failure = {
                    "artifact": artifact,
                    "attempt": int(attempt_index),
                    "attempts": int(attempts),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                self.append_trace(
                    "nonfatal_artifact_retry"
                    if attempt_index < attempts
                    else "nonfatal_artifact_warning",
                    {
                        **failure,
                        "status": "retrying" if attempt_index < attempts else "failed_after_retry",
                    },
                )

        if last_error is None:
            return None
        warning = {
            "artifact": artifact,
            "status": "failed_after_retry",
            "attempts": int(attempts),
            "error_type": type(last_error).__name__,
            "message": str(last_error),
        }
        self.nonfatal_artifact_warnings.append(warning)
        return None

    def _review_video_fps(self) -> float:
        """Resolve the human-review playback rate without changing agent-facing FPS."""

        configured_fps = self.config.visual.review_video_fps
        if configured_fps is not None:
            return float(configured_fps)
        source_fps = _positive_float_or_none(getattr(self.episode_source, "source_fps", None))
        if source_fps is not None:
            return source_fps
        inferred_fps = _positive_float_or_none(self._infer_video_metadata().get("fps"))
        if inferred_fps is not None:
            return inferred_fps
        return 8.0

    def _review_visual_frame_indices(self) -> list[int]:
        """Return the full episode frame axis used by the human-review video."""

        num_frames = (
            _positive_int_or_none(self.metadata.get("num_frames"))
            or _positive_int_or_none(getattr(self.episode_source, "num_frames", None))
            or _positive_int_or_none(self._infer_video_metadata().get("num_frames"))
        )
        if num_frames is None:
            if len(self.frame_indices) == 0:
                raise ValueError("Cannot infer review video frame axis without progress or episode frames.")
            num_frames = int(self.frame_indices[-1]) + 1
        if num_frames <= 0:
            raise ValueError(f"Review video frame count must be positive, got {num_frames}.")
        return list(range(int(num_frames)))

    def inspect_visual(
        self,
        *,
        start_frame: int | str,
        end_frame: int | str,
        sample_every_n: int | str = 1,
        mode: str | None = None,
        crop: dict[str, Any] | None = None,
        reason: str,
        stage_id: str | None = None,
        provided_arguments: set[str] | None = None,
    ) -> ToolResult:
        """Generate bounded visual evidence without exposing private paths."""

        # `start_frame`, `end_frame`, and `reason` define the actual visual
        # question. Empty strings here should be rejected explicitly instead of
        # silently expanding to a broad default inspection.
        start = _required_int_tool_arg("start_frame", start_frame, tool_name="inspect_visual")
        end = _required_int_tool_arg("end_frame", end_frame, tool_name="inspect_visual")
        cleaned_reason = _required_str_tool_arg("reason", reason, tool_name="inspect_visual")

        valid_start = int(self.frame_indices[0])
        valid_end = int(self.frame_indices[-1])
        if start < valid_start or end > valid_end:
            result = tool_blocked(
                blocked_items=[
                    f"inspect_visual frame range [{start}, {end}] is outside valid video range "
                    f"[{valid_start}, {valid_end}]."
                ],
                next_required_actions=[
                    f"Retry inspect_visual with start_frame/end_frame clamped to "
                    f"[{valid_start}, {valid_end}]."
                ],
                data={
                    "valid_frame_range": [valid_start, valid_end],
                    "requested_frame_range": [start, end],
                },
            )
            self.append_trace("inspect_visual", result)
            return result

        policy = _inspect_visual_policy(self._stage_tool_config(stage_id).get("inspect_visual"))
        provided = set(provided_arguments or set())
        policy_blockers: list[str] = []
        policy_next_actions: list[str] = []

        # Stage-specific schemas hide fixed fields from the model. Rejecting
        # hidden fields at runtime enforces the policy even when the provider
        # forwards arguments outside the current schema.
        if policy["mode"] != "model" and "mode" in provided:
            policy_blockers.append(
                "inspect_visual.mode is fixed by the current stage tool_config and is not "
                "available in this stage's tool schema."
            )
        if policy["sample_every_n"] != "model" and "sample_every_n" in provided:
            policy_blockers.append(
                "inspect_visual.sample_every_n is fixed by the current stage tool_config and "
                "is not available in this stage's tool schema."
            )
        if policy["crop"] == "disabled" and "crop" in provided:
            policy_blockers.append(
                "inspect_visual.crop is disabled by the current stage tool_config and is not "
                "available in this stage's tool schema."
            )
        if policy_blockers:
            policy_next_actions.append(
                "Retry inspect_visual using exactly the arguments exposed in the current "
                "stage's tool schema; omit fields fixed by stage tool_config."
            )
            result = tool_blocked(
                blocked_items=policy_blockers,
                next_required_actions=policy_next_actions,
                data={
                    "stage_id": str(stage_id or ""),
                    "tool": "inspect_visual",
                    "provided_arguments": sorted(provided),
                    "policy": _json_ready(policy),
                },
            )
            self.append_trace("inspect_visual", result)
            return result

        # Optional knobs may arrive as empty strings from some model/tool stacks.
        # Treat those as omitted, then apply the current stage policy.
        if policy["sample_every_n"] == "model":
            step = _optional_int_tool_arg("sample_every_n", sample_every_n, default=1, tool_name="inspect_visual")
        else:
            step = int(policy["sample_every_n"])
        frame_indices = sample_frame_indices(
            start_frame=start,
            end_frame=end,
            sample_every_n=step,
        )
        frame_budget = max(1, int(self.config.visual.max_attached_frames_per_call))
        requested_count = len(frame_indices)
        if requested_count > frame_budget:
            raise ValueError(
                "FrameLimitExceeded: inspect_visual requested "
                f"{requested_count} frames after applying sample_every_n={step}, "
                f"but visual.max_attached_frames_per_call is {frame_budget}. "
                "Reduce the frame range, increase sample_every_n, or split the request."
            )
        if policy["mode"] == "model":
            media_mode = _optional_enum_tool_arg(
                "mode",
                mode,
                allowed={"video", "tile"},
                default=self.config.visual.media_mode,
                tool_name="inspect_visual",
            )
        else:
            media_mode = str(policy["mode"])
        inspect_warnings: list[str] = []
        # vLLM video processors expect temporal evidence. A single selected
        # frame has image semantics, so the harness makes that representation
        # explicit while preserving the requested frame coverage.
        if len(frame_indices) == 1 and media_mode == "video":
            media_mode = "tile"
            inspect_warnings.append(_ONE_FRAME_VIDEO_TO_TILE_WARNING)
        normalized_crop = (
            None
            if policy["crop"] == "disabled"
            else _optional_crop_tool_arg("crop", crop, tool_name="inspect_visual")
        )
        show_frame_label, show_progress_label = self._external_label_bar_options(media_mode)
        if _is_observation_stage(stage_id):
            show_progress_label = False
        progress_label_revision = int(self.progress_revision) if show_progress_label else None
        existing_asset = self._find_duplicate_visual_asset(
            stage_id=stage_id,
            frame_indices=frame_indices,
            media_mode=media_mode,
            crop=normalized_crop,
            progress_label_revision=progress_label_revision,
        )
        if existing_asset is not None:
            result = tool_blocked(
                blocked_items=[
                    "Duplicate inspect_visual request: this visual evidence was already generated "
                    "in the current stage."
                ],
                next_required_actions=[
                    "Use existing_asset.asset_id with an appropriate non-inspect tool for the "
                    "current stage, or request a materially different visual range/configuration."
                ],
                data={
                    "tool": "inspect_visual",
                    "stage_id": str(stage_id or ""),
                    "duplicate_of_asset_id": existing_asset.asset_id,
                    "existing_asset": {
                        "asset_id": existing_asset.asset_id,
                        "kind": existing_asset.kind,
                        "frame_range": [
                            int(existing_asset.frame_range[0]),
                            int(existing_asset.frame_range[1]),
                        ],
                        "sample_every_n": int(existing_asset.sample_every_n),
                        "attached_frame_count": int(existing_asset.attached_frame_count),
                    },
                    "requested_frame_range": [int(start), int(end)],
                    "requested_frame_indices": [int(frame) for frame in frame_indices],
                    "requested_mode": media_mode,
                    "requested_crop": dict(normalized_crop) if normalized_crop is not None else None,
                    "progress_label_revision": progress_label_revision,
                },
                warnings=inspect_warnings,
            )
            self.append_trace("inspect_visual", result)
            return result
        progress_labels = self._progress_labels_for_frames(frame_indices) if show_progress_label else None
        frames = self.episode_source.read_selected_frames(frame_indices)
        asset = inspect_visual_frames(
            frames=frames,
            frame_indices=frame_indices,
            output_dir=self.visual_assets_dir,
            start_frame=start,
            end_frame=end,
            sample_every_n=step,
            mode=media_mode,
            reason=cleaned_reason,
            progress_labels=progress_labels,
            show_frame_label=show_frame_label,
            crop=normalized_crop,
            tile_columns=self.config.visual.tile_columns,
            tile_max_width=self.config.visual.tile_max_width,
            video_codec=self.config.visual.video_codec,
            video_fps=float(self.config.visual.agent_video_fps),
        )
        asset = replace(
            asset,
            created_stage_id=str(stage_id) if stage_id else None,
            progress_label_revision=progress_label_revision,
        )
        existing_index = next(
            (index for index, item in enumerate(self.visual_assets) if item.asset_id == asset.asset_id),
            None,
        )
        if existing_index is None:
            self.visual_assets.append(asset)
        else:
            self.visual_assets[existing_index] = asset
        data = asset.to_agent_dict()
        result = tool_success(data, warnings=inspect_warnings)
        self.append_trace(
            "inspect_visual",
            tool_success(asset.to_dict(include_path=True), warnings=inspect_warnings),
        )
        self._save_state()
        return result

    def create_full_video_asset(self) -> VisualAsset:
        """Register a full-trajectory video attachment, transcoding if needed.

        The bootstrap stage should see the full frame sequence without model-
        requested sampling. We still route the media through the harness so the
        attachment has an `asset_id`, a traceable output path, and a codec known
        to work with the OpenAI-compatible video input path.
        """

        metadata = self._public_metadata()
        num_frames = _positive_int_or_none(metadata.get("num_frames"))
        if num_frames is None:
            num_frames = int(self.frame_indices[-1]) + 1
        frame_budget = max(1, int(self.config.visual.max_attached_frames_per_call))
        if num_frames > frame_budget:
            raise ValueError(
                "FrameLimitExceeded: full bootstrap video has "
                f"{num_frames} frames, but visual.max_attached_frames_per_call is {frame_budget}. "
                "Use focused or segmented inspect_visual calls, or raise the configured limit."
            )

        existing = self._find_visual_asset("full_video")
        if existing is not None:
            return existing

        frame_indices = list(range(num_frames))
        frames = self.episode_source.read_selected_frames(frame_indices)
        output_path = self.visual_assets_dir / "full_video.mp4"
        show_frame_label, _show_progress_label = self._external_label_bar_options("video")
        write_video_clip(
            frames=frames,
            frame_indices=frame_indices,
            output_path=output_path,
            fps=float(self.config.visual.agent_video_fps),
            codec=self.config.visual.video_codec,
            progress_labels=None,
            show_frame_label=show_frame_label,
        )
        asset = VisualAsset(
            asset_id="full_video",
            kind="video_clip",
            path=output_path,
            frame_range=(0, num_frames - 1),
            frame_indices=frame_indices,
            sample_every_n=1,
            attached_frame_count=num_frames,
            reason="Initial full-trajectory video attached for bootstrap observation.",
            created_stage_id="bootstrap_observations",
        )
        self.visual_assets.append(asset)
        self.append_trace("full_video_asset_created", asset.to_dict(include_path=True))
        self._save_state()
        return asset

    def full_video_attachment(self) -> PendingMediaAttachment:
        """Return a pending media attachment for the registered full video."""

        asset = self.create_full_video_asset()
        return PendingMediaAttachment(
            asset_id=asset.asset_id,
            kind=asset.kind,
            path=asset.path,
            frame_indices=list(asset.frame_indices),
            frame_range=asset.frame_range,
            sample_every_n=asset.sample_every_n,
        )

    def add_observations(
        self,
        *,
        observations: list[dict[str, Any]],
        stage_id: str | None = None,
    ) -> ToolResult:
        """Record one or more interval observations tied to visual asset ids."""

        if not observations:
            raise ValueError("observations must be non-empty.")
        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}

        # Validate the full batch before mutating session state so a malformed
        # late item cannot leave earlier observations partially appended.
        records: list[ObservationRecord] = []
        blockers: list[str] = []
        next_actions: list[str] = []
        blocked_indices: list[int] = []
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                raise TypeError(f"observations[{index}] must be an object.")

            asset_ids = observation.get("asset_ids")
            start_frame = observation.get("start_frame")
            end_frame = observation.get("end_frame")
            text = observation.get("text")
            confidence = observation.get("confidence")

            if not asset_ids:
                raise ValueError(f"observations[{index}].asset_ids must be non-empty.")
            start = _required_int_tool_arg(
                f"observations[{index}].start_frame",
                start_frame,
                tool_name="add_observations",
            )
            end = _required_int_tool_arg(
                f"observations[{index}].end_frame",
                end_frame,
                tool_name="add_observations",
            )
            if start > end:
                raise ValueError(
                    f"observations[{index}].start_frame must be <= end_frame, got {start} > {end}."
                )
            if not str(text or "").strip():
                raise ValueError(f"observations[{index}].text must be non-empty.")
            parsed_confidence = _required_confidence_tool_arg(
                f"observations[{index}].confidence",
                confidence,
                tool_name="add_observations",
            )

            normalized_asset_ids = [str(asset_id) for asset_id in asset_ids]
            missing = [asset_id for asset_id in normalized_asset_ids if asset_id not in asset_by_id]
            if missing:
                raise KeyError(f"observations[{index}] has unknown visual asset ids: {missing}")

            # Keep the model responsible for event boundaries, but require those
            # boundaries to be covered by the referenced visual attachments.
            coverage_blocker = self._observation_asset_coverage_blocker(
                tool_name=f"add_observations.observations[{index}]",
                retry_tool_name="add_observations",
                asset_ids=normalized_asset_ids,
                start=start,
                end=end,
                asset_by_id=asset_by_id,
            )
            if coverage_blocker is not None:
                blocked_item, next_action = coverage_blocker
                blockers.append(blocked_item)
                next_actions.append(next_action)
                blocked_indices.append(index)
                continue

            records.append(
                ObservationRecord(
                    observation_id=f"obs_{len(self.observations) + len(records) + 1:04d}",
                    asset_ids=normalized_asset_ids,
                    start_frame=start,
                    end_frame=end,
                    text=str(text).strip(),
                    confidence=parsed_confidence,
                    status="draft",
                    created_stage_id=str(stage_id) if stage_id else None,
                    updated_stage_id=str(stage_id) if stage_id else None,
                )
            )

        if blockers:
            data = {
                "num_added": 0,
                "blocked_observation_indices": blocked_indices,
                "coverage": self._observation_coverage_agent_view(
                    self.observations,
                    stage_id=stage_id,
                    include_status=False,
                ),
            }
            result = tool_blocked(
                blocked_items=blockers,
                next_required_actions=_dedupe_preserve_order(next_actions),
                data=data,
            )
            self.append_trace("add_observations", result)
            return result

        self.observations.extend(records)
        data = {
            "num_added": len(records),
            "observations": [
                self._observation_agent_view(record, stage_id=stage_id, include_status=False)
                for record in records
            ],
            "coverage": self._observation_coverage_agent_view(
                self.observations,
                stage_id=stage_id,
                include_status=False,
            ),
        }
        result = tool_success(data)
        self.append_trace("add_observations", result)
        self._save_state()
        return result

    def read_observations(self, *, stage_id: str | None = None) -> ToolResult:
        """Return all observations recorded so far."""

        observations = self._observations_visible_to_stage(stage_id)
        data = {
            "observations": [
                self._observation_agent_view(observation, stage_id=stage_id)
                for observation in observations
            ],
            "coverage": self._observation_coverage_agent_view(observations, stage_id=stage_id),
        }
        if str(stage_id or "").strip() == "progress":
            data["observation_filter"] = "confirmed_only"
        result = tool_success(data)
        self.append_trace("read_observations", result)
        return result

    def confirm_observations(
        self,
        *,
        confirmations: list[dict[str, Any]],
        stage_id: str | None = None,
    ) -> ToolResult:
        """Mark one or more current observations confirmed with cited evidence."""

        if not confirmations:
            raise ValueError("confirmations must be non-empty.")
        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}
        observation_by_id = {observation.observation_id: observation for observation in self.observations}

        updates: dict[str, ObservationRecord] = {}
        blockers: list[str] = []
        next_actions: list[str] = []
        invalid_ids: list[str] = []
        for index, confirmation in enumerate(confirmations):
            if not isinstance(confirmation, dict):
                raise TypeError(f"confirmations[{index}] must be an object.")
            observation_id = _required_str_tool_arg(
                f"confirmations[{index}].observation_id",
                confirmation.get("observation_id"),
                tool_name="confirm_observations",
            )
            if observation_id not in observation_by_id:
                blockers.append(f"confirmations[{index}] has unknown observation_id: {observation_id}")
                invalid_ids.append(observation_id)
                next_actions.append("Use read_observations to get current observation ids before confirming.")
                continue
            target_observation = observation_by_id[observation_id]
            if target_observation.status == "inactive":
                blockers.append(
                    f"confirmations[{index}] observation_id {observation_id} is inactive and cannot be confirmed."
                )
                invalid_ids.append(observation_id)
                next_actions.append(
                    "Use read_observations to get current active observation ids before confirming."
                )
                continue
            if observation_id in updates:
                blockers.append(f"confirmations[{index}] repeats observation_id {observation_id}.")
                invalid_ids.append(observation_id)
                next_actions.append(f"Submit at most one confirmation entry for {observation_id} in a batch.")
                continue
            asset_ids = confirmation.get("asset_ids")
            if not asset_ids:
                blockers.append(f"confirmations[{index}].asset_ids must be non-empty.")
                invalid_ids.append(observation_id)
                next_actions.append(self._challenge_observation_next_action(observation_by_id[observation_id]))
                continue
            normalized_asset_ids = [str(asset_id) for asset_id in asset_ids]
            missing = [asset_id for asset_id in normalized_asset_ids if asset_id not in asset_by_id]
            if missing:
                blockers.append(f"confirmations[{index}] has unknown visual asset ids: {missing}")
                invalid_ids.append(observation_id)
                next_actions.append(self._challenge_observation_next_action(observation_by_id[observation_id]))
                continue
            reason = _required_str_tool_arg(
                f"confirmations[{index}].reason",
                confirmation.get("reason"),
                tool_name="confirm_observations",
            )
            current = updates.get(observation_id, observation_by_id[observation_id])
            updates[observation_id] = replace(
                current,
                status="confirmed",
                updated_stage_id=str(stage_id) if stage_id else None,
                confirmed_stage_id=str(stage_id) if stage_id else None,
                confirmation_asset_ids=normalized_asset_ids,
                confirmation_reason=reason,
            )
            evidence_blockers = self._confirmation_evidence_blockers(updates[observation_id])
            if evidence_blockers:
                blockers.extend(evidence_blockers)
                invalid_ids.append(observation_id)
                next_actions.append(self._challenge_observation_next_action(updates[observation_id]))

        if blockers:
            data = {
                "confirmed_observation_ids": [],
                "invalid_observation_ids": _dedupe_preserve_order(invalid_ids),
                **self._observation_status_summary(),
            }
            result = tool_blocked(
                blocked_items=blockers,
                next_required_actions=_dedupe_preserve_order(next_actions),
                data=data,
            )
            self.append_trace("confirm_observations", result)
            return result

        for index, observation in enumerate(self.observations):
            if observation.observation_id in updates:
                self.observations[index] = updates[observation.observation_id]

        data = {
            "confirmed_observation_ids": sorted(updates),
            **self._observation_status_summary(),
        }
        result = tool_success(data)
        self.append_trace("confirm_observations", result)
        self._save_state()
        return result

    def revise_observations(
        self,
        *,
        revisions: list[dict[str, Any]],
        stage_id: str | None = None,
    ) -> ToolResult:
        """Revise one or more observations in place while recording version history."""

        if not revisions:
            raise ValueError("revisions must be non-empty.")

        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}
        observation_by_id = {
            observation.observation_id: (position, observation)
            for position, observation in enumerate(self.observations)
        }
        updates: dict[str, ObservationRecord] = {}
        revision_records: list[ObservationRevisionRecord] = []
        blockers: list[str] = []
        next_actions: list[str] = []
        blocked_observation_ids: list[str] = []
        seen_observation_ids: set[str] = set()

        for index, revision in enumerate(revisions):
            if not isinstance(revision, dict):
                raise TypeError(f"revisions[{index}] must be an object.")

            observation_id = _required_str_tool_arg(
                f"revisions[{index}].observation_id",
                revision.get("observation_id"),
                tool_name="revise_observations",
            )
            if observation_id in seen_observation_ids:
                raise ValueError(f"revisions[{index}] repeats observation_id {observation_id}.")
            seen_observation_ids.add(observation_id)
            if observation_id not in observation_by_id:
                raise KeyError(f"Unknown observation_id: {observation_id}")
            _, existing_observation = observation_by_id[observation_id]
            if existing_observation.status == "inactive":
                raise ValueError(
                    f"revisions[{index}].observation_id {observation_id} is inactive and cannot be revised."
                )

            asset_ids = revision.get("asset_ids")
            if not asset_ids:
                raise ValueError(f"revisions[{index}].asset_ids must be non-empty.")
            normalized_asset_ids = [str(asset_id) for asset_id in asset_ids]
            missing = [asset_id for asset_id in normalized_asset_ids if asset_id not in asset_by_id]
            if missing:
                raise KeyError(f"revisions[{index}] has unknown visual asset ids: {missing}")

            start = _required_int_tool_arg(
                f"revisions[{index}].start_frame",
                revision.get("start_frame"),
                tool_name="revise_observations",
            )
            end = _required_int_tool_arg(
                f"revisions[{index}].end_frame",
                revision.get("end_frame"),
                tool_name="revise_observations",
            )
            if start > end:
                raise ValueError(f"revisions[{index}].start_frame must be <= end_frame, got {start} > {end}.")

            clean_text = _required_str_tool_arg(
                f"revisions[{index}].text",
                revision.get("text"),
                tool_name="revise_observations",
            )
            revision_reason = _required_str_tool_arg(
                f"revisions[{index}].reason",
                revision.get("reason"),
                tool_name="revise_observations",
            )
            confidence = _required_confidence_tool_arg(
                f"revisions[{index}].confidence",
                revision.get("confidence"),
                tool_name="revise_observations",
            )
            coverage_blocker = self._observation_asset_coverage_blocker(
                tool_name=f"revise_observations.revisions[{index}]",
                retry_tool_name="revise_observations",
                asset_ids=normalized_asset_ids,
                start=start,
                end=end,
                asset_by_id=asset_by_id,
            )
            if coverage_blocker is not None:
                blocked_item, next_action = coverage_blocker
                blockers.append(blocked_item)
                next_actions.append(next_action)
                blocked_observation_ids.append(observation_id)
                continue

            _, old_observation = observation_by_id[observation_id]
            new_observation = replace(
                old_observation,
                asset_ids=normalized_asset_ids,
                start_frame=start,
                end_frame=end,
                text=clean_text,
                confidence=confidence,
                status="draft",
                updated_stage_id=str(stage_id) if stage_id else None,
                confirmed_stage_id=None,
                confirmation_asset_ids=[],
                confirmation_reason="",
                revision_count=int(old_observation.revision_count) + 1,
            )
            updates[observation_id] = new_observation
            revision_records.append(
                ObservationRevisionRecord(
                    revision_id=f"obsrev_{len(self.observation_revision_history) + len(revision_records) + 1:04d}",
                    observation_id=observation_id,
                    stage_id=str(stage_id) if stage_id else None,
                    reason=revision_reason,
                    old_observation=old_observation.to_dict(),
                    new_observation=new_observation.to_dict(),
                )
            )

        if blockers:
            data = {
                "revised_observation_ids": [],
                "blocked_observation_ids": _dedupe_preserve_order(blocked_observation_ids),
                **self._observation_status_summary(),
                "coverage": self._observation_coverage_agent_view(self.observations, stage_id=stage_id),
            }
            result = tool_blocked(
                blocked_items=blockers,
                next_required_actions=_dedupe_preserve_order(next_actions),
                data=data,
            )
            self.append_trace("revise_observations", result)
            return result

        for position, observation in enumerate(self.observations):
            if observation.observation_id in updates:
                self.observations[position] = updates[observation.observation_id]
        self.observation_revision_history.extend(revision_records)

        revised_observations = [updates[str(item["observation_id"]).strip()] for item in revisions]
        data = {
            "revised_observation_ids": [observation.observation_id for observation in revised_observations],
            "observations": [
                self._observation_agent_view(observation, stage_id=stage_id)
                for observation in revised_observations
            ],
            **self._observation_status_summary(),
        }
        result = tool_success(data)
        self.append_trace(
            "revise_observations",
            tool_success({**data, "revisions": [record.to_dict() for record in revision_records]}),
        )
        self._save_state()
        return result

    def merge_observations(
        self,
        *,
        observation_ids: list[str],
        asset_ids: list[str],
        merged_text: str,
        confidence: int,
        reason: str,
        stage_id: str | None = None,
    ) -> ToolResult:
        """Merge active observations whose closed-frame union is continuous."""

        normalized_stage_id = str(stage_id or "").strip()
        if not isinstance(observation_ids, list) or len(observation_ids) < 2:
            raise ValueError("merge_observations.observation_ids must contain at least two observation ids.")
        normalized_observation_ids = [
            str(observation_id).strip()
            for observation_id in observation_ids
            if str(observation_id).strip()
        ]
        if len(normalized_observation_ids) != len(observation_ids):
            raise ValueError("merge_observations.observation_ids must contain only non-empty strings.")
        if len(set(normalized_observation_ids)) != len(normalized_observation_ids):
            raise ValueError("merge_observations.observation_ids must not repeat ids.")

        normalized_asset_ids = _required_asset_id_list(
            "asset_ids",
            asset_ids,
            tool_name="merge_observations",
        )
        clean_text = _required_str_tool_arg("merged_text", merged_text, tool_name="merge_observations")
        clean_reason = _required_str_tool_arg("reason", reason, tool_name="merge_observations")
        parsed_confidence = _required_confidence_tool_arg(
            "confidence",
            confidence,
            tool_name="merge_observations",
        )

        active_observations = self._active_observations_sorted()
        active_by_id = {
            observation.observation_id: (position, observation)
            for position, observation in enumerate(active_observations)
        }
        missing_or_inactive = [
            observation_id
            for observation_id in normalized_observation_ids
            if observation_id not in active_by_id
        ]
        if missing_or_inactive:
            raise KeyError(
                "merge_observations.observation_ids must refer to current active observations; "
                f"not active or unknown: {missing_or_inactive}."
            )

        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}
        missing_assets = [asset_id for asset_id in normalized_asset_ids if asset_id not in asset_by_id]
        if missing_assets:
            raise KeyError(f"merge_observations has unknown visual asset ids: {missing_assets}")

        selected_observations = [active_by_id[observation_id][1] for observation_id in normalized_observation_ids]
        merged_start = min(int(observation.start_frame) for observation in selected_observations)
        merged_end = max(int(observation.end_frame) for observation in selected_observations)

        def interval_view(observation: ObservationRecord) -> dict[str, Any]:
            return {
                "observation_id": observation.observation_id,
                "start_frame": int(observation.start_frame),
                "end_frame": int(observation.end_frame),
            }

        selected_intervals = sorted(
            selected_observations,
            key=lambda observation: (
                int(observation.start_frame),
                int(observation.end_frame),
                observation.observation_id,
            ),
        )
        union_end = int(selected_intervals[0].end_frame)
        gap: list[int] | None = None
        for observation in selected_intervals[1:]:
            start = int(observation.start_frame)
            end = int(observation.end_frame)
            if start > union_end + 1:
                gap = [union_end + 1, start - 1]
                break
            union_end = max(union_end, end)
        if gap is not None:
            raise ValueError(
                "merge_observations.observation_ids must form one continuous closed-frame union; "
                f"selected intervals leave gap {gap}. "
                f"selected_intervals={[interval_view(observation) for observation in selected_intervals]}; "
                f"current_active_timeline={[interval_view(observation) for observation in active_observations]}."
            )

        selected_id_set = set(normalized_observation_ids)
        unselected_inside_union: list[ObservationRecord] = []
        for observation in active_observations:
            if observation.observation_id in selected_id_set:
                continue
            start = int(observation.start_frame)
            end = int(observation.end_frame)
            intersection_start = max(start, merged_start)
            intersection_end = min(end, merged_end)
            if intersection_start > intersection_end:
                continue
            left_boundary_only = (
                intersection_start == intersection_end == merged_start
                and start < merged_start
                and end == merged_start
            )
            right_boundary_only = (
                intersection_start == intersection_end == merged_end
                and start == merged_end
                and end > merged_end
            )
            if not (left_boundary_only or right_boundary_only):
                unselected_inside_union.append(observation)
        if unselected_inside_union:
            raise ValueError(
                "merge_observations.observation_ids must include every active observation inside the merged "
                "frame union. Add the unselected active observations to observation_ids or revise the timeline "
                "before merging. "
                f"unselected_active_observations={[interval_view(observation) for observation in unselected_inside_union]}; "
                f"selected_intervals={[interval_view(observation) for observation in selected_intervals]}; "
                f"current_active_timeline={[interval_view(observation) for observation in active_observations]}."
            )

        coverage_blocker = self._observation_asset_coverage_blocker(
            tool_name="merge_observations",
            retry_tool_name="merge_observations",
            asset_ids=normalized_asset_ids,
            start=merged_start,
            end=merged_end,
            asset_by_id=asset_by_id,
        )
        if coverage_blocker is not None:
            blocked_item, next_action = coverage_blocker
            data = {
                "merged_observation_id": None,
                "inactive_observation_ids": [],
                **self._observation_status_summary(),
            }
            result = tool_blocked(
                blocked_items=[blocked_item],
                next_required_actions=[next_action],
                data=data,
            )
            self.append_trace("merge_observations", result)
            return result

        kept_observation = selected_observations[0]
        inactive_observations = selected_observations[1:]
        merged_observation = replace(
            kept_observation,
            asset_ids=normalized_asset_ids,
            start_frame=merged_start,
            end_frame=merged_end,
            text=clean_text,
            confidence=parsed_confidence,
            status="draft",
            updated_stage_id=normalized_stage_id or None,
            confirmed_stage_id=None,
            confirmation_asset_ids=[],
            confirmation_reason="",
            revision_count=int(kept_observation.revision_count) + 1,
        )
        inactive_updates = [
            replace(
                observation,
                status="inactive",
                updated_stage_id=normalized_stage_id or None,
                confirmed_stage_id=None,
                confirmation_asset_ids=[],
                confirmation_reason=(
                    f"Merged into {merged_observation.observation_id} by merge_observations: {clean_reason}"
                ),
                revision_count=int(observation.revision_count) + 1,
            )
            for observation in inactive_observations
        ]
        updates = {
            merged_observation.observation_id: merged_observation,
            **{observation.observation_id: observation for observation in inactive_updates},
        }

        revision_records: list[ObservationRevisionRecord] = []
        for old_observation, new_observation in [
            (kept_observation, merged_observation),
            *zip(inactive_observations, inactive_updates),
        ]:
            revision_records.append(
                ObservationRevisionRecord(
                    revision_id=f"obsrev_{len(self.observation_revision_history) + len(revision_records) + 1:04d}",
                    observation_id=old_observation.observation_id,
                    stage_id=normalized_stage_id or None,
                    reason=clean_reason,
                    old_observation=old_observation.to_dict(),
                    new_observation=new_observation.to_dict(),
                )
            )

        for position, observation in enumerate(self.observations):
            if observation.observation_id in updates:
                self.observations[position] = updates[observation.observation_id]
        self.observation_revision_history.extend(revision_records)

        data = {
            "action": "merge",
            "merged_observation_id": merged_observation.observation_id,
            "inactive_observation_ids": [observation.observation_id for observation in inactive_updates],
            "observations": [
                self._observation_agent_view(observation, stage_id=normalized_stage_id)
                for observation in self._active_observations_sorted()
            ],
            **self._observation_status_summary(),
        }
        result = tool_success(data)
        self.append_trace(
            "merge_observations",
            tool_success({**data, "revisions": [record.to_dict() for record in revision_records]}),
        )
        self._save_state()
        return result

    def complete_stage(self, *, stage_id: str | None = None) -> ToolResult:
        """Attempt to complete the current stage after deterministic gate checks."""

        normalized_stage_id = str(stage_id or "").strip()
        if not normalized_stage_id:
            raise ValueError("complete_stage requires the runner-provided stage_id.")
        if normalized_stage_id == "bootstrap_observations":
            data = self._complete_observation_stage(stage_id=normalized_stage_id, require_confirmed=False)
            if (
                tool_result_ok(data)
                and self._stage_promotes_observations_on_complete(normalized_stage_id)
            ):
                promoted_ids = self._promote_active_observations_to_confirmed(
                    stage_id=normalized_stage_id
                )
                tool_result_data(data)["promoted_observation_ids"] = promoted_ids
        elif normalized_stage_id == "challenge_observations":
            data = self._complete_observation_stage(stage_id=normalized_stage_id, require_confirmed=True)
        elif normalized_stage_id == "progress":
            data = self._complete_progress_stage(stage_id=normalized_stage_id)
        else:
            raise ValueError(f"Unsupported stage for complete_stage: {normalized_stage_id!r}")
        self.append_trace("complete_stage", data)
        self._save_state()
        return data

    def save_result(
        self,
        *,
        summary: Any,
        external_review_need: Any,
        external_review_focus: Any = None,
        payload: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Save the final reward-agent result and manifest."""

        clean_summary = _required_str_tool_arg("summary", summary, tool_name="save_result")
        parsed_external_review_need = _required_external_review_need_tool_arg(
            "external_review_need",
            external_review_need,
            tool_name="save_result",
        )
        clean_external_review_focus = _optional_external_review_focus_tool_arg(
            "external_review_focus",
            external_review_focus,
            external_review_need=parsed_external_review_need,
            tool_name="save_result",
        )
        final_payload = dict(payload or {})
        result_path = self.run_dir / "final_result.json"
        summary_path = self.run_dir / "final_summary.md"
        # Human-review artifacts only: render the corrected/generated progress
        # curve after finalization, but do not expose them as model input.
        self.nonfatal_artifact_warnings = []
        self.progress_plot_path = self._write_progress_plot()
        self.progress_review_video_path = self._write_optional_progress_review_video()
        if self.nonfatal_artifact_warnings:
            final_payload["nonfatal_artifact_warnings"] = [
                dict(warning) for warning in self.nonfatal_artifact_warnings
            ]
        final_result = FinalResult(
            summary=clean_summary,
            external_review_need=parsed_external_review_need,
            external_review_focus=clean_external_review_focus,
            payload=final_payload,
        )
        # Keep the in-memory assessment aligned with the canonical final result.
        # This also covers harness-level blocked fallbacks that bypass write_summary.
        self.final_summary = final_result.summary
        self.external_review_need = int(final_result.external_review_need)
        self.external_review_focus = final_result.external_review_focus
        result_path.write_text(
            json.dumps(final_result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(final_result.summary.rstrip() + "\n", encoding="utf-8")
        manifest = self._manifest(
            final_result_path=result_path,
            final_summary_path=summary_path,
        )
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.final_result = final_result
        data = {
            "saved": True,
            "summary": final_result.summary,
            "external_review_need": int(final_result.external_review_need),
            "external_review_focus": final_result.external_review_focus,
            "payload": final_result.payload,
        }
        self.append_trace("save_result", tool_success(manifest))
        self._save_state()
        return tool_success(data)

    def tool_manifest(
        self,
        *,
        allowed_tools: tuple[str, ...] | list[str] | None = None,
        stage_id: str | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for the active stage."""

        tools = _tool_definitions(
            value_max_points=self.config.progress.value_max_points,
            default_jump_threshold=self.config.progress.default_jump_threshold_points,
            tool_config=tool_config if tool_config is not None else self._stage_tool_config(stage_id),
        )
        if allowed_tools:
            allowed = {str(tool) for tool in allowed_tools}
            tools = [tool for tool in tools if tool["function"]["name"] in allowed]
        return tools

    def dispatch_tool(self, name: str, arguments: dict[str, Any], *, stage_id: str | None = None) -> dict[str, Any]:
        """Execute one agent-facing tool by name."""

        if name == "read_robot_instruction":
            return self.read_robot_instruction()
        if name == "read_metadata":
            return self.read_metadata()
        if name == "read_progress":
            tool_arguments = dict(arguments)
            # `trace_source` is a harness-owned provenance marker. Keep it out
            # of agent-supplied tool arguments so trace/source labels remain
            # auditable even if a model emits undeclared fields.
            tool_arguments.pop("trace_source", None)
            return self.read_progress(**tool_arguments, trace_source="tool")
        if name == "summarize_progress":
            tool_arguments = dict(arguments)
            # See the read_progress branch above: provenance labels must be set
            # by the harness path, not by model-controlled tool arguments.
            tool_arguments.pop("trace_source", None)
            return self.summarize_progress(**tool_arguments, trace_source="tool")
        if name == "inspect_visual":
            return self.inspect_visual(**arguments, stage_id=stage_id, provided_arguments=set(arguments))
        if name == "edit_progress_segments":
            return self.edit_progress_segments(**arguments)
        if name == "calibrate_progress_curve":
            return self.calibrate_progress_curve(**arguments)
        if name == "write_summary":
            return self.write_summary(**arguments)
        if name == "export_corrected_progress":
            return self.export_corrected_progress(**arguments)
        if name == "validate_progress":
            return self.validate_progress()
        if name == "add_observations":
            return self.add_observations(**arguments, stage_id=stage_id)
        if name == "read_observations":
            return self.read_observations(stage_id=stage_id)
        if name == "confirm_observations":
            return self.confirm_observations(**arguments, stage_id=stage_id)
        if name == "revise_observations":
            return self.revise_observations(**arguments, stage_id=stage_id)
        if name == "merge_observations":
            return self.merge_observations(**arguments, stage_id=stage_id)
        if name == "complete_stage":
            return self.complete_stage(stage_id=stage_id)
        if name == "save_result":
            return self.save_result(**arguments)
        raise KeyError(f"Unknown reward-agent tool: {name}")

    def pending_media_for_tool_results(self, tool_results: list[dict[str, Any]]) -> list[PendingMediaAttachment]:
        """Resolve pathful media attachments for visual tool results."""

        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}
        attachments: list[PendingMediaAttachment] = []
        for result in tool_results:
            if not tool_result_ok(result):
                continue
            asset_id = tool_result_data(result).get("asset_id")
            if not asset_id or asset_id not in asset_by_id:
                continue
            asset = asset_by_id[str(asset_id)]
            attachments.append(
                PendingMediaAttachment(
                    asset_id=asset.asset_id,
                    kind=asset.kind,
                    path=asset.path,
                    frame_indices=list(asset.frame_indices),
                    frame_range=asset.frame_range,
                    sample_every_n=asset.sample_every_n,
                )
            )
        return attachments

    def _find_visual_asset(self, asset_id: str) -> VisualAsset | None:
        """Return a registered visual asset by id without exposing its path."""

        for asset in self.visual_assets:
            if asset.asset_id == asset_id:
                return asset
        return None

    def _find_duplicate_visual_asset(
        self,
        *,
        stage_id: str | None,
        frame_indices: list[int],
        media_mode: str,
        crop: dict[str, int] | None,
        progress_label_revision: int | None,
    ) -> VisualAsset | None:
        """Return an existing same-stage asset for an identical visual request."""

        expected_kind = "video_clip" if media_mode == "video" else "tile"
        normalized_stage_id = str(stage_id or "") if stage_id else None
        requested_frames = [int(frame) for frame in frame_indices]
        requested_crop = dict(crop) if crop is not None else None
        for asset in self.visual_assets:
            if asset.created_stage_id != normalized_stage_id:
                continue
            if asset.kind != expected_kind:
                continue
            if [int(frame) for frame in asset.frame_indices] != requested_frames:
                continue
            asset_crop = dict(asset.crop) if asset.crop is not None else None
            if asset_crop != requested_crop:
                continue
            if asset.progress_label_revision != progress_label_revision:
                continue
            return asset
        return None

    def _challenge_observation_next_action(self, observation: ObservationRecord) -> str:
        """Tell the model exactly how to gather valid challenge evidence."""

        target_start = int(self.frame_indices[0])
        target_end = int(self.frame_indices[-1])
        start = int(observation.start_frame)
        end = int(observation.end_frame)
        return (
            f"Challenge {observation.observation_id} with new inspect_visual evidence from the "
            "challenge_observations stage, then either confirm it as-is or revise it. "
            f"For observation [s, e] = [{start}, {end}], use start_frame < s when possible "
            "and end_frame > e when possible; clamp only at the trajectory boundaries "
            f"[{target_start}, {target_end}]. The evidence must also include frames inside [s, e]."
        )

    def record_tool_call(
        self,
        *,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Persist one executed tool call in session state."""

        self.tool_history.append(
            ToolCallRecord(
                tool_call_id=str(tool_call_id),
                name=str(name),
                arguments=dict(arguments),
                result=dict(result),
            )
        )
        self._save_state()

    def append_trace(self, event: str, payload: dict[str, Any]) -> None:
        """Append one JSONL trace event."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().astimezone().isoformat(),
            "event": event,
            "payload": _json_ready(payload),
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def build_case_context_payload(self, *, stage_id: str) -> dict[str, Any]:
        """Build the path-redacted context payload injected into a stage."""

        context = {
            "case_id": self.case_id,
            "stage_id": stage_id,
            "robot_instruction": self.robot_instruction,
            "robot_instruction_usage_rule": (
                "Use the robot instruction and any appended reward-agent-only context only to "
                "identify relevant visible entities, controls, and goal relations. Do not treat "
                "it as evidence that any action, state transition, success, or failure occurred."
            ),
            "metadata": self._public_metadata(),
            "observations": [
                self._observation_agent_view(observation, stage_id=stage_id)
                for observation in self._observations_visible_to_stage(stage_id)
            ],
            "observation_coverage": self._observation_coverage_agent_view(
                self._observations_visible_to_stage(stage_id),
                stage_id=stage_id,
            ),
            "visual_assets": [
                self.visual_asset_context_summary(asset)
                for asset in self._visual_assets_visible_to_stage(stage_id)
            ],
            "note": (
                "Use runner-attached visual evidence when present. Request additional bounded "
                "visual evidence with tools before making new visual claims."
            ),
        }
        if str(stage_id or "").strip() == "progress":
            context["progress_state"] = {
                "has_progress_proposal": self.has_progress_proposal,
                "mode": self._progress_mode(),
                "curve_status": self._progress_curve_status(),
                "value_range": [0, int(self.config.progress.value_max_points)],
                "default_jump_threshold": int(self.config.progress.default_jump_threshold_points),
                "num_rows": int(len(self.frame_indices)),
                "start_frame": int(self.frame_indices[0]),
                "end_frame": int(self.frame_indices[-1]),
                "assigned_count": int(np.sum(self.assigned_mask)),
                "total_count": int(len(self.frame_indices)),
            }
        return context

    def reference_example_messages(self) -> list[dict[str, Any]]:
        """Return task-bound reference media messages for conversation bootstrap."""

        return [example.to_message() for example in self.reference_examples]

    def metadata_resolution_summary(self) -> dict[str, Any]:
        """Return public video shape and resolution metadata for trace audits."""

        metadata = self._public_metadata()
        summary: dict[str, Any] = {}
        for key in ("num_frames", "frame_shape_hwc", "height", "width", "resolution"):
            if key in metadata:
                summary[key] = metadata[key]
        shape = metadata.get("frame_shape_hwc")
        if isinstance(shape, list) and len(shape) >= 2:
            summary.setdefault("height", shape[0])
            summary.setdefault("width", shape[1])
            summary.setdefault("resolution", {"height": shape[0], "width": shape[1]})
        return _json_ready(summary)

    @staticmethod
    def compact_visual_asset_summary(asset: VisualAsset) -> dict[str, Any]:
        """Return a trace-friendly media summary without a full frame index list."""

        return {
            "asset_id": asset.asset_id,
            "kind": asset.kind,
            "frame_range": [int(asset.frame_range[0]), int(asset.frame_range[1])],
            "sample_every_n": int(asset.sample_every_n),
            "attached_frame_count": int(asset.attached_frame_count),
            "reason": asset.reason,
            "crop": dict(asset.crop) if asset.crop is not None else None,
            "created_stage_id": asset.created_stage_id,
        }

    @staticmethod
    def visual_asset_context_summary(asset: VisualAsset) -> dict[str, Any]:
        """Return factual visual asset context without attachment instructions."""

        return asset.to_dict(include_path=False)

    def _stage_tool_config(self, stage_id: str | None) -> dict[str, Any]:
        """Return per-tool policy configured for one workflow stage."""

        normalized_stage_id = str(stage_id or "").strip()
        workflow = self.config.workflow
        if not normalized_stage_id or workflow is None:
            return {}
        for stage in workflow.stages:
            if stage.stage_id == normalized_stage_id:
                return dict(stage.tool_config)
        return {}

    def _stage_promotes_observations_on_complete(self, stage_id: str | None) -> bool:
        """Return whether a successful stage completion accepts its observations as final."""

        normalized_stage_id = str(stage_id or "").strip()
        workflow = self.config.workflow
        if not normalized_stage_id or workflow is None:
            return False
        for stage in workflow.stages:
            if stage.stage_id == normalized_stage_id:
                return stage.promote_observations_to_confirmed_on_complete
        return False

    def _promote_active_observations_to_confirmed(self, *, stage_id: str) -> list[str]:
        """Accept bootstrap observations after the deterministic completion gate passes.

        Bootstrap assets remain source provenance only. They are deliberately not copied into
        confirmation_asset_ids because a later fresh model session cannot see those attachments.
        """

        promoted_ids: list[str] = []
        for index, observation in enumerate(self.observations):
            if observation.status != "draft":
                continue
            self.observations[index] = replace(
                observation,
                status="confirmed",
                updated_stage_id=stage_id,
                confirmed_stage_id=stage_id,
                confirmation_asset_ids=[],
                confirmation_reason="",
            )
            promoted_ids.append(observation.observation_id)
        return promoted_ids

    def _active_observations_sorted(self) -> list[ObservationRecord]:
        """Return non-inactive observations in timeline order."""

        return sorted(
            (observation for observation in self.observations if observation.status != "inactive"),
            key=lambda observation: (
                int(observation.start_frame),
                int(observation.end_frame),
                observation.observation_id,
            ),
        )

    def _resolve_run_dir(self, *, run_dir: str | Path | None) -> Path:
        if run_dir is not None:
            return Path(run_dir).expanduser().resolve()
        return Path("outputs/reward_agent/runs").resolve().joinpath(self.run_id, self.case_id)

    def _frame_indices_without_progress_proposal(self) -> np.ndarray:
        """Infer a frame axis for generation-only progress cases."""

        num_frames = (
            self.metadata.get("num_frames")
            or getattr(self.episode_source, "num_frames", None)
            or self._infer_video_metadata().get("num_frames")
        )
        if num_frames is not None:
            count = int(num_frames)
            if count <= 0:
                raise ValueError(f"metadata.num_frames must be positive, got {count}.")
            return np.arange(count, dtype=np.int64)
        if self.video_path is None:
            raise ValueError("Cannot infer progress frame axis without metadata.num_frames or a video file.")
        reader = imageio.get_reader(self.video_path)
        try:
            count = int(reader.count_frames())
        finally:
            reader.close()
        if count <= 0:
            raise ValueError(f"Video has no decodable frames: {self.video_path}")
        return np.arange(count, dtype=np.int64)

    def _public_metadata(self) -> dict[str, Any]:
        """Return the strict metadata whitelist exposed to the model."""

        public: dict[str, Any] = {}
        for key in _PUBLIC_METADATA_KEYS:
            value = self.metadata.get(key)
            if value is not None:
                public[key] = value

        inferable_missing = any(key not in public for key in ("num_frames", "frame_shape_hwc"))
        if inferable_missing:
            inferred = self._infer_video_metadata()
            for key in ("num_frames", "frame_shape_hwc"):
                if key not in public and inferred.get(key) is not None:
                    public[key] = inferred[key]

        public["agent_video_fps"] = int(self.config.visual.agent_video_fps)
        public["frame_indexing_rule"] = (
            "Use videos from the trajectory being evaluated as frame-indexed videos, "
            "not real-time videos. Do not report time in minutes, seconds, or mm:ss "
            "format. Treat each trajectory-video timestamp value as an indexing unit, "
            "not real time. When agent_video_fps "
            "is 1, visible timestamps label temporal groups of original frames "
            "rather than individual frames. A normal two-frame group containing "
            "frames N and N + 1 is labeled <N + 0.5 seconds> and should be "
            "interpreted as frame group [N, N + 1], not only frame N. For example, "
            "<68.5 seconds> corresponds to frame group [68, 69], not only frame 68. "
            "If the final temporal group contains one unpaired final frame L, its "
            "timestamp is <L.0 seconds> and should be interpreted as frame L alone. "
            "A terminal <14.0 seconds> in a 15-frame attachment means frame 14 "
            "alone, not frames [14, 15]."
        )

        return {key: _json_ready(public[key]) for key in _PUBLIC_METADATA_KEYS if key in public}

    def _infer_video_metadata(self) -> dict[str, Any]:
        """Infer safe, public video metadata from the hidden raw video path."""

        if self._video_metadata_cache is not None:
            return dict(self._video_metadata_cache)
        if self.video_path is None:
            self._video_metadata_cache = {}
            return {}

        inferred: dict[str, Any] = {}
        reader = imageio.get_reader(self.video_path)
        try:
            try:
                video_metadata = reader.get_meta_data()
            except Exception:
                video_metadata = {}

            fps = _positive_float_or_none(video_metadata.get("fps"))
            if fps is not None:
                inferred["fps"] = fps

            count = _positive_int_or_none(video_metadata.get("nframes"))
            if count is None:
                try:
                    count = int(reader.count_frames())
                except Exception:
                    count = None
            if count is not None and count > 0:
                inferred["num_frames"] = count

            duration = _positive_float_or_none(video_metadata.get("duration"))
            if duration is not None:
                inferred["duration_seconds"] = duration
            elif count is not None and count > 0 and fps is not None:
                inferred["duration_seconds"] = round(float(count) / fps, 6)

            # Shape is useful public context and can be inferred without
            # exposing the hidden video path or any simulator/debug state.
            try:
                frame = np.asarray(reader.get_data(0))
            except Exception:
                frame = None
            if frame is not None and frame.ndim >= 2:
                if frame.ndim == 2:
                    inferred["frame_shape_hwc"] = [int(frame.shape[0]), int(frame.shape[1]), 1]
                else:
                    inferred["frame_shape_hwc"] = [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
        finally:
            reader.close()

        self._video_metadata_cache = inferred
        return dict(inferred)

    def _observations_visible_to_stage(self, stage_id: str | None) -> list[ObservationRecord]:
        """Return the current observations exposed to the active workflow stage."""

        normalized_stage_id = str(stage_id or "").strip()
        if normalized_stage_id == "progress":
            return [
                observation
                for observation in self.observations
                if observation.status == "confirmed"
            ]
        return [observation for observation in self.observations if observation.status != "inactive"]

    def _visual_assets_visible_to_stage(self, stage_id: str | None) -> list[VisualAsset]:
        """Return visual assets that are meaningful evidence in the active stage."""

        normalized_stage_id = str(stage_id or "").strip()
        if normalized_stage_id == "challenge_observations":
            return [
                asset
                for asset in self.visual_assets
                if asset.created_stage_id == normalized_stage_id
            ]
        if normalized_stage_id == "progress":
            return [
                asset
                for asset in self.visual_assets
                if asset.created_stage_id in {"challenge_observations", "progress"}
            ]
        return list(self.visual_assets)

    def _observation_agent_view(
        self,
        observation: ObservationRecord,
        *,
        stage_id: str | None,
        include_status: bool | None = None,
    ) -> dict[str, Any]:
        """Return the stage-specific observation fields exposed to the model."""

        normalized_stage_id = str(stage_id or "").strip()
        data: dict[str, Any] = {
            "observation_id": observation.observation_id,
            "start_frame": int(observation.start_frame),
            "end_frame": int(observation.end_frame),
            "text": observation.text,
            "confidence": int(observation.confidence),
        }

        # Source assets are useful while drafting observations, but they become
        # misleading in later stages because challenge confirmation requires
        # fresh challenge-stage evidence. Keep provenance internally and expose
        # only the evidence ids that are valid for the current stage.
        if normalized_stage_id in {"", "bootstrap_observations"}:
            data["asset_ids"] = list(observation.asset_ids)

        # Bootstrap is only about drafting a full-trajectory timeline, so the
        # confirmation state machine stays hidden until the challenge stage.
        should_include_status = (
            normalized_stage_id == "challenge_observations"
            if include_status is None
            else bool(include_status)
        )
        if should_include_status:
            data["status"] = observation.status

        if normalized_stage_id == "challenge_observations" and observation.status == "draft":
            data["required_challenge"] = self._challenge_observation_next_action(observation)
        if (
            normalized_stage_id in {"challenge_observations", "progress"}
            and observation.status == "confirmed"
            and observation.confirmation_asset_ids
        ):
            data["confirmation_asset_ids"] = list(observation.confirmation_asset_ids)
            data["confirmation_reason"] = observation.confirmation_reason
        if int(observation.revision_count):
            data["revision_count"] = int(observation.revision_count)
        return data

    def _observation_coverage_agent_view(
        self,
        observations: list[ObservationRecord],
        *,
        stage_id: str | None,
        include_status: bool | None = None,
    ) -> dict[str, Any]:
        """Return coverage diagnostics with status fields only when useful."""

        normalized_stage_id = str(stage_id or "").strip()
        data = self._observation_coverage_summary(observations)
        should_include_status = (
            normalized_stage_id == "challenge_observations"
            if include_status is None
            else bool(include_status)
        )
        if not should_include_status:
            data.pop("draft_ids", None)
            data.pop("confirmed_ids", None)
            data.pop("unconfirmed_ids", None)
            data.pop("confirmed_count", None)
            data.pop("draft_count", None)
        return data

    def _observation_status_summary(self) -> dict[str, Any]:
        """Summarize draft/confirmed observation state for tool responses."""

        observations = self._active_observations_sorted()
        coverage = self._observation_coverage_summary(observations)
        confirmed_ids = [observation.observation_id for observation in observations if observation.status == "confirmed"]
        draft_ids = [observation.observation_id for observation in observations if observation.status == "draft"]
        return {
            "active_count": len(observations),
            "confirmed_count": len(confirmed_ids),
            "draft_count": len(draft_ids),
            "confirmed_ids": confirmed_ids,
            "draft_ids": draft_ids,
            "unconfirmed_ids": draft_ids,
            "coverage": coverage,
        }

    def _observation_coverage_summary(self, observations: list[ObservationRecord]) -> dict[str, Any]:
        """Return deterministic closed-interval coverage diagnostics."""

        target_start = int(self.frame_indices[0])
        target_end = int(self.frame_indices[-1])
        intervals = sorted(
            (
                (int(observation.start_frame), int(observation.end_frame), observation.observation_id)
                for observation in observations
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        gaps: list[list[int]] = []
        overlaps: list[list[int]] = []
        shared_boundary_candidates: list[dict[str, Any]] = []
        seen_shared_boundary_frames: set[int] = set()
        outside_ranges: list[dict[str, Any]] = []
        covered_ranges: list[list[int]] = []

        def record_overlap(start: int, end: int) -> None:
            overlap = [int(start), int(end)]
            if overlap not in overlaps:
                overlaps.append(overlap)

        cursor = target_start
        covered_end: int | None = None
        covered_end_observation_id: str | None = None
        for start, end, observation_id in intervals:
            if start < target_start or end > target_end:
                outside_ranges.append(
                    {
                        "observation_id": observation_id,
                        "range": [start, end],
                        "target_range": [target_start, target_end],
                    }
                )

            clipped_start = max(start, target_start)
            clipped_end = min(end, target_end)
            if clipped_start > clipped_end:
                continue

            if covered_end is None:
                if clipped_start > cursor:
                    gaps.append([cursor, clipped_start - 1])
                covered_ranges.append([clipped_start, clipped_end])
                covered_end = clipped_end
                covered_end_observation_id = observation_id
                cursor = max(cursor, clipped_end + 1)
                continue

            if clipped_start <= covered_end:
                overlap_end = min(clipped_end, covered_end)
                if clipped_start == covered_end and overlap_end == clipped_start:
                    # Adjacent closed intervals may intentionally share one
                    # boundary frame as visual evidence for both sides of a
                    # semantic transition. A second shared boundary at the same
                    # frame means three or more observations cover that frame,
                    # so it is reported as a blocking one-frame overlap.
                    boundary_frame = int(clipped_start)
                    if boundary_frame in seen_shared_boundary_frames:
                        record_overlap(boundary_frame, boundary_frame)
                    else:
                        seen_shared_boundary_frames.add(boundary_frame)
                        shared_boundary_candidates.append(
                            {
                                "frame": boundary_frame,
                                "left_observation_id": covered_end_observation_id,
                                "right_observation_id": observation_id,
                            }
                        )
                else:
                    record_overlap(clipped_start, overlap_end)
                covered_ranges[-1][1] = max(covered_ranges[-1][1], clipped_end)
                if clipped_end > covered_end:
                    covered_end = clipped_end
                    covered_end_observation_id = observation_id
                cursor = max(cursor, covered_end + 1)
                continue

            if clipped_start > cursor:
                gaps.append([cursor, clipped_start - 1])
            covered_ranges.append([clipped_start, clipped_end])
            covered_end = clipped_end
            covered_end_observation_id = observation_id
            cursor = max(cursor, clipped_end + 1)

        if cursor <= target_end:
            gaps.append([cursor, target_end])

        confirmed_ids = [observation.observation_id for observation in observations if observation.status == "confirmed"]
        draft_ids = [observation.observation_id for observation in observations if observation.status == "draft"]
        shared_boundaries = [
            candidate
            for candidate in shared_boundary_candidates
            if not any(overlap[0] <= int(candidate["frame"]) <= overlap[1] for overlap in overlaps)
        ]
        return {
            "target_start_frame": target_start,
            "target_end_frame": target_end,
            "covered_ranges": covered_ranges,
            "gaps": gaps,
            "overlaps": overlaps,
            "shared_boundaries": shared_boundaries,
            "outside_ranges": outside_ranges,
            "draft_ids": draft_ids,
            "confirmed_ids": confirmed_ids,
            "unconfirmed_ids": draft_ids,
            "active_count": len(observations),
            "confirmed_count": len(confirmed_ids),
            "draft_count": len(draft_ids),
        }

    def _observation_asset_coverage_blocker(
        self,
        *,
        tool_name: str,
        retry_tool_name: str,
        asset_ids: list[str],
        start: int,
        end: int,
        asset_by_id: dict[str, VisualAsset],
    ) -> tuple[str, str] | None:
        """Return a recoverable blocker when cited assets do not cover an interval."""

        asset_ranges = sorted(
            (int(asset_by_id[asset_id].frame_range[0]), int(asset_by_id[asset_id].frame_range[1]))
            for asset_id in asset_ids
        )
        cursor = int(start)
        for range_start, range_end in asset_ranges:
            if range_end < cursor:
                continue
            if range_start > cursor:
                return self._format_observation_asset_coverage_blocker(
                    tool_name=tool_name,
                    retry_tool_name=retry_tool_name,
                    start=start,
                    end=end,
                    asset_ranges=asset_ranges,
                    first_missing_frame=cursor,
                )
            cursor = max(cursor, range_end + 1)
            if cursor > end:
                return None

        return self._format_observation_asset_coverage_blocker(
            tool_name=tool_name,
            retry_tool_name=retry_tool_name,
            start=start,
            end=end,
            asset_ranges=asset_ranges,
            first_missing_frame=cursor,
        )

    def _format_observation_asset_coverage_blocker(
        self,
        *,
        tool_name: str,
        retry_tool_name: str,
        start: int,
        end: int,
        asset_ranges: list[tuple[int, int]],
        first_missing_frame: int,
    ) -> tuple[str, str]:
        """Format a blocked result for recoverable observation evidence gaps."""

        range_text = [[range_start, range_end] for range_start, range_end in asset_ranges]
        blocked_item = (
            f"{tool_name} frame range [{start}, {end}] is not fully covered by the "
            f"referenced asset frame ranges {range_text}; missing coverage begins at "
            f"frame {first_missing_frame}."
        )
        next_action = (
            f"Cite additional asset_ids whose frame ranges cover every frame in [{start}, {end}], "
            f"or call inspect_visual for the missing frames, then retry {retry_tool_name}."
        )
        return blocked_item, next_action

    def _complete_observation_stage(self, *, stage_id: str, require_confirmed: bool) -> ToolResult:
        """Gate bootstrap/challenge stages on observation coverage and status."""

        blockers: list[str] = []
        next_actions: list[str] = []
        observations = self._active_observations_sorted()
        coverage = self._observation_coverage_summary(observations)
        if not observations:
            blockers.append("No observations have been recorded.")
            next_actions.append("Inspect visual evidence and add observations covering the full trajectory.")
        for gap in coverage["gaps"]:
            blockers.append(f"Observation coverage has gap frames [{gap[0]}, {gap[1]}].")
            next_actions.append(f"Add or revise observations to cover frames [{gap[0]}, {gap[1]}].")
        for overlap in coverage["overlaps"]:
            blockers.append(f"Observation coverage has overlapping frames [{overlap[0]}, {overlap[1]}].")
            next_actions.append(
                "Call read_observations to get current ids, then use revise_observations "
                "to fix overlapping existing observations."
            )
        for outside in coverage["outside_ranges"]:
            blockers.append(
                f"{outside['observation_id']} range {outside['range']} falls outside target "
                f"range {outside['target_range']}."
            )
            next_actions.append("Revise out-of-range observations to use valid original frame indices.")

        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}
        for observation in observations:
            missing = [asset_id for asset_id in observation.asset_ids if asset_id not in asset_by_id]
            if missing:
                blockers.append(f"{observation.observation_id} references unknown visual asset ids: {missing}.")
                next_actions.append(f"Revise {observation.observation_id} with valid visual evidence.")

        if require_confirmed:
            for observation in observations:
                if observation.status != "confirmed":
                    blockers.append(f"{observation.observation_id} is still draft.")
                    next_actions.append(self._challenge_observation_next_action(observation))
                    continue
                evidence_blockers = self._confirmation_evidence_blockers(observation)
                blockers.extend(evidence_blockers)
                if evidence_blockers:
                    next_actions.append(self._challenge_observation_next_action(observation))

        if blockers:
            data = {
                "stage_id": stage_id,
            }
            if require_confirmed:
                data.update(self._observation_status_summary())
            else:
                data["active_count"] = len(observations)
                data["coverage"] = self._observation_coverage_agent_view(
                    observations,
                    stage_id=stage_id,
                    include_status=False,
                )
            return tool_blocked(
                blocked_items=blockers,
                next_required_actions=_dedupe_preserve_order(next_actions),
                data=data,
            )

        data = {"stage_id": stage_id}
        if require_confirmed:
            data.update(self._observation_status_summary())
        else:
            data["active_count"] = len(observations)
            data["coverage"] = self._observation_coverage_agent_view(
                observations,
                stage_id=stage_id,
                include_status=False,
            )
        return tool_success(data)

    def _confirmation_evidence_blockers(self, observation: ObservationRecord) -> list[str]:
        """Return blockers when confirmation evidence does not cover a current observation."""

        blockers: list[str] = []
        if not observation.confirmation_asset_ids:
            return [f"{observation.observation_id} has no confirmation evidence asset ids."]

        asset_by_id = {asset.asset_id: asset for asset in self.visual_assets}
        assets: list[VisualAsset] = []
        for asset_id in observation.confirmation_asset_ids:
            asset = asset_by_id.get(asset_id)
            if asset is None:
                blockers.append(f"{observation.observation_id} confirmation references unknown asset {asset_id}.")
                continue
            if asset.created_stage_id != "challenge_observations":
                blockers.append(
                    f"{observation.observation_id} confirmation asset {asset_id} was not created "
                    "during challenge_observations."
                )
            assets.append(asset)
        if not assets:
            return blockers

        target_start = int(self.frame_indices[0])
        target_end = int(self.frame_indices[-1])
        sampled_frames = sorted({int(frame) for asset in assets for frame in asset.frame_indices})
        observation_start = int(observation.start_frame)
        observation_end = int(observation.end_frame)
        inside_frames = [
            frame for frame in sampled_frames
            if observation_start <= frame <= observation_end
        ]
        if not inside_frames:
            blockers.append(
                f"{observation.observation_id} confirmation evidence does not cover the "
                "observation interval itself."
            )
        if observation_start > target_start:
            left_start = max(target_start, observation_start - _CONFIRMATION_BOUNDARY_SEARCH_FRAMES)
            left_end = observation_start - 1
            left_boundary_frames = [
                frame for frame in sampled_frames
                if left_start <= frame <= left_end
            ]
            if not left_boundary_frames:
                blockers.append(
                    f"{observation.observation_id} confirmation evidence does not cover visual "
                    "context before the start boundary."
                )
        if observation_end < target_end:
            right_start = observation_end + 1
            right_end = min(target_end, observation_end + _CONFIRMATION_BOUNDARY_SEARCH_FRAMES)
            right_boundary_frames = [
                frame for frame in sampled_frames
                if right_start <= frame <= right_end
            ]
            if not right_boundary_frames:
                blockers.append(
                    f"{observation.observation_id} confirmation evidence does not cover visual "
                    "context after the end boundary."
                )
        return blockers

    def _complete_progress_stage(self, *, stage_id: str) -> ToolResult:
        """Gate finalization, export final progress artifacts, and save results."""

        blockers: list[str] = []
        next_actions: list[str] = []
        confirmed_observations = [observation for observation in self.observations if observation.status == "confirmed"]
        coverage = self._observation_coverage_summary(confirmed_observations)
        if not confirmed_observations:
            blockers.append("No confirmed observations are available for progress.")
            next_actions.append(
                "Return to the observation challenge stage and produce full-trajectory confirmed observations."
            )
        for gap in coverage["gaps"]:
            blockers.append(f"Confirmed observation coverage has gap frames [{gap[0]}, {gap[1]}].")
        for overlap in coverage["overlaps"]:
            blockers.append(f"Confirmed observation coverage has overlapping frames [{overlap[0]}, {overlap[1]}].")
        if not bool(np.all(self.assigned_mask)):
            unassigned = np.flatnonzero(~self.assigned_mask)
            first = int(self.frame_indices[int(unassigned[0])])
            last = int(self.frame_indices[int(unassigned[-1])])
            blockers.append(f"Progress rows are still unassigned between frames [{first}, {last}].")
            next_actions.append("Use edit_progress_segments to assign all remaining frame rows.")

        if blockers:
            return tool_blocked(
                blocked_items=blockers,
                next_required_actions=_dedupe_preserve_order(next_actions),
                data={
                    "stage_id": stage_id,
                    "progress": {
                        "curve_status": self._progress_curve_status(),
                        "assigned_count": int(np.sum(self.assigned_mask)),
                        "total_count": int(len(self.frame_indices)),
                        "progress_revision": int(self.progress_revision),
                        "validated_progress_revision": self.validated_progress_revision,
                        "summary_progress_revision": self.summary_progress_revision,
                    },
                    "confirmed_observation_coverage": coverage,
                },
            )

        if (
            not str(self.final_summary or "").strip()
            or self.external_review_need is None
            or (
                int(self.external_review_need) > 1
                and not str(self.external_review_focus or "").strip()
            )
        ):
            return tool_blocked(
                blocked_items=[
                    "Final summary and external review assessment have not been written for the "
                    "validated progress curve."
                ],
                next_required_actions=[
                    "Call write_summary with summary and external_review_need; include "
                    "external_review_focus when external_review_need is greater than 1."
                ],
                data={
                    "stage_id": stage_id,
                    "progress_revision": int(self.progress_revision),
                    "validated_progress_revision": self.validated_progress_revision,
                    "summary_progress_revision": self.summary_progress_revision,
                },
            )
        if self.summary_progress_revision != self.progress_revision:
            return tool_blocked(
                blocked_items=[
                    "Final summary is stale because progress was edited after the summary was written."
                ],
                next_required_actions=[
                    "Call write_summary again so the final summary describes the current validated progress curve."
                ],
                data={
                    "stage_id": stage_id,
                    "progress_revision": int(self.progress_revision),
                    "validated_progress_revision": self.validated_progress_revision,
                    "summary_progress_revision": self.summary_progress_revision,
                },
            )

        validation, trace_validation = self._progress_validation_result(csv_path=None)
        if not tool_result_ok(validation):
            validation_blockers = tool_result_blocked_items(validation) or ["Progress validation failed."]
            return tool_blocked(
                blocked_items=validation_blockers,
                next_required_actions=[
                    "Fix the progress curve with an appropriate progress-writing tool, "
                    "then call validate_progress again before write_summary."
                ],
                data={
                    "stage_id": stage_id,
                    "validation": validation,
                    "progress_revision": int(self.progress_revision),
                    "validated_progress_revision": self.validated_progress_revision,
                    "summary_progress_revision": self.summary_progress_revision,
                },
                warnings=tool_result_warnings(validation),
            )
        self.validated_progress_revision = int(self.progress_revision)
        tool_result_data(validation)["validated_progress_revision"] = int(self.validated_progress_revision)
        tool_result_data(trace_validation)["validated_progress_revision"] = int(self.validated_progress_revision)
        self.append_trace("complete_stage_progress_validation", trace_validation)

        export_data = self.export_corrected_progress()
        export_validation, trace_export_validation = self._progress_validation_result(
            csv_path=self.exported_progress_path
        )
        self.append_trace("complete_stage_export_validation", trace_export_validation)
        if not tool_result_ok(export_validation):
            validation_blockers = tool_result_blocked_items(export_validation) or [
                "Exported progress CSV validation failed."
            ]
            return tool_blocked(
                blocked_items=validation_blockers,
                next_required_actions=[
                    "Retry complete_stage so the final CSV is regenerated from the current progress curve."
                ],
                data={
                    "stage_id": stage_id,
                    "export": export_data,
                    "validation": validation,
                    "export_validation": export_validation,
                    "progress_revision": int(self.progress_revision),
                    "validated_progress_revision": self.validated_progress_revision,
                    "summary_progress_revision": self.summary_progress_revision,
                },
                warnings=tool_result_warnings(export_validation),
            )

        result = self.save_result(
            summary=str(self.final_summary),
            external_review_need=int(self.external_review_need),
            external_review_focus=self.external_review_focus,
            payload={
                "robot_instruction": self.robot_instruction,
                "progress_mode": self._progress_mode(),
                "candidate_progress_present": self.has_progress_proposal,
                "corrected_progress_exported": True,
                "progress_validation_ok": True,
                "num_observations": len(confirmed_observations),
            },
        )
        return tool_success(
            {
                "stage_id": stage_id,
                "export": export_data,
                "validation": validation,
                "export_validation": export_validation,
                "result": result,
            }
        )

    def _progress_mode(self) -> str:
        if self.has_progress_proposal:
            return "audit_and_correct_candidate_progress"
        return "generate_progress_from_visual_evidence"

    def _progress_curve_status(self) -> str:
        assigned_count = int(np.sum(self.assigned_mask))
        has_agent_revision = bool(self.progress_edit_history or self.progress_calibration_history)
        if self.has_progress_proposal:
            return "edited" if has_agent_revision else "proposal_loaded"
        if assigned_count == 0:
            return "empty"
        if assigned_count < len(self.assigned_mask):
            return "partial"
        return "generated" if has_agent_revision else "assigned"

    def _sample_progress_positions(
        self,
        *,
        start_frame: int,
        end_frame: int,
        sample_every_n: int,
        max_rows: int | None,
    ) -> list[int]:
        if start_frame > end_frame:
            raise ValueError(f"start_frame must be <= end_frame, got {start_frame} > {end_frame}.")
        step = max(1, int(sample_every_n))
        positions = np.flatnonzero((self.frame_indices >= start_frame) & (self.frame_indices <= end_frame))
        if len(positions) == 0:
            raise ValueError(f"No progress rows fall inside frame range [{start_frame}, {end_frame}].")
        sampled = positions[::step]
        if max_rows is not None and len(sampled) > int(max_rows):
            linspace_positions = np.linspace(0, len(sampled) - 1, num=max(1, int(max_rows)))
            sampled = np.asarray([sampled[int(round(position))] for position in linspace_positions])
        return [int(position) for position in sampled]

    def _progress_edit_spans(self, *, start_frame: int, end_frame: int) -> list[dict[str, Any]]:
        positions = np.flatnonzero((self.frame_indices >= start_frame) & (self.frame_indices <= end_frame))
        if len(positions) == 0:
            return []

        spans: list[dict[str, Any]] = []
        span_start_pos = int(positions[0])
        previous_pos = int(positions[0])
        previous_reason = self.edit_reasons[previous_pos]

        for pos in positions[1:]:
            current_pos = int(pos)
            current_reason = self.edit_reasons[current_pos]
            if current_reason != previous_reason:
                spans.append(
                    {
                        "start_frame": int(self.frame_indices[span_start_pos]),
                        "end_frame": int(self.frame_indices[previous_pos]),
                        "edit_reason": previous_reason,
                    }
                )
                span_start_pos = current_pos
                previous_reason = current_reason
            previous_pos = current_pos

        spans.append(
            {
                "start_frame": int(self.frame_indices[span_start_pos]),
                "end_frame": int(self.frame_indices[previous_pos]),
                "edit_reason": previous_reason,
            }
        )
        return spans

    def _progress_labels_for_frames(self, frame_indices: list[int]) -> dict[int, str]:
        """Return compact progress labels for rendered visual evidence."""

        labels: dict[int, str] = {}
        for frame_index in frame_indices:
            positions = np.flatnonzero(self.frame_indices == int(frame_index))
            if len(positions) != 1:
                continue
            pos = int(positions[0])
            current = (
                str(int(self.current_progress_points[pos]))
                if bool(self.assigned_mask[pos])
                else "unassigned"
            )
            if self.proposal_progress_points is None:
                labels[int(frame_index)] = f"progress {current}"
            else:
                proposal = int(self.proposal_progress_points[pos])
                labels[int(frame_index)] = f"proposal {proposal} | current {current}"
        return labels

    def _external_label_bar_options(self, media_mode: str) -> tuple[bool, bool]:
        """Return frame/progress label visibility for one inspect_visual artifact."""

        label_config = self.config.visual.external_label_bar
        if not label_config.enabled:
            return False, False
        if media_mode == "video":
            return label_config.video.frame, label_config.video.progress
        if media_mode == "tile":
            return label_config.tile.frame, label_config.tile.progress
        raise ValueError(f"Unsupported media mode for label bar options: {media_mode!r}")

    def _mark_progress_assigned(self, *, start_frame: int, end_frame: int, propagation: PropagationMode) -> None:
        segment_mask = (self.frame_indices >= int(start_frame)) & (self.frame_indices <= int(end_frame))
        self.assigned_mask[segment_mask] = True
        if propagation == "shift_after":
            self.assigned_mask[self.frame_indices > int(end_frame)] = True

    def _save_state(self) -> None:
        state = {
            "schema_version": 7,
            "case_id": self.case_id,
            "case_dir": str(self.case_dir),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "base_robot_instruction": self.base_robot_instruction,
            "robot_instruction": self.robot_instruction,
            "task_context": dict(self.task_context),
            "video_path": str(self.video_path) if self.video_path else None,
            "has_progress_proposal": self.has_progress_proposal,
            "current_progress_points": self.current_progress_points.astype(int).tolist(),
            "assigned_mask": self.assigned_mask.astype(bool).tolist(),
            "edit_reasons": list(self.edit_reasons),
            "progress_edit_history": [record.to_dict() for record in self.progress_edit_history],
            "progress_calibration_history": [
                record.to_dict() for record in self.progress_calibration_history
            ],
            "exported_progress_path": str(self.exported_progress_path) if self.exported_progress_path else None,
            "progress_plot_path": str(self.progress_plot_path) if self.progress_plot_path else None,
            "progress_review_video_path": str(self.progress_review_video_path)
            if self.progress_review_video_path
            else None,
            "nonfatal_artifact_warnings": [
                dict(warning) for warning in self.nonfatal_artifact_warnings
            ],
            "progress_revision": int(self.progress_revision),
            "validated_progress_revision": self.validated_progress_revision,
            "summary_progress_revision": self.summary_progress_revision,
            "visual_assets": [self._asset_to_state(asset) for asset in self.visual_assets],
            "observations": [observation.to_dict() for observation in self.observations],
            "observation_revision_history": [record.to_dict() for record in self.observation_revision_history],
            "tool_history": [record.to_dict() for record in self.tool_history],
            "final_summary": self.final_summary,
            "external_review_need": self.external_review_need,
            "external_review_focus": self.external_review_focus,
            "final_result": self.final_result.to_dict() if self.final_result else None,
        }
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(_json_ready(state), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_path.replace(self.state_path)

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.base_robot_instruction = str(state["base_robot_instruction"])
        self.robot_instruction = str(state["robot_instruction"])
        self.task_context = dict(state["task_context"])
        current = np.asarray(state["current_progress_points"], dtype=np.int64)
        if len(current) != len(self.current_progress_points):
            raise ValueError("Session state progress length does not match current case progress length.")
        self.current_progress_points = current
        assigned_mask = np.asarray(state["assigned_mask"], dtype=bool)
        if len(assigned_mask) != len(self.current_progress_points):
            raise ValueError("Session state assigned_mask length does not match progress length.")
        self.assigned_mask = assigned_mask
        self.edit_reasons = [str(reason) for reason in state["edit_reasons"]]
        if len(self.edit_reasons) != len(self.current_progress_points):
            raise ValueError("Session state edit_reasons length does not match progress length.")
        self.progress_edit_history = [
            ProgressEditRecord(
                edit_id=str(record["edit_id"]),
                start_frame=int(record["start_frame"]),
                end_frame=int(record["end_frame"]),
                start_progress_points=int(record["start_progress_points"]),
                end_progress_points=int(record["end_progress_points"]),
                trend=record["trend"],
                propagation=record["propagation"],
                reason=str(record["reason"]),
                confidence=int(record["confidence"]),
                affected_frames=int(record["affected_frames"]),
            )
            for record in state.get("progress_edit_history", [])
        ]
        self.progress_calibration_history = [
            ProgressCalibrationRecord(
                calibration_id=str(record["calibration_id"]),
                anchor_a_frame=int(record["anchor_a_frame"]),
                anchor_a_old_progress_points=int(record["anchor_a_old_progress_points"]),
                anchor_a_target_progress_points=int(record["anchor_a_target_progress_points"]),
                anchor_b_frame=int(record["anchor_b_frame"]),
                anchor_b_old_progress_points=int(record["anchor_b_old_progress_points"]),
                anchor_b_target_progress_points=int(record["anchor_b_target_progress_points"]),
                scale=float(record["scale"]),
                offset=float(record["offset"]),
                old_start_progress_points=int(record["old_start_progress_points"]),
                old_end_progress_points=int(record["old_end_progress_points"]),
                old_min_progress_points=int(record["old_min_progress_points"]),
                old_max_progress_points=int(record["old_max_progress_points"]),
                new_start_progress_points=int(record["new_start_progress_points"]),
                new_end_progress_points=int(record["new_end_progress_points"]),
                new_min_progress_points=int(record["new_min_progress_points"]),
                new_max_progress_points=int(record["new_max_progress_points"]),
                reason=str(record["reason"]),
                confidence=int(record["confidence"]),
                affected_frames=int(record["affected_frames"]),
            )
            for record in state.get("progress_calibration_history", [])
        ]
        exported_progress_path = state.get("exported_progress_path")
        self.exported_progress_path = Path(exported_progress_path) if exported_progress_path else None
        progress_plot_path = state.get("progress_plot_path")
        self.progress_plot_path = Path(progress_plot_path) if progress_plot_path else None
        progress_review_video_path = state.get("progress_review_video_path")
        self.progress_review_video_path = Path(progress_review_video_path) if progress_review_video_path else None
        self.nonfatal_artifact_warnings = [
            dict(warning) for warning in state.get("nonfatal_artifact_warnings", [])
        ]
        self.progress_revision = int(state["progress_revision"])
        raw_validated_progress_revision = state["validated_progress_revision"]
        self.validated_progress_revision = (
            None if raw_validated_progress_revision is None else int(raw_validated_progress_revision)
        )
        raw_summary_progress_revision = state["summary_progress_revision"]
        self.summary_progress_revision = (
            None if raw_summary_progress_revision is None else int(raw_summary_progress_revision)
        )
        self.visual_assets = [
            VisualAsset(
                asset_id=str(asset["asset_id"]),
                kind=asset["kind"],
                path=self._path_from_state(asset["path"]),
                frame_range=(int(asset["frame_range"][0]), int(asset["frame_range"][1])),
                frame_indices=[int(frame) for frame in asset["frame_indices"]],
                sample_every_n=int(asset["sample_every_n"]),
                attached_frame_count=int(asset["attached_frame_count"]),
                reason=str(asset.get("reason", "")),
                crop=asset.get("crop"),
                created_stage_id=asset.get("created_stage_id"),
                progress_label_revision=(
                    None
                    if asset.get("progress_label_revision") is None
                    else int(asset["progress_label_revision"])
                ),
            )
            for asset in state.get("visual_assets", [])
        ]
        self.observations = [
            ObservationRecord(
                observation_id=str(observation["observation_id"]),
                asset_ids=[str(asset_id) for asset_id in observation["asset_ids"]],
                start_frame=int(observation["start_frame"]),
                end_frame=int(observation["end_frame"]),
                text=str(observation["text"]),
                confidence=int(observation["confidence"]),
                status=observation.get("status", "draft"),
                created_stage_id=observation.get("created_stage_id"),
                updated_stage_id=observation.get("updated_stage_id"),
                confirmed_stage_id=observation.get("confirmed_stage_id"),
                confirmation_asset_ids=[
                    str(asset_id) for asset_id in observation.get("confirmation_asset_ids", [])
                ],
                confirmation_reason=str(observation.get("confirmation_reason", "")),
                revision_count=int(observation.get("revision_count", 0)),
            )
            for observation in state.get("observations", [])
        ]
        self.observation_revision_history = [
            ObservationRevisionRecord(
                revision_id=str(record["revision_id"]),
                observation_id=str(record["observation_id"]),
                stage_id=record.get("stage_id"),
                reason=str(record["reason"]),
                old_observation=dict(record["old_observation"]),
                new_observation=dict(record["new_observation"]),
            )
            for record in state.get("observation_revision_history", [])
        ]
        self.tool_history = [
            ToolCallRecord(
                tool_call_id=str(record["tool_call_id"]),
                name=str(record["name"]),
                arguments=dict(record.get("arguments", {})),
                result=dict(record.get("result", {})),
            )
            for record in state.get("tool_history", [])
        ]
        self.final_summary = state.get("final_summary")
        raw_external_review_need = state["external_review_need"]
        self.external_review_need = (
            None if raw_external_review_need is None else int(raw_external_review_need)
        )
        self.external_review_focus = state["external_review_focus"]
        final_result = state.get("final_result")
        if final_result:
            raw_external_review_focus = final_result.get("external_review_focus")
            self.final_result = FinalResult(
                summary=str(final_result["summary"]),
                external_review_need=int(final_result["external_review_need"]),
                external_review_focus=(
                    None
                    if raw_external_review_focus is None
                    else str(raw_external_review_focus)
                ),
                payload=dict(final_result.get("payload", {})),
            )

    def _asset_to_state(self, asset: VisualAsset) -> dict[str, Any]:
        data = asset.to_dict(include_path=True)
        data["path"] = self._path_to_state(asset.path)
        return data

    def _path_to_state(self, path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return str(resolved.relative_to(self.run_dir))
        except ValueError:
            return str(resolved)

    def _path_from_state(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (self.run_dir / path).resolve()

    def _manifest(
        self,
        *,
        final_result_path: Path | None = None,
        final_summary_path: Path | None = None,
    ) -> dict[str, Any]:
        workflow = self.config.workflow
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "model": self.config.vllm.model,
            "base_url": self.config.vllm.base_url,
            "context_window": self.config.vllm.context_window,
            "generation": self.config.generation.to_request_payload(),
            "external_review_need": self.external_review_need,
            # Persist the resolved condition used by this run in one canonical
            # result artifact: true/false means enabled, while null means disabled.
            "is_success_ground_truth": self.is_success_ground_truth,
            "progress_config": {
                "value_min_points": 0,
                "value_max_points": int(self.config.progress.value_max_points),
                "default_jump_threshold": int(self.config.progress.default_jump_threshold_points),
            },
            "visual": {
                "media_mode": self.config.visual.media_mode,
                "max_attached_frames_per_call": self.config.visual.max_attached_frames_per_call,
                "agent_video_fps": self.config.visual.agent_video_fps,
                "review_video_fps": self.config.visual.review_video_fps,
                "external_label_bar": asdict(self.config.visual.external_label_bar),
            },
            "task_context": dict(self.task_context),
            "reference_examples": [
                example.to_manifest_dict(run_dir=self.run_dir)
                for example in self.reference_examples
            ],
            "workflow": None
            if workflow is None
            else {
                "name": workflow.name,
                "source_path": workflow.source_path,
                "prompt_hashes": workflow.prompt_hashes,
                "stages": [
                        {
                            "stage_id": stage.stage_id,
                            "prompt": stage.prompt,
                            "max_steps": stage.max_steps,
                            "fresh_session": stage.fresh_session,
                            "allowed_tools": list(stage.allowed_tools),
                            "initial_media": dict(stage.initial_media),
                            "initial_context": dict(stage.initial_context),
                            "tool_config": dict(stage.tool_config),
                            "generation": dict(stage.generation),
                            "promote_observations_to_confirmed_on_complete": (
                                stage.promote_observations_to_confirmed_on_complete
                            ),
                        }
                        for stage in workflow.stages
                    ],
            },
            "visual_assets": [asset.to_dict(include_path=False) for asset in self.visual_assets],
            "observations": [observation.to_dict() for observation in self.observations],
            "progress": {
                "has_progress_proposal": self.has_progress_proposal,
                "mode": self._progress_mode(),
                "curve_status": self._progress_curve_status(),
                "num_rows": int(len(self.frame_indices)),
                "num_progress_edits": len(self.progress_edit_history),
                "num_progress_calibrations": len(self.progress_calibration_history),
                "progress_revision": int(self.progress_revision),
                "validated_progress_revision": self.validated_progress_revision,
                "summary_progress_revision": self.summary_progress_revision,
                "corrected_progress_csv": self._path_to_state(self.exported_progress_path)
                if self.exported_progress_path
                else None,
                "corrected_progress_plot": self._path_to_state(self.progress_plot_path)
                if self.progress_plot_path
                else None,
                "corrected_progress_review_video": self._path_to_state(self.progress_review_video_path)
                if self.progress_review_video_path
                else None,
            },
            "nonfatal_artifact_warnings": [
                dict(warning) for warning in self.nonfatal_artifact_warnings
            ],
            "num_observation_revisions": len(self.observation_revision_history),
            "final_result_path": self._path_to_state(final_result_path) if final_result_path else None,
            "final_summary_path": self._path_to_state(final_summary_path) if final_summary_path else None,
        }

    @staticmethod
    def _validate_confidence(confidence: int) -> None:
        if not isinstance(confidence, int) or isinstance(confidence, bool):
            raise TypeError(f"confidence must be an integer in [1, 5], got {confidence!r}.")
        if not 1 <= confidence <= 5:
            raise ValueError(f"confidence must be an integer in [1, 5], got {confidence}.")


def _inspect_visual_policy(tool_config: Any) -> dict[str, Any]:
    """Normalize stage-local inspect_visual policy into runtime decisions."""

    config = dict(tool_config or {}) if isinstance(tool_config, dict) else {}
    mode = str(config.get("mode", "model")).strip()
    if mode not in {"model", "video", "tile"}:
        mode = "model"

    raw_sample_every_n = config.get("sample_every_n", "model")
    if str(raw_sample_every_n).strip() == "model":
        sample_every_n: int | str = "model"
    else:
        sample_every_n = max(1, int(raw_sample_every_n))

    crop = str(config.get("crop", "enabled")).strip()
    if crop not in {"enabled", "disabled"}:
        crop = "enabled"
    return {
        "mode": mode,
        "sample_every_n": sample_every_n,
        "crop": crop,
    }


def _inspect_visual_tool_definition(tool_config: dict[str, Any] | None) -> dict[str, Any]:
    """Build the current stage's inspect_visual schema from tool_config."""

    policy = _inspect_visual_policy((tool_config or {}).get("inspect_visual"))
    properties: dict[str, Any] = {
        "start_frame": {"type": "integer"},
        "end_frame": {"type": "integer"},
        "reason": {"type": "string"},
    }
    if policy["sample_every_n"] == "model":
        properties["sample_every_n"] = {"type": "integer", "default": 1}
    if policy["mode"] == "model":
        properties["mode"] = {"type": "string", "enum": ["video", "tile"]}
    if policy["crop"] == "enabled":
        properties["crop"] = {
            "type": "object",
            "description": "Optional focused crop in integer percentages of the full frame.",
            "properties": {
                "left_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "top_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "right_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "bottom_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["left_pct", "top_pct", "right_pct", "bottom_pct"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": "inspect_visual",
            "description": "Request bounded visual evidence from a frame interval.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["start_frame", "end_frame", "reason"],
                "additionalProperties": False,
            },
        },
    }


def _tool_definitions(
    *,
    value_max_points: int,
    default_jump_threshold: int,
    tool_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return tool schemas with the configured progress value space embedded."""

    max_points = int(value_max_points)
    inspect_visual_tool = _inspect_visual_tool_definition(tool_config)
    return [
        {
            "type": "function",
            "function": {
                "name": "read_robot_instruction",
                "description": (
                    "Read the robot instruction plus any reward-agent-only appended context. "
                    "This text is not visual evidence."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_metadata",
                "description": "Read public case metadata.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_progress",
                "description": "Read candidate and current progress values by frame index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "sample_every_n": {"type": "integer", "default": 1},
                        "max_rows": {"type": "integer"},
                        "view": {"type": "string", "enum": ["proposal", "current", "both"], "default": "both"},
                        "include_edit_spans": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_progress",
                "description": "Summarize progress values and jump windows for visual inspection planning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_k": {"type": "integer", "default": 8},
                        "jump_threshold": {"type": "integer", "default": int(default_jump_threshold)},
                    },
                    "additionalProperties": False,
                },
            },
        },
        inspect_visual_tool,
        {
            "type": "function",
            "function": {
                "name": "edit_progress_segments",
                "description": "Edit or generate one or more local current-progress segments by linear interpolation between typed endpoint progress values. Segments must be ordered, may share at most one boundary frame, and shared boundaries must use the same progress value on both sides.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segments": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_frame": {"type": "integer"},
                                    "end_frame": {"type": "integer"},
                                    "start_progress_points": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": max_points,
                                        "description": "Progress value at start_frame. Equal endpoints create a plateau; lower end value creates a regression. For adjacent piecewise segments, this should match the previous segment's end_progress_points when start_frame equals the previous end_frame.",
                                    },
                                    "end_progress_points": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": max_points,
                                        "description": "Progress value at end_frame. The tool linearly interpolates inclusive frame rows between endpoints; adjacent segments may reuse this exact end_frame as the next segment's start_frame. The returned edit record automatically includes trend='increase', 'flat', or 'decrease' from the endpoint direction; do not pass trend as an input.",
                                    },
                                    "propagation": {"type": "string", "enum": ["none"], "default": "none"},
                                    "reason": {"type": "string"},
                                    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                                },
                                "required": [
                                    "start_frame",
                                    "end_frame",
                                    "start_progress_points",
                                    "end_progress_points",
                                    "reason",
                                    "confidence",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["segments"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calibrate_progress_curve",
                "description": (
                    "Apply one whole-trajectory affine calibration to assigned current-progress rows "
                    "using two explicit frame/value anchors. Use this when the current curve's relative "
                    "trend is credible but two key frames should map to clearer semantic progress values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor_a_frame": {
                            "type": "integer",
                            "description": "First anchor frame. It must exist in the progress frame axis and already be assigned.",
                        },
                        "anchor_a_progress_points": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max_points,
                            "description": "Target progress value for anchor_a_frame after calibration.",
                        },
                        "anchor_b_frame": {
                            "type": "integer",
                            "description": "Second anchor frame. It must exist in the progress frame axis, already be assigned, and have a distinct current progress value from anchor_a_frame.",
                        },
                        "anchor_b_progress_points": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max_points,
                            "description": "Target progress value for anchor_b_frame after calibration.",
                        },
                        "reason": {"type": "string"},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": [
                        "anchor_a_frame",
                        "anchor_a_progress_points",
                        "anchor_b_frame",
                        "anchor_b_progress_points",
                        "reason",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_progress",
                "description": "Validate the current in-memory progress curve after edits and before writing the final summary. Takes no arguments.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_summary",
                "description": (
                    "Write the final narrative summary and external-review assessment after "
                    "validate_progress succeeds; the tool also validates the current progress curve "
                    "before accepting the assessment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "external_review_need": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": (
                                "How much this result needs independent external review. Base this on "
                                "the limits of your own ability and how difficult the video was for you "
                                "to understand, not on whether the trajectory succeeded or failed. "
                                "1 means no review is needed; 5 means review is strongly needed."
                            ),
                        },
                        "external_review_focus": {
                            "type": "string",
                            "description": (
                                "When external_review_need is greater than 1, briefly state what the "
                                "external reviewer should check. Omit this field when no review is needed."
                            ),
                        },
                    },
                    "required": ["summary", "external_review_need"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_observations",
                "description": "Record one or more closed-interval visual observations tied to generated asset ids. Cited asset_ids must fully cover each submitted closed frame interval. Adjacent observations may share exactly one boundary frame, but broader overlaps are invalid. Choose boundaries from visible task or state transitions, not because they are numerically convenient, round, tidy, evenly spaced, or inherited from a coarse proposal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "asset_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                        "description": "Visual asset ids whose combined frame ranges fully cover this observation's closed interval.",
                                    },
                                    "start_frame": {
                                        "type": "integer",
                                        "description": "First frame in the closed event interval described by this observation. It must be covered by the cited asset_ids. It may equal the previous observation's end_frame only when sharing a single semantic boundary frame. Choose the best visually supported transition frame; do not choose it because it is round, tidy, evenly spaced, or a common grid-like value such as a value ending in 0 or 5.",
                                    },
                                    "end_frame": {
                                        "type": "integer",
                                        "description": "Last frame in the closed event interval described by this observation. It must be covered by the cited asset_ids. It may equal the next observation's start_frame only when sharing a single semantic boundary frame. Choose the best visually supported transition frame; do not choose it because it is round, tidy, evenly spaced, or a common grid-like value such as a value ending in 0 or 5.",
                                    },
                                    "text": {"type": "string"},
                                    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                                },
                                "required": ["asset_ids", "start_frame", "end_frame", "text", "confidence"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["observations"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_observations",
                "description": "Read current observations and deterministic coverage diagnostics.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_observations",
                "description": "Confirm one or more current observations with new visual evidence created by inspect_visual during the challenge_observations stage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "observation_id": {"type": "string"},
                                    "asset_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                    "reason": {"type": "string"},
                                },
                                "required": ["observation_id", "asset_ids", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["confirmations"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "merge_observations",
                "description": "Merge two or more active observations into one draft observation when no meaningful task-progress boundary separates them. Selected observations may overlap, but their closed-frame union must be continuous and must not contain any unselected active observation. Cited asset_ids must fully cover the merged closed frame interval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "observation_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "description": "Current active observation ids to merge. The first id is kept as the merged observation id; all other selected ids become inactive. Include every active observation inside the merged frame union.",
                        },
                        "asset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Visual asset ids whose combined frame ranges fully cover the merged closed interval.",
                        },
                        "merged_text": {"type": "string"},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {"type": "string"},
                    },
                    "required": ["observation_ids", "asset_ids", "merged_text", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "revise_observations",
                "description": "Revise one or more closed-interval visual observations in place and reset each revised observation to draft. Cited asset_ids must fully cover each submitted closed frame interval. Adjacent observations may share exactly one boundary frame, but broader overlaps are invalid. Use revisions to replace coarse, arbitrary, tidy, evenly spaced, or proposal-inherited boundaries with evidence-supported visual transition frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "revisions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "observation_id": {"type": "string"},
                                    "asset_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                        "description": "Visual asset ids whose combined frame ranges fully cover this revised closed interval.",
                                    },
                                    "start_frame": {
                                        "type": "integer",
                                        "description": "First frame in the revised closed event interval. It must be covered by the cited asset_ids. It may equal the previous observation's end_frame only when sharing a single semantic boundary frame. Choose the best visually supported transition frame; do not preserve or choose a boundary because it is round, tidy, evenly spaced, proposal-inherited, or a common grid-like value such as a value ending in 0 or 5.",
                                    },
                                    "end_frame": {
                                        "type": "integer",
                                        "description": "Last frame in the revised closed event interval. It must be covered by the cited asset_ids. It may equal the next observation's start_frame only when sharing a single semantic boundary frame. Choose the best visually supported transition frame; do not preserve or choose a boundary because it is round, tidy, evenly spaced, proposal-inherited, or a common grid-like value such as a value ending in 0 or 5.",
                                    },
                                    "text": {"type": "string"},
                                    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "observation_id",
                                    "asset_ids",
                                    "start_frame",
                                    "end_frame",
                                    "text",
                                    "confidence",
                                    "reason",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["revisions"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "complete_stage",
                "description": "Attempt to complete the current workflow stage using deterministic gate checks.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _normalize_enum_string(value: Any) -> Any:
    """Clean common model-produced enum string artifacts before validation."""

    if not isinstance(value, str):
        return value
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1].strip()
    return normalized


def _is_blank_tool_arg(value: Any) -> bool:
    """Return whether a model-supplied tool argument is effectively omitted."""

    return value is None or (isinstance(value, str) and not value.strip())


def _is_observation_stage(stage_id: str | None) -> bool:
    """Return whether progress labels should stay hidden for observation work."""

    return str(stage_id or "").strip() in {
        "bootstrap_observations",
        "challenge_observations",
    }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicate user-facing hints while preserving their first order."""

    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _required_int_tool_arg(name: str, value: Any, *, tool_name: str) -> int:
    """Parse a required integer tool argument with an agent-readable error."""

    if _is_blank_tool_arg(value):
        raise ValueError(f"{tool_name}.{name} must be an integer frame index; do not pass an empty value.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool_name}.{name} must be an integer frame index, got {value!r}.") from exc


def _required_progress_point_tool_arg(
    name: str,
    value: Any,
    *,
    tool_name: str,
    value_max_points: int,
) -> int:
    """Parse a required integer progress-point argument inside the configured range."""

    if _is_blank_tool_arg(value):
        raise ValueError(f"{tool_name}.{name} must be an integer progress value; do not pass an empty value.")
    if isinstance(value, bool):
        raise ValueError(f"{tool_name}.{name} must be an integer progress value, got {value!r}.")
    try:
        point = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool_name}.{name} must be an integer progress value, got {value!r}.") from exc
    max_points = int(value_max_points)
    if not 0 <= point <= max_points:
        raise ValueError(f"{tool_name}.{name} must be in [0, {max_points}], got {point}.")
    return point


def _optional_int_tool_arg(name: str, value: Any, *, default: int, tool_name: str) -> int:
    """Parse an optional integer tool argument, treating blank strings as omitted."""

    if _is_blank_tool_arg(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool_name}.{name} must be an integer when provided, got {value!r}.") from exc


def _required_str_tool_arg(name: str, value: Any, *, tool_name: str) -> str:
    """Parse a required non-empty string tool argument."""

    if _is_blank_tool_arg(value):
        raise ValueError(f"{tool_name}.{name} must be a non-empty string; do not pass an empty value.")
    return str(value).strip()


def _required_asset_id_list(name: str, value: Any, *, tool_name: str) -> list[str]:
    """Parse a required non-empty list of visual asset ids."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{tool_name}.{name} must be a non-empty list of asset id strings.")
    parsed = [str(asset_id).strip() for asset_id in value if str(asset_id).strip()]
    if len(parsed) != len(value):
        raise ValueError(f"{tool_name}.{name} must contain only non-empty asset id strings.")
    return parsed


def _required_confidence_tool_arg(name: str, value: Any, *, tool_name: str) -> int:
    """Parse a required integer confidence argument and reuse the session validator."""

    if _is_blank_tool_arg(value):
        raise ValueError(f"{tool_name}.{name} must be an integer confidence in [1, 5].")
    if isinstance(value, bool):
        raise TypeError(f"{tool_name}.{name} must be an integer confidence in [1, 5], got {value!r}.")
    if not isinstance(value, int):
        raise TypeError(f"{tool_name}.{name} must be an integer confidence in [1, 5], got {value!r}.")
    confidence = int(value)
    RewardAgentSession._validate_confidence(confidence)
    return confidence


def _required_external_review_need_tool_arg(
    name: str,
    value: Any,
    *,
    tool_name: str,
) -> int:
    """Parse the required 1-5 external-review routing signal."""

    if _is_blank_tool_arg(value):
        raise ValueError(f"{tool_name}.{name} must be an integer in [1, 5].")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{tool_name}.{name} must be an integer in [1, 5], got {value!r}.")
    review_need = int(value)
    if not 1 <= review_need <= 5:
        raise ValueError(f"{tool_name}.{name} must be an integer in [1, 5], got {review_need}.")
    return review_need


def _optional_external_review_focus_tool_arg(
    name: str,
    value: Any,
    *,
    external_review_need: int,
    tool_name: str,
) -> str | None:
    """Require an actionable review focus only when external review is needed."""

    if _is_blank_tool_arg(value):
        if int(external_review_need) == 1:
            return None
        raise ValueError(
            f"{tool_name}.{name} must state what the reviewer should check when "
            "external_review_need is greater than 1."
        )
    focus = str(value).strip()
    if focus.casefold().rstrip(".") in {"none", "n/a", "na", "null"}:
        raise ValueError(
            f"{tool_name}.{name} must state what the reviewer should check; "
            "placeholder values such as None or N/A are not allowed."
        )
    return focus


def _optional_crop_tool_arg(name: str, value: Any, *, tool_name: str) -> dict[str, int] | None:
    """Parse an optional integer-percent crop object for visual inspection."""

    if _is_blank_tool_arg(value):
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"{tool_name}.{name} must be an object with left_pct, top_pct, right_pct, and bottom_pct."
        )

    required = ("left_pct", "top_pct", "right_pct", "bottom_pct")
    keys = set(value)
    missing = [key for key in required if key not in keys]
    extra = sorted(str(key) for key in keys.difference(required))
    if missing:
        raise ValueError(f"{tool_name}.{name} is missing required field(s): {', '.join(missing)}.")
    if extra:
        raise ValueError(f"{tool_name}.{name} has unsupported field(s): {', '.join(extra)}.")

    crop: dict[str, int] = {}
    for key in required:
        raw_value = value[key]
        if isinstance(raw_value, bool):
            raise ValueError(f"{tool_name}.{name}.{key} must be an integer percentage, got {raw_value!r}.")
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{tool_name}.{name}.{key} must be an integer percentage, got {raw_value!r}."
            ) from exc
        if not parsed.is_integer():
            raise ValueError(f"{tool_name}.{name}.{key} must be an integer percentage, got {raw_value!r}.")
        crop[key] = int(parsed)

    if not (0 <= crop["left_pct"] < crop["right_pct"] <= 100):
        raise ValueError(f"{tool_name}.{name} must satisfy 0 <= left_pct < right_pct <= 100.")
    if not (0 <= crop["top_pct"] < crop["bottom_pct"] <= 100):
        raise ValueError(f"{tool_name}.{name} must satisfy 0 <= top_pct < bottom_pct <= 100.")
    return crop


def _optional_enum_tool_arg(
    name: str,
    value: Any,
    *,
    allowed: set[str],
    default: str,
    tool_name: str,
) -> str:
    """Parse an optional enum string with common model quoting artifacts removed."""

    if _is_blank_tool_arg(value):
        return default
    normalized = _normalize_enum_string(value)
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(repr(item) for item in allowed))
        raise ValueError(f"{tool_name}.{name} must be one of {allowed_text}, got {value!r}.")
    return str(normalized)


def _positive_int_or_none(value: Any) -> int | None:
    """Convert a metadata value into a positive integer when possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return int(numeric)


def _positive_float_or_none(value: Any) -> float | None:
    """Convert a metadata value into a positive finite float when possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric
