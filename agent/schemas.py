"""Typed records shared by the reward-agent harness, tools, and runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


MediaMode = Literal["video", "tile"]
ResetServerCacheMode = Literal["never", "before-run", "before-case"]
ProgressView = Literal["proposal", "current", "both"]
PropagationMode = Literal["none", "shift_after"]
ProgressTrend = Literal["increase", "flat", "decrease"]
ObservationStatus = Literal["draft", "confirmed", "inactive"]


@dataclass(frozen=True, slots=True)
class ProgressRow:
    """One candidate progress row loaded from the internal case pack."""

    frame_index: int
    progress_points: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressSummary:
    """Compact numeric summary used to guide reward-agent visual inspection."""

    num_rows: int
    start_frame: int
    end_frame: int
    start_progress_points: int
    end_progress_points: int
    min_progress_points: int
    max_progress_points: int
    top_delta_windows: list[dict[str, int]]
    jump_points: list[dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressEditRecord:
    """Audit record for one progress edit or generated progress segment."""

    edit_id: str
    start_frame: int
    end_frame: int
    start_progress_points: int
    end_progress_points: int
    trend: ProgressTrend
    propagation: PropagationMode
    reason: str
    confidence: int
    affected_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressCalibrationRecord:
    """Audit record for one trajectory-level affine progress calibration."""

    calibration_id: str
    anchor_a_frame: int
    anchor_a_old_progress_points: int
    anchor_a_target_progress_points: int
    anchor_b_frame: int
    anchor_b_old_progress_points: int
    anchor_b_target_progress_points: int
    scale: float
    offset: float
    old_start_progress_points: int
    old_end_progress_points: int
    old_min_progress_points: int
    old_max_progress_points: int
    new_start_progress_points: int
    new_end_progress_points: int
    new_min_progress_points: int
    new_max_progress_points: int
    reason: str
    confidence: int
    affected_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressValidationResult:
    """Result of checking the in-memory or exported reward progress curve."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    csv_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["csv_path"] = str(self.csv_path) if self.csv_path is not None else None
        return data


@dataclass(frozen=True, slots=True)
class VisualAsset:
    """One bounded visual evidence artifact generated from the hidden raw video."""

    asset_id: str
    kind: Literal["video_clip", "tile"]
    path: Path
    frame_range: tuple[int, int]
    frame_indices: list[int]
    sample_every_n: int
    attached_frame_count: int
    reason: str
    crop: dict[str, int] | None = None
    created_stage_id: str | None = None
    progress_label_revision: int | None = None

    def to_dict(self, *, include_path: bool = True) -> dict[str, Any]:
        data = {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "frame_range": [int(self.frame_range[0]), int(self.frame_range[1])],
            "frame_indices": [int(frame) for frame in self.frame_indices],
            "sample_every_n": int(self.sample_every_n),
            "attached_frame_count": int(self.attached_frame_count),
            "reason": self.reason,
            "crop": dict(self.crop) if self.crop is not None else None,
            "created_stage_id": self.created_stage_id,
            "progress_label_revision": self.progress_label_revision,
        }
        if include_path:
            data["path"] = str(self.path)
        return data

    def to_agent_dict(self) -> dict[str, Any]:
        """Return the path-redacted metadata exposed to the model."""

        data = self.to_dict(include_path=False)
        data["message"] = "The requested visual evidence will be attached in the next model turn."
        return data


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """Agent-authored observation grounded in generated visual assets."""

    observation_id: str
    asset_ids: list[str]
    start_frame: int
    end_frame: int
    text: str
    confidence: int
    status: ObservationStatus = "draft"
    created_stage_id: str | None = None
    updated_stage_id: str | None = None
    confirmed_stage_id: str | None = None
    confirmation_asset_ids: list[str] = field(default_factory=list)
    confirmation_reason: str = ""
    revision_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationRevisionRecord:
    """Internal version-history entry for one observation revision."""

    revision_id: str
    observation_id: str
    stage_id: str | None
    reason: str
    old_observation: dict[str, Any]
    new_observation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Persisted record for one model-requested tool call."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Token accounting reported by the model server for one chat response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_window: int = 65536
    reserved_output_tokens: int = 2048

    @property
    def remaining_tokens(self) -> int:
        return int(self.context_window) - int(self.total_tokens) - int(self.reserved_output_tokens)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["remaining_tokens"] = self.remaining_tokens
        return data

    @classmethod
    def from_response_usage(
        cls,
        usage: dict[str, Any] | None,
        *,
        context_window: int,
        reserved_output_tokens: int,
    ) -> "UsageRecord":
        usage = usage or {}
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            context_window=int(context_window),
            reserved_output_tokens=int(reserved_output_tokens),
        )


@dataclass(frozen=True, slots=True)
class PendingMediaAttachment:
    """Internal pointer to a generated asset that should be attached next turn."""

    asset_id: str
    kind: Literal["video_clip", "tile"]
    path: Path
    frame_indices: list[int]
    frame_range: tuple[int, int] | None = None
    sample_every_n: int | None = None

    def to_message_content(self) -> list[dict[str, Any]]:
        summary = {
            "type": "text",
            "text": (
                "Attached visual evidence: "
                f"asset_id={self.asset_id}, kind={self.kind}, "
                f"{_frame_indices_summary(self.asset_id, self.frame_indices, self.frame_range, self.sample_every_n)}."
            ),
        }
        if self.kind == "video_clip":
            media = {
                "type": "video_url",
                "video_url": {"url": self.path.resolve().as_uri()},
            }
        else:
            media = {
                "type": "image_url",
                "image_url": {"url": self.path.resolve().as_uri()},
            }
        return [summary, media]


def _frame_indices_summary(
    asset_id: str,
    frame_indices: list[int],
    frame_range: tuple[int, int] | None,
    sample_every_n: int | None,
) -> str:
    """Summarize attached frames without hiding sparse sampling."""

    frames = [int(frame) for frame in frame_indices]
    if len(frames) <= 24:
        return f"frames={frames}"
    if asset_id == "full_video":
        return f"frames={frames[0]}-{frames[-1]} (total={len(frames)})"

    range_text = (
        f"frame_range={int(frame_range[0])}-{int(frame_range[1])}"
        if frame_range is not None
        else f"frame_range={frames[0]}-{frames[-1]}"
    )
    stride_text = f", sample_every_n={int(sample_every_n)}" if sample_every_n is not None else ""
    return (
        f"{range_text}{stride_text}, sampled_frames="
        f"{frames[:8]} ... {frames[-4:]} (total={len(frames)})"
    )


@dataclass(frozen=True, slots=True)
class FinalResult:
    """Final structured result saved by the reward agent."""

    summary: str
    external_review_need: int
    external_review_focus: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "external_review_need": int(self.external_review_need),
            "external_review_focus": self.external_review_focus,
            "payload": dict(self.payload),
        }
