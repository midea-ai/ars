"""Reward-agent progress-curve loading, editing, export, validation, and plots."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .schemas import (
    ProgressCalibrationRecord,
    ProgressEditRecord,
    ProgressRow,
    ProgressSummary,
    ProgressValidationResult,
    ProgressTrend,
    PropagationMode,
)


REQUIRED_PROGRESS_COLUMNS = ("frame_index", "progress_points", "source")
EXPORTED_PROGRESS_COLUMNS = (
    "frame_index",
    "proposal_progress_points",
    "current_progress_points",
    "current_progress_normalized",
    "is_model_output",
    "edit_reason",
    "source",
)


class ProgressCalibrationError(ValueError):
    """Expected blocker for an affine progress calibration request."""

    def __init__(self, message: str, *, data: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.data = dict(data or {})


def build_progress_sample_indices(*, num_frames: int, max_frames: int) -> np.ndarray:
    """Uniformly sample progress frames while keeping both episode endpoints.

    Reward-agent sensitivity evaluation and sampled video models must use the
    same index construction.  Keeping the helper next to the canonical progress
    contract prevents those two paths from drifting over time.
    """

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if num_frames == 1:
        return np.array([0], dtype=np.int64)
    if max_frames < 2:
        raise ValueError(
            "max_frames must be at least 2 for multi-frame episodes so both "
            f"the first and final frames can be sampled, got {max_frames}."
        )

    num_sampled = min(int(num_frames), int(max_frames))
    if num_sampled == num_frames:
        return np.arange(num_frames, dtype=np.int64)

    rounded = np.rint(np.linspace(0, num_frames - 1, num_sampled)).astype(np.int64)
    rounded[0] = 0
    rounded[-1] = num_frames - 1

    # Rounding can theoretically duplicate an index for very short sequences.
    # The floor-linspace fallback keeps the axis increasing and pins both ends.
    if len(set(int(index) for index in rounded.tolist())) != num_sampled:
        rounded = np.linspace(0, num_frames - 1, num_sampled, dtype=np.int64)
        rounded[0] = 0
        rounded[-1] = num_frames - 1

    if np.any(np.diff(rounded) <= 0):
        raise ValueError(
            f"Internal sampling error: expected increasing indices, got {rounded.tolist()}."
        )
    return rounded


def load_progress_csv(path: Path, *, value_max_points: int) -> list[ProgressRow]:
    """Load candidate progress rows from a reward-agent case pack.

    The reward-agent contract uses `frame_index` as the only editing axis and
    stores progress as integer points in the configured closed value range.
    """

    max_points = _coerce_value_max_points(value_max_points)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [column for column in REQUIRED_PROGRESS_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing progress.csv columns {missing}; got {reader.fieldnames}.")

        rows: list[ProgressRow] = []
        seen_frames: set[int] = set()
        for row_idx, row in enumerate(reader, start=2):
            frame_index = int(row["frame_index"])
            progress_points = int(round(float(row["progress_points"])))
            source = str(row.get("source") or "").strip()
            if frame_index in seen_frames:
                raise ValueError(f"Duplicated frame_index {frame_index} at CSV line {row_idx}.")
            if not 0 <= progress_points <= max_points:
                raise ValueError(
                    f"progress_points must be in [0, {max_points}], got {progress_points} "
                    f"for frame_index {frame_index}."
                )
            rows.append(
                ProgressRow(
                    frame_index=frame_index,
                    progress_points=progress_points,
                    source=source,
                )
            )
            seen_frames.add(frame_index)

    if not rows:
        raise ValueError(f"Empty progress CSV: {path}")
    return rows


def summarize_progress(
    frame_indices: np.ndarray,
    progress_points: np.ndarray,
    *,
    top_k: int = 8,
    jump_threshold: int = 5,
) -> ProgressSummary:
    """Return a compact summary that helps choose visual inspection windows."""

    if len(frame_indices) != len(progress_points):
        raise ValueError("frame_indices and progress_points must have the same length.")
    if len(frame_indices) == 0:
        raise ValueError("Cannot summarize an empty progress curve.")

    deltas = np.diff(progress_points.astype(np.int64))
    abs_deltas = np.abs(deltas)
    top_delta_indices = np.argsort(-abs_deltas)[: max(0, int(top_k))]

    top_delta_windows: list[dict[str, int]] = []
    for idx in top_delta_indices:
        if idx < 0 or idx >= len(deltas) or abs_deltas[idx] == 0:
            continue
        top_delta_windows.append(
            {
                "from_frame": int(frame_indices[idx]),
                "to_frame": int(frame_indices[idx + 1]),
                "delta_points": int(deltas[idx]),
                "from_progress_points": int(progress_points[idx]),
                "to_progress_points": int(progress_points[idx + 1]),
            }
        )

    jump_points: list[dict[str, int]] = []
    for idx, delta in enumerate(deltas):
        if abs(int(delta)) >= jump_threshold:
            jump_points.append(
                {
                    "from_frame": int(frame_indices[idx]),
                    "to_frame": int(frame_indices[idx + 1]),
                    "delta_points": int(delta),
                }
            )

    return ProgressSummary(
        num_rows=int(len(frame_indices)),
        start_frame=int(frame_indices[0]),
        end_frame=int(frame_indices[-1]),
        start_progress_points=int(progress_points[0]),
        end_progress_points=int(progress_points[-1]),
        min_progress_points=int(np.min(progress_points)),
        max_progress_points=int(np.max(progress_points)),
        top_delta_windows=top_delta_windows,
        jump_points=jump_points,
    )


def edit_progress_segment(
    *,
    frame_indices: np.ndarray,
    current_progress_points: np.ndarray,
    edit_reasons: list[str],
    edit_id: str,
    start_frame: int,
    end_frame: int,
    start_progress_points: int,
    end_progress_points: int,
    propagation: PropagationMode,
    reason: str,
    confidence: int,
    value_max_points: int,
) -> ProgressEditRecord:
    """Apply one reward-agent progress edit by linear endpoint interpolation.

    The function mutates `current_progress_points` and `edit_reasons` in place.
    Keeping interpolation and propagation here lets the session expose a simple
    typed tool while Python owns indexing, clipping, and audit bookkeeping.
    """

    _validate_reason_and_confidence(reason=reason, confidence=confidence)
    max_points = _coerce_value_max_points(value_max_points)
    if start_frame > end_frame:
        raise ValueError(f"start_frame must be <= end_frame, got {start_frame} > {end_frame}.")
    if propagation not in {"none", "shift_after"}:
        raise ValueError(f"Unsupported propagation mode: {propagation!r}")

    segment_mask = (frame_indices >= int(start_frame)) & (frame_indices <= int(end_frame))
    segment_positions = np.flatnonzero(segment_mask)
    if len(segment_positions) == 0:
        raise ValueError(f"No progress rows fall inside frame range [{start_frame}, {end_frame}].")

    old_end_pos = int(segment_positions[-1])
    old_end_value = int(current_progress_points[old_end_pos])
    start_value = _coerce_int_point(start_progress_points, value_max_points=max_points)
    end_value = _coerce_int_point(end_progress_points, value_max_points=max_points)
    trend = _progress_trend(start_progress_points=start_value, end_progress_points=end_value)
    new_values = _linear_values(
        num_positions=len(segment_positions),
        start_progress_points=start_value,
        end_progress_points=end_value,
        value_max_points=max_points,
    )

    current_progress_points[segment_positions] = new_values
    edit_reason = f"{edit_id}: {reason.strip()}"
    for pos in segment_positions:
        edit_reasons[int(pos)] = edit_reason

    affected_positions: set[int] = {int(pos) for pos in segment_positions}
    if propagation == "shift_after":
        delta_points = int(current_progress_points[old_end_pos]) - old_end_value
        after_positions = np.flatnonzero(frame_indices > int(end_frame))
        if len(after_positions) > 0 and delta_points != 0:
            shifted = current_progress_points[after_positions].astype(np.int64) + delta_points
            current_progress_points[after_positions] = np.clip(shifted, 0, max_points).astype(np.int64)
            for pos in after_positions:
                edit_reasons[int(pos)] = edit_reason
                affected_positions.add(int(pos))

    np.clip(current_progress_points, 0, max_points, out=current_progress_points)

    return ProgressEditRecord(
        edit_id=edit_id,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        start_progress_points=start_value,
        end_progress_points=end_value,
        trend=trend,
        propagation=propagation,
        reason=reason.strip(),
        confidence=int(confidence),
        affected_frames=len(affected_positions),
    )


def calibrate_progress_curve(
    *,
    frame_indices: np.ndarray,
    current_progress_points: np.ndarray,
    assigned_mask: np.ndarray,
    edit_reasons: list[str],
    calibration_id: str,
    anchor_a_frame: int,
    anchor_a_progress_points: int,
    anchor_b_frame: int,
    anchor_b_progress_points: int,
    reason: str,
    confidence: int,
    value_max_points: int,
) -> ProgressCalibrationRecord:
    """Apply one whole-trajectory affine calibration to assigned progress rows.

    The calibration uses two frame/value anchors to infer `new = scale * old +
    offset`. Unassigned rows are intentionally left untouched: they do not
    participate in the transform, bounds checks, or edit-reason updates.
    """

    _validate_reason_and_confidence(reason=reason, confidence=confidence)
    max_points = _coerce_value_max_points(value_max_points)
    if len(current_progress_points) != len(frame_indices):
        raise ProgressCalibrationError("current_progress_points length does not match frame_indices length.")
    if len(assigned_mask) != len(frame_indices):
        raise ProgressCalibrationError("assigned_mask length does not match frame_indices length.")
    if len(edit_reasons) != len(frame_indices):
        raise ProgressCalibrationError("edit_reasons length does not match frame_indices length.")

    assigned_positions = np.flatnonzero(assigned_mask.astype(bool))
    if len(assigned_positions) == 0:
        raise ProgressCalibrationError("calibrate_progress_curve requires at least one assigned progress row.")

    anchor_a_pos = _progress_position_at_frame(frame_indices, int(anchor_a_frame), tool_name="calibrate_progress_curve")
    anchor_b_pos = _progress_position_at_frame(frame_indices, int(anchor_b_frame), tool_name="calibrate_progress_curve")
    if not bool(assigned_mask[anchor_a_pos]):
        raise ProgressCalibrationError(
            f"anchor_a_frame {int(anchor_a_frame)} is not assigned; choose an assigned anchor frame."
        )
    if not bool(assigned_mask[anchor_b_pos]):
        raise ProgressCalibrationError(
            f"anchor_b_frame {int(anchor_b_frame)} is not assigned; choose an assigned anchor frame."
        )

    old_a = int(current_progress_points[anchor_a_pos])
    old_b = int(current_progress_points[anchor_b_pos])
    target_a = _coerce_int_point(anchor_a_progress_points, value_max_points=max_points)
    target_b = _coerce_int_point(anchor_b_progress_points, value_max_points=max_points)
    if old_a == old_b:
        raise ProgressCalibrationError(
            "Cannot infer affine calibration because the two anchor frames have "
            f"the same current progress value: {old_a}.",
            data={
                "anchor_a_frame": int(anchor_a_frame),
                "anchor_b_frame": int(anchor_b_frame),
                "anchor_current_progress_points": old_a,
            },
        )

    scale = (target_b - target_a) / float(old_b - old_a)
    offset = target_a - scale * old_a
    if scale < 0:
        raise ProgressCalibrationError(
            f"Refusing calibrate_progress_curve with negative scale {scale:.6f}; "
            "it would reverse progress ordering.",
            data={
                "anchor_a_frame": int(anchor_a_frame),
                "anchor_a_old_progress_points": old_a,
                "anchor_a_target_progress_points": target_a,
                "anchor_b_frame": int(anchor_b_frame),
                "anchor_b_old_progress_points": old_b,
                "anchor_b_target_progress_points": target_b,
                "scale": float(scale),
                "offset": float(offset),
            },
        )

    old_values = current_progress_points[assigned_positions].astype(np.int64)
    transformed = np.rint(scale * old_values.astype(np.float64) + offset).astype(np.int64)
    out_of_range_mask = (transformed < 0) | (transformed > max_points)
    if bool(np.any(out_of_range_mask)):
        offending_positions = assigned_positions[out_of_range_mask]
        offending_values = transformed[out_of_range_mask]
        offending_frames = [
            {
                "frame_index": int(frame_indices[int(pos)]),
                "transformed_progress_points": int(value),
            }
            for pos, value in zip(offending_positions[:8], offending_values[:8], strict=False)
        ]
        raise ProgressCalibrationError(
            "Refusing calibrate_progress_curve because transformed assigned rows would fall outside "
            f"[0, {max_points}].",
            data={
                "value_range": [0, max_points],
                "transformed_min_progress_points": int(np.min(transformed)),
                "transformed_max_progress_points": int(np.max(transformed)),
                "num_offending_frames": int(np.sum(out_of_range_mask)),
                "offending_frames": offending_frames,
                "scale": float(scale),
                "offset": float(offset),
            },
        )

    current_progress_points[assigned_positions] = transformed
    edit_reason = f"{calibration_id}: {reason.strip()}"
    for pos in assigned_positions:
        edit_reasons[int(pos)] = edit_reason

    return ProgressCalibrationRecord(
        calibration_id=calibration_id,
        anchor_a_frame=int(anchor_a_frame),
        anchor_a_old_progress_points=old_a,
        anchor_a_target_progress_points=target_a,
        anchor_b_frame=int(anchor_b_frame),
        anchor_b_old_progress_points=old_b,
        anchor_b_target_progress_points=target_b,
        scale=float(scale),
        offset=float(offset),
        old_start_progress_points=int(old_values[0]),
        old_end_progress_points=int(old_values[-1]),
        old_min_progress_points=int(np.min(old_values)),
        old_max_progress_points=int(np.max(old_values)),
        new_start_progress_points=int(transformed[0]),
        new_end_progress_points=int(transformed[-1]),
        new_min_progress_points=int(np.min(transformed)),
        new_max_progress_points=int(np.max(transformed)),
        reason=reason.strip(),
        confidence=int(confidence),
        affected_frames=int(len(assigned_positions)),
    )


def export_corrected_progress_csv(
    *,
    path: Path,
    frame_indices: np.ndarray,
    proposal_progress_points: np.ndarray | None,
    current_progress_points: np.ndarray,
    assigned_mask: np.ndarray,
    edit_reasons: list[str],
    value_max_points: int,
) -> Path:
    """Export the full corrected or generated progress curve from session state."""

    max_points = _coerce_value_max_points(value_max_points)
    if len(assigned_mask) != len(frame_indices):
        raise ValueError("assigned_mask length must match frame_indices length.")
    if not bool(np.all(assigned_mask)):
        missing = int(np.sum(~assigned_mask))
        raise ValueError(f"Cannot export progress while {missing} frames are still unassigned.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORTED_PROGRESS_COLUMNS)
        writer.writeheader()
        for idx, frame_index in enumerate(frame_indices):
            proposal_value = (
                ""
                if proposal_progress_points is None
                else int(proposal_progress_points[idx])
            )
            current_points = int(current_progress_points[idx])
            normalized = float(current_points) / float(max_points)
            writer.writerow(
                {
                    "frame_index": int(frame_index),
                    "proposal_progress_points": proposal_value,
                    "current_progress_points": current_points,
                    "current_progress_normalized": f"{normalized:.17g}",
                    # Reward-agent emits a native estimate for every LeRobot row.
                    # Sampled models may use zero here for rows introduced by a
                    # later fill operation.
                    "is_model_output": 1,
                    "edit_reason": edit_reasons[idx],
                    "source": "reward_agent",
                }
            )
    return path


def validate_progress_curve(
    *,
    frame_indices: np.ndarray,
    proposal_progress_points: np.ndarray | None,
    current_progress_points: np.ndarray,
    assigned_mask: np.ndarray,
    edit_reasons: list[str],
    value_max_points: int,
    is_success_ground_truth: bool | None = None,
    csv_path: Path | None = None,
) -> ProgressValidationResult:
    """Validate the in-memory reward progress curve and optional exported CSV."""

    max_points = _coerce_value_max_points(value_max_points)
    errors: list[str] = []
    warnings: list[str] = []
    row_count = len(frame_indices)

    if proposal_progress_points is not None and len(proposal_progress_points) != row_count:
        errors.append("proposal_progress_points length does not match frame_indices length.")
    if len(current_progress_points) != row_count:
        errors.append("current_progress_points length does not match frame_indices length.")
    if len(assigned_mask) != row_count:
        errors.append("assigned_mask length does not match frame_indices length.")
    if len(edit_reasons) != row_count:
        errors.append("edit_reasons length does not match frame_indices length.")
    if len(set(int(frame) for frame in frame_indices)) != row_count:
        errors.append("frame_index contains duplicates.")
    if np.any(current_progress_points < 0) or np.any(current_progress_points > max_points):
        errors.append(f"current_progress_points contains values outside [0, {max_points}].")
    if len(assigned_mask) == row_count and not bool(np.all(assigned_mask)):
        errors.append(f"{int(np.sum(~assigned_mask))} progress rows are still unassigned.")
    if any(bool(assigned_mask[idx]) and not str(reason).strip() for idx, reason in enumerate(edit_reasons)):
        errors.append("assigned progress rows must have non-empty edit_reason values.")
    if is_success_ground_truth is not None and not isinstance(is_success_ground_truth, bool):
        errors.append("is_success_ground_truth must be a boolean when provided.")
    if not errors and row_count > 0:
        terminal_progress = int(current_progress_points[-1])
        max_progress = int(np.max(current_progress_points))
        has_current_regression = bool(np.any(np.diff(current_progress_points.astype(np.int64)) < 0))
        if is_success_ground_truth is True and terminal_progress != max_points:
            errors.append(
                "Ground-truth success requires terminal progress to equal "
                f"{max_points}, got {terminal_progress}."
            )
        elif is_success_ground_truth is False and terminal_progress >= max_points:
            errors.append(
                "Ground-truth failure requires terminal progress to remain below "
                f"{max_points}, got {terminal_progress}."
            )
        if terminal_progress < max_points and max_progress > 0 and not has_current_regression:
            errors.append(
                "Incomplete terminal progress curves with nonzero achieved progress must contain "
                "at least one current decreasing interval."
            )

    if csv_path is not None:
        _validate_exported_csv(
            csv_path=csv_path,
            frame_indices=frame_indices,
            proposal_progress_points=proposal_progress_points,
            current_progress_points=current_progress_points,
            edit_reasons=edit_reasons,
            value_max_points=max_points,
            errors=errors,
        )

    return ProgressValidationResult(ok=not errors, errors=errors, warnings=warnings, csv_path=csv_path)


def render_progress_plot(
    *,
    path: Path,
    frame_indices: np.ndarray,
    proposal_progress_points: np.ndarray | None,
    current_progress_points: np.ndarray,
    value_max_points: int,
    assigned_mask: np.ndarray | None = None,
    title: str = "Reward Agent Progress Curve",
) -> Path:
    """Render proposal and current reward progress curves as a PNG."""

    max_points = _coerce_value_max_points(value_max_points)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=160)
    if proposal_progress_points is not None:
        ax.plot(frame_indices, proposal_progress_points, label="proposal", linewidth=2.0, color="#777777")
    plotted_current = current_progress_points.astype(np.float64)
    if assigned_mask is not None:
        plotted_current = plotted_current.copy()
        plotted_current[~assigned_mask.astype(bool)] = np.nan
    ax.plot(frame_indices, plotted_current, label="current", linewidth=2.0, color="#0072B2")
    ax.set_title(title)
    ax.set_xlabel("frame_index")
    ax.set_ylabel("progress_points")
    if len(frame_indices) > 1:
        ax.set_xlim(0, int(frame_indices[-1]))
    elif len(frame_indices) == 1:
        ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, max_points)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _linear_values(
    *,
    num_positions: int,
    start_progress_points: int,
    end_progress_points: int,
    value_max_points: int,
) -> np.ndarray:
    """Resolve inclusive linear endpoint values for selected progress rows."""

    max_points = _coerce_value_max_points(value_max_points)
    if num_positions <= 0:
        raise ValueError("num_positions must be positive.")
    if num_positions == 1:
        return np.array([end_progress_points], dtype=np.int64)
    interpolated = np.linspace(start_progress_points, end_progress_points, num=num_positions)
    return np.clip(np.rint(interpolated), 0, max_points).astype(np.int64)


def _progress_trend(*, start_progress_points: int, end_progress_points: int) -> ProgressTrend:
    """Classify an edit segment by its endpoint direction for audit output."""

    if end_progress_points > start_progress_points:
        return "increase"
    if end_progress_points < start_progress_points:
        return "decrease"
    return "flat"


def _coerce_int_point(value: int, *, value_max_points: int) -> int:
    max_points = _coerce_value_max_points(value_max_points)
    point = int(round(float(value)))
    if not 0 <= point <= max_points:
        raise ValueError(f"Progress point must be in [0, {max_points}], got {point}.")
    return point


def _coerce_value_max_points(value_max_points: int) -> int:
    max_points = int(value_max_points)
    if max_points <= 0:
        raise ValueError(f"value_max_points must be positive, got {value_max_points}.")
    return max_points


def _progress_position_at_frame(frame_indices: np.ndarray, frame_index: int, *, tool_name: str) -> int:
    """Return the single progress-row position for an exact frame anchor."""

    positions = np.flatnonzero(frame_indices == int(frame_index))
    if len(positions) == 0:
        raise ProgressCalibrationError(f"{tool_name} anchor frame {int(frame_index)} is not in the progress frame axis.")
    if len(positions) > 1:
        raise ProgressCalibrationError(f"{tool_name} anchor frame {int(frame_index)} appears more than once.")
    return int(positions[0])


def _validate_reason_and_confidence(*, reason: str, confidence: int) -> None:
    if not str(reason).strip():
        raise ValueError("reason must be non-empty.")
    if not isinstance(confidence, int) or isinstance(confidence, bool):
        raise TypeError(f"confidence must be an integer in [1, 5], got {confidence!r}.")
    if not 1 <= confidence <= 5:
        raise ValueError(f"confidence must be an integer in [1, 5], got {confidence}.")


def _validate_exported_csv(
    *,
    csv_path: Path,
    frame_indices: np.ndarray,
    proposal_progress_points: np.ndarray | None,
    current_progress_points: np.ndarray,
    edit_reasons: list[str],
    value_max_points: int,
    errors: list[str],
) -> None:
    if not csv_path.is_file():
        errors.append(f"corrected progress CSV does not exist: {csv_path}")
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [column for column in EXPORTED_PROGRESS_COLUMNS if column not in reader.fieldnames]
        if missing:
            errors.append(f"corrected CSV missing columns: {missing}")
            return
        rows = list(reader)

    if len(rows) != len(frame_indices):
        errors.append(f"corrected CSV row count {len(rows)} != expected {len(frame_indices)}.")
        return

    seen_frames: set[int] = set()
    max_points = _coerce_value_max_points(value_max_points)
    for idx, row in enumerate(rows):
        frame_index = int(row["frame_index"])
        proposal_raw = str(row["proposal_progress_points"]).strip()
        current = int(row["current_progress_points"])
        normalized = float(row["current_progress_normalized"])
        is_model_output = str(row["is_model_output"]).strip()
        edit_reason = str(row["edit_reason"])
        source = str(row["source"]).strip()
        if frame_index in seen_frames:
            errors.append(f"corrected CSV duplicates frame_index {frame_index}.")
        seen_frames.add(frame_index)
        if frame_index != int(frame_indices[idx]):
            errors.append(
                f"CSV frame_index mismatch at row {idx}: got {frame_index}, "
                f"expected {int(frame_indices[idx])}."
            )
        if proposal_progress_points is None:
            if proposal_raw:
                errors.append(
                    f"CSV proposal_progress_points should be empty at frame {frame_index}, "
                    f"got {proposal_raw!r}."
                )
        else:
            if not proposal_raw:
                errors.append(f"CSV proposal_progress_points is empty at frame {frame_index}.")
                continue
            proposal = int(proposal_raw)
            if proposal != int(proposal_progress_points[idx]):
                errors.append(
                    f"CSV proposal_progress_points mismatch at frame {frame_index}: "
                    f"got {proposal}, expected {int(proposal_progress_points[idx])}."
                )
        if current != int(current_progress_points[idx]):
            errors.append(
                f"CSV current_progress_points mismatch at frame {frame_index}: "
                f"got {current}, expected {int(current_progress_points[idx])}."
            )
        expected_normalized = float(current_progress_points[idx]) / float(max_points)
        if not 0.0 <= normalized <= 1.0:
            errors.append(
                f"CSV current_progress_normalized at frame {frame_index} "
                f"is outside [0, 1]: {normalized}."
            )
        if not np.isclose(normalized, expected_normalized, rtol=0.0, atol=1e-12):
            errors.append(
                f"CSV current_progress_normalized mismatch at frame {frame_index}: "
                f"got {normalized}, expected {expected_normalized}."
            )
        if is_model_output != "1":
            errors.append(
                f"CSV is_model_output at frame {frame_index} must be 1 for reward-agent, "
                f"got {is_model_output!r}."
            )
        if edit_reason != edit_reasons[idx]:
            errors.append(f"CSV edit_reason mismatch at frame {frame_index}.")
        if source != "reward_agent":
            errors.append(
                f"CSV source at frame {frame_index} must be 'reward_agent', got {source!r}."
            )
