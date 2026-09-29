"""Bounded visual evidence generation for reward-agent tool calls."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schemas import VisualAsset


_REVIEW_MIN_RENDER_WIDTH = 720
_REVIEW_MIN_RENDER_HEIGHT = 720
_REVIEW_PLOT_MIN_WIDTH = 480
_REVIEW_PLOT_WIDTH_PER_HEIGHT = 2.0


def inspect_visual_frames(
    *,
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    output_dir: str | Path,
    start_frame: int,
    end_frame: int,
    sample_every_n: int,
    mode: str,
    reason: str,
    progress_labels: dict[int, str] | None = None,
    crop: dict[str, int] | None = None,
    tile_columns: int = 4,
    tile_max_width: int = 4096,
    video_codec: str = "libx264",
    video_fps: float | None = None,
    show_frame_label: bool = True,
) -> VisualAsset:
    """Create one bounded video clip or frame tile from decoded RGB frames."""

    if not frames:
        raise ValueError("Cannot inspect visual evidence without frames.")
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must have the same length.")
    if start_frame > end_frame:
        raise ValueError(f"start_frame must be <= end_frame, got {start_frame} > {end_frame}.")
    if mode not in {"video", "tile"}:
        raise ValueError(f"mode must be 'video' or 'tile', got {mode!r}.")
    if not str(reason).strip():
        raise ValueError("reason must be non-empty.")
    crop = _normalize_crop(crop)

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    asset_id = _asset_id_from_frames(
        mode=mode,
        start_frame=start_frame,
        end_frame=end_frame,
        sample_every_n=sample_every_n,
        frame_indices=frame_indices,
        frames=frames,
        progress_labels=progress_labels,
        show_frame_label=show_frame_label,
        crop=crop,
    )
    if mode == "video":
        path = output_root / f"{asset_id}.mp4"
        write_video_clip(
            frames=frames,
            frame_indices=frame_indices,
            output_path=path,
            fps=video_fps or 8.0,
            codec=video_codec,
            progress_labels=progress_labels,
            show_frame_label=show_frame_label,
            crop=crop,
        )
        kind = "video_clip"
    else:
        path = output_root / f"{asset_id}.png"
        write_tile(
            frames=frames,
            frame_indices=frame_indices,
            output_path=path,
            progress_labels=progress_labels,
            show_frame_label=show_frame_label,
            crop=crop,
            columns=tile_columns,
            max_tile_width=tile_max_width,
        )
        kind = "tile"

    return VisualAsset(
        asset_id=asset_id,
        kind=kind,  # type: ignore[arg-type]
        path=path,
        frame_range=(int(start_frame), int(end_frame)),
        frame_indices=frame_indices,
        sample_every_n=max(1, int(sample_every_n)),
        attached_frame_count=len(frame_indices),
        reason=str(reason).strip(),
        crop=dict(crop) if crop is not None else None,
    )


def sample_frame_indices(
    *,
    start_frame: int,
    end_frame: int,
    sample_every_n: int = 1,
) -> list[int]:
    """Choose the exact closed-interval frame set for visual inspection.

    Strided requests always retain both closed interval endpoints so the
    attached evidence matches the frame range reported to the agent.
    """

    if start_frame > end_frame:
        raise ValueError(f"start_frame must be <= end_frame, got {start_frame} > {end_frame}.")
    step = max(1, int(sample_every_n))
    indices = list(range(int(start_frame), int(end_frame) + 1, step))
    indices.append(int(end_frame))
    return _dedupe_preserve_order(indices)


def read_selected_frames(video_path: Path, frame_indices: Sequence[int]) -> list[np.ndarray]:
    """Decode only selected RGB frames from a video path."""

    if not frame_indices:
        raise ValueError("frame_indices must be non-empty.")
    targets = {int(frame) for frame in frame_indices}
    max_target = max(targets)
    frames_by_index: dict[int, np.ndarray] = {}
    reader = imageio.get_reader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if idx in targets:
                frames_by_index[idx] = _normalize_rgb_frame(frame)
                if len(frames_by_index) == len(targets):
                    break
            if idx > max_target:
                break
    finally:
        reader.close()
    missing = [frame for frame in frame_indices if int(frame) not in frames_by_index]
    if missing:
        raise IndexError(f"Requested frames outside decoded video range: {missing}")
    return [frames_by_index[int(frame)] for frame in frame_indices]


def write_video_clip(
    *,
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    output_path: str | Path,
    fps: float,
    codec: str = "libx264",
    progress_labels: dict[int, str] | None = None,
    show_frame_label: bool = False,
    crop: dict[str, int] | None = None,
) -> Path:
    """Write selected frames to a compact MP4 clip for vLLM video_url input."""

    if not frames:
        raise ValueError("Cannot write a video clip without frames.")
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must have the same length.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=float(fps), codec=codec, macro_block_size=1) as writer:
        for frame, frame_index in zip(frames, frame_indices, strict=True):
            prepared = _prepare_frame(
                _normalize_rgb_frame(frame),
                crop=crop,
            )
            label = _frame_label(
                int(frame_index),
                progress_labels,
                show_frame_label=show_frame_label,
            )
            output_frame = prepared if label is None else _add_label_bar(prepared, label)
            writer.append_data(_pad_frame_to_even_shape(output_frame))
    return path


def transcode_reference_video(
    *,
    source_path: str | Path,
    output_path: str | Path,
    fps: float,
    codec: str = "libx264",
) -> tuple[Path, int]:
    """Re-encode an independent reference video without trajectory labels.

    Reference examples share the model-facing FPS and codec with trajectory
    videos, but they do not share trajectory frame indices. This streaming
    encoder therefore writes decoded frames in source order without adding a
    frame bar, progress bar, or any other trajectory-specific annotation.
    """

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reference video does not exist: {source}")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_fps = float(fps)
    if output_fps <= 0:
        raise ValueError(f"Reference video fps must be positive, got {fps!r}.")

    frames_written = 0
    reader = imageio.get_reader(source)
    writer = imageio.get_writer(
        output,
        fps=output_fps,
        codec=codec,
        macro_block_size=1,
    )
    try:
        for raw_frame in reader:
            frame = _pad_frame_to_even_shape(_normalize_rgb_frame(raw_frame))
            writer.append_data(frame)
            frames_written += 1
    finally:
        reader.close()
        writer.close()

    if frames_written == 0:
        output.unlink(missing_ok=True)
        raise ValueError(f"Reference video contains no decodable frames: {source}")
    return output, frames_written


def write_progress_review_video(
    *,
    frames: Sequence[np.ndarray],
    visual_frame_indices: Sequence[int] | np.ndarray,
    output_path: str | Path,
    progress_frame_indices: Sequence[int] | np.ndarray,
    current_progress_points: Sequence[int] | np.ndarray,
    proposal_progress_points: Sequence[int] | np.ndarray | None = None,
    assigned_mask: Sequence[bool] | np.ndarray | None = None,
    fps: float,
    codec: str = "libx264",
    title: str = "Reward Agent Progress Curve",
    value_max_points: int = 1000,
) -> Path:
    """Write a human-only side-by-side review video.

    Each output frame keeps the original video frame on the left. The right
    panel shows the full corrected progress curve for the whole trajectory and
    only moves one vertical dashed marker to identify the current video frame.
    """

    if len(frames) == 0:
        raise ValueError("Cannot render a progress review video without frames.")
    if len(frames) != len(visual_frame_indices):
        raise ValueError("frames and visual_frame_indices must have the same length.")

    cursor_frames = np.asarray(visual_frame_indices, dtype=np.float64)
    progress_frames = np.asarray(progress_frame_indices, dtype=np.float64)
    current_points = np.asarray(current_progress_points, dtype=np.float64)
    if cursor_frames.ndim != 1:
        raise ValueError("visual_frame_indices must be one-dimensional.")
    if progress_frames.ndim != 1 or current_points.ndim != 1:
        raise ValueError("progress_frame_indices and current_progress_points must be one-dimensional.")
    if len(progress_frames) == 0:
        raise ValueError("Cannot render a progress review video without progress points.")
    if len(progress_frames) != len(current_points):
        raise ValueError("progress_frame_indices and current_progress_points must have the same length.")

    proposal_points: np.ndarray | None = None
    if proposal_progress_points is not None:
        proposal_points = np.asarray(proposal_progress_points, dtype=np.float64)
        if proposal_points.shape != current_points.shape:
            raise ValueError("proposal_progress_points must match current_progress_points length.")

    assigned: np.ndarray | None = None
    if assigned_mask is not None:
        assigned = np.asarray(assigned_mask, dtype=bool)
        if assigned.shape != current_points.shape:
            raise ValueError("assigned_mask must match current_progress_points length.")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_fps = float(fps)
    if output_fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}.")

    writer = None
    plotter: _ProgressReviewPlotter | None = None
    render_size: tuple[int, int] | None = None
    plot_size: tuple[int, int] | None = None
    frames_written = 0
    try:
        for visual_frame_index, raw_frame in zip(cursor_frames, frames, strict=True):
            original = _normalize_rgb_frame(raw_frame)
            if render_size is None:
                render_size = _resolve_review_render_frame_size(
                    frame_width=int(original.shape[1]),
                    frame_height=int(original.shape[0]),
                )
                plot_size = _resolve_review_plot_frame_size(
                    frame_width=render_size[0],
                    frame_height=render_size[1],
                )
                plotter = _ProgressReviewPlotter(
                    frame_indices=progress_frames,
                    proposal_progress_points=proposal_points,
                    current_progress_points=current_points,
                    assigned_mask=assigned,
                    image_size=plot_size,
                    title=title,
                    value_max_points=value_max_points,
                )
                writer = imageio.get_writer(
                    output,
                    fps=output_fps,
                    format="FFMPEG",
                    codec=codec,
                    pixelformat="yuv420p",
                    quality=None,
                    macro_block_size=1,
                )

            if writer is None or plotter is None or render_size is None or plot_size is None:
                raise RuntimeError("Progress review video writer was not initialized.")

            # Enlarge small simulator frames before adding the wider diagnostic
            # panel so both the trajectory and progress curve remain readable.
            original = _resize_if_needed(original, size=render_size)
            plot_frame = plotter.render(current_frame_index=int(visual_frame_index))
            writer.append_data(_pad_frame_to_even_shape(np.concatenate([original, plot_frame], axis=1)))
            frames_written += 1
    finally:
        if writer is not None:
            writer.close()
        if plotter is not None:
            plotter.close()

    if frames_written == 0:
        raise ValueError("No frames were written to the progress review video.")
    return output


def write_tile(
    *,
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    output_path: str | Path,
    progress_labels: dict[int, str] | None = None,
    show_frame_label: bool = True,
    crop: dict[str, int] | None = None,
    columns: int = 4,
    max_tile_width: int = 4096,
) -> Path:
    """Write selected frames into a labeled tile image for focused inspection."""

    if not frames:
        raise ValueError("Cannot write a tile without frames.")
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must have the same length.")

    columns = max(1, int(columns))
    cell_images = [
        _resize_for_tile(
            Image.fromarray(
                _prepare_frame(
                    _normalize_rgb_frame(frame),
                    crop=crop,
                )
            ),
            max_width=max_tile_width,
        )
        for frame in frames
    ]
    cell_width = max(image.width for image in cell_images)
    cell_height = max(image.height for image in cell_images)
    labels = [
        _frame_label(int(frame_index), progress_labels, show_frame_label=show_frame_label)
        for frame_index in frame_indices
    ]
    has_label_bar = any(label is not None for label in labels)
    label_height = 38 if has_label_bar else 0
    rows = int(math.ceil(len(cell_images) / columns))

    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = _default_font()
    for idx, image in enumerate(cell_images):
        row = idx // columns
        column = idx % columns
        x = column * cell_width
        y = row * (cell_height + label_height)
        canvas.paste(image, (x, y + label_height))
        label = labels[idx]
        if label is not None:
            draw.rectangle((x, y, x + cell_width - 1, y + label_height - 1), fill=(25, 25, 25))
            draw.text(
                (x + 8, y + 10),
                label,
                fill=(255, 255, 255),
                font=font,
            )
        draw.rectangle(
            (x, y, x + cell_width - 1, y + cell_height + label_height - 1),
            outline=(80, 80, 80),
            width=1,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


class _ProgressReviewPlotter:
    """Reusable matplotlib canvas for drawing the full curve plus one cursor."""

    def __init__(
        self,
        *,
        frame_indices: np.ndarray,
        proposal_progress_points: np.ndarray | None,
        current_progress_points: np.ndarray,
        assigned_mask: np.ndarray | None,
        image_size: tuple[int, int],
        title: str,
        value_max_points: int,
    ) -> None:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plt = plt
        self._width, self._height = image_size
        dpi = 100.0
        self._fig, self._ax = plt.subplots(figsize=(self._width / dpi, self._height / dpi), dpi=dpi)

        if proposal_progress_points is not None:
            self._ax.plot(frame_indices, proposal_progress_points, label="proposal", linewidth=2.0, color="#777777")

        plotted_current = current_progress_points.astype(np.float64)
        if assigned_mask is not None:
            plotted_current = plotted_current.copy()
            plotted_current[~assigned_mask.astype(bool)] = np.nan
        self._ax.plot(frame_indices, plotted_current, label="current", linewidth=2.0, color="#0072B2")

        self._ax.set_title(title)
        self._ax.set_xlabel("frame_index")
        self._ax.set_ylabel("progress_points")
        if len(frame_indices) > 1:
            self._ax.set_xlim(0, int(frame_indices[-1]))
        elif len(frame_indices) == 1:
            self._ax.set_xlim(-0.5, 0.5)
        self._ax.set_ylim(0, int(value_max_points))
        self._ax.grid(True, alpha=0.25)
        self._cursor = self._ax.axvline(0, linestyle="--", linewidth=1.8, color="#D55E00", alpha=0.95)
        # Keep the legend fixed across frames. Matplotlib's "best" placement
        # can jump when the moving frame cursor changes the overlap estimate.
        self._ax.legend(loc="upper left")
        # Use explicit margins instead of tight_layout so tiny smoke-test
        # videos do not emit layout warnings while the full curve remains
        # visible on the human-review panel.
        self._fig.subplots_adjust(left=0.18, right=0.96, bottom=0.2, top=0.86)

    def render(self, *, current_frame_index: int) -> np.ndarray:
        """Render the full curve with the cursor located at current_frame_index."""

        self._cursor.set_xdata([float(current_frame_index), float(current_frame_index)])
        self._fig.canvas.draw()
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        rgb = np.ascontiguousarray(rgba[..., :3])
        if rgb.shape[0] != self._height or rgb.shape[1] != self._width:
            return _resize_if_needed(rgb, size=(self._width, self._height))
        return rgb

    def close(self) -> None:
        """Release the matplotlib figure owned by this renderer."""

        self._plt.close(self._fig)


def _asset_id_from_frames(
    *,
    mode: str,
    start_frame: int,
    end_frame: int,
    sample_every_n: int,
    frame_indices: Sequence[int],
    frames: Sequence[np.ndarray],
    progress_labels: dict[int, str] | None = None,
    show_frame_label: bool = False,
    crop: dict[str, int] | None = None,
) -> str:
    """Build a deterministic attachment id from rendered input frames."""

    digest = hashlib.sha1()
    for item in (
        str(mode),
        str(int(start_frame)),
        str(int(end_frame)),
        str(int(sample_every_n)),
        ",".join(str(int(frame)) for frame in frame_indices),
        repr({int(key): str(value) for key, value in sorted((progress_labels or {}).items())}),
        str(bool(show_frame_label)),
        repr(crop),
    ):
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    for frame in frames:
        normalized = _normalize_rgb_frame(frame)
        digest.update(str(tuple(int(dim) for dim in normalized.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.tobytes())
        digest.update(b"\0")
    prefix = "video_clip" if mode == "video" else "tile"
    return f"{prefix}_{digest.hexdigest()[:12]}"


def _normalize_rgb_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected RGB frame with 3 dimensions, got shape {array.shape}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame, got shape {array.shape}.")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)
    return np.ascontiguousarray(array.astype(np.uint8))


def _pad_frame_to_even_shape(frame: np.ndarray) -> np.ndarray:
    """Pad RGB frames to even dimensions for broadly compatible H.264 output."""

    pad_height = frame.shape[0] % 2
    pad_width = frame.shape[1] % 2
    if pad_height == 0 and pad_width == 0:
        return frame
    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def _prepare_frame(
    frame: np.ndarray,
    *,
    crop: dict[str, int] | None,
) -> np.ndarray:
    """Apply optional percent crop before adding labels outside the visual content."""

    image = Image.fromarray(_normalize_rgb_frame(frame))
    normalized_crop = _normalize_crop(crop)
    if normalized_crop is not None:
        image = image.crop(_resolve_crop(image.size, crop=normalized_crop))
    return np.asarray(image)


def _add_label_bar(frame: np.ndarray, label: str) -> np.ndarray:
    """Place frame/progress text outside the image content in a top label bar."""

    image = Image.fromarray(_normalize_rgb_frame(frame))
    font = _default_font()
    label_height = 38
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(25, 25, 25))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), str(label), fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


def _normalize_crop(crop: dict[str, int] | None) -> dict[str, int] | None:
    """Validate and canonicalize the agent-facing percent crop object."""

    if crop is None:
        return None
    if not isinstance(crop, dict):
        raise ValueError(f"crop must be an object with percentage fields, got {crop!r}.")

    required = ("left_pct", "top_pct", "right_pct", "bottom_pct")
    missing = [key for key in required if key not in crop]
    if missing:
        raise ValueError(f"crop is missing required percentage field(s): {', '.join(missing)}.")

    normalized: dict[str, int] = {}
    for key in required:
        value = crop[key]
        if isinstance(value, bool):
            raise ValueError(f"crop.{key} must be an integer percentage, got {value!r}.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"crop.{key} must be an integer percentage, got {value!r}.") from exc
        if not parsed.is_integer():
            raise ValueError(f"crop.{key} must be an integer percentage, got {value!r}.")
        normalized[key] = int(parsed)

    left = normalized["left_pct"]
    top = normalized["top_pct"]
    right = normalized["right_pct"]
    bottom = normalized["bottom_pct"]
    if not (0 <= left < right <= 100):
        raise ValueError("crop must satisfy 0 <= left_pct < right_pct <= 100.")
    if not (0 <= top < bottom <= 100):
        raise ValueError("crop must satisfy 0 <= top_pct < bottom_pct <= 100.")
    return normalized


def _resolve_crop(
    size: tuple[int, int],
    *,
    crop: dict[str, int],
) -> tuple[int, int, int, int]:
    width, height = size
    normalized = _normalize_crop(crop)
    if normalized is None:
        return 0, 0, width, height

    left = int(round(width * normalized["left_pct"] / 100.0))
    top = int(round(height * normalized["top_pct"] / 100.0))
    right = int(round(width * normalized["right_pct"] / 100.0))
    bottom = int(round(height * normalized["bottom_pct"] / 100.0))

    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def _resolve_review_render_frame_size(*, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Upscale tiny raw videos so the human-review canvas stays readable."""

    scale = max(
        _REVIEW_MIN_RENDER_WIDTH / float(frame_width),
        _REVIEW_MIN_RENDER_HEIGHT / float(frame_height),
        1.0,
    )
    return int(round(frame_width * scale)), int(round(frame_height * scale))


def _resolve_review_plot_frame_size(*, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Choose a readable landscape plot panel independent of the raw frame width."""

    plot_width = max(
        frame_width,
        _REVIEW_PLOT_MIN_WIDTH,
        int(round(frame_height * _REVIEW_PLOT_WIDTH_PER_HEIGHT)),
    )
    return plot_width, frame_height


def _frame_label(
    frame_index: int,
    progress_labels: dict[int, str] | None,
    *,
    show_frame_label: bool,
) -> str | None:
    """Format an optional external label from independent frame/progress parts."""

    parts: list[str] = []
    if show_frame_label:
        parts.append(f"frame {int(frame_index)}")
    if progress_labels is not None and int(frame_index) in progress_labels:
        parts.append(progress_labels[int(frame_index)])
    if not parts:
        return None
    return " | ".join(parts)


def _resize_for_tile(image: Image.Image, *, max_width: int) -> Image.Image:
    if image.width <= int(max_width):
        return image
    height = max(1, int(round(image.height * (int(max_width) / float(image.width)))))
    return image.resize((int(max_width), height), resample=Image.BICUBIC)


def _resize_if_needed(frame: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB frame only when it does not already match width/height."""

    image = Image.fromarray(_normalize_rgb_frame(frame))
    width, height = int(size[0]), int(size[1])
    if image.size == (width, height):
        return np.ascontiguousarray(np.asarray(image))
    resized = image.resize((width, height), resample=Image.BICUBIC)
    return np.ascontiguousarray(np.asarray(resized))


def _default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _dedupe_preserve_order(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        normalized = int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
