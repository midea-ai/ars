"""Task-bound multimodal reference examples injected into model conversations."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image

from .visual import transcode_reference_video


ReferenceInputType = Literal["image", "video"]
_PROGRESS_MARKER_PREFIX = "[[progress:"
_PROGRESS_MARKER_SUFFIX = "]]"
_SUPPORTED_INPUT_TYPES = {"image", "video"}


@dataclass(frozen=True, slots=True)
class ReferenceInputConfig:
    """One configured image or video belonging to a reference example."""

    type: ReferenceInputType
    path: Path


@dataclass(frozen=True, slots=True)
class ReferenceExampleConfig:
    """One task-bound reference example before run-local materialization."""

    inputs: tuple[ReferenceInputConfig, ...]
    description: str


@dataclass(frozen=True, slots=True)
class ProgressMarker:
    """One normalized progress marker compiled from reference prose."""

    ratio: Decimal
    progress_points: int
    start_offset: int
    end_offset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": str(self.ratio),
            "progress_points": int(self.progress_points),
            "start_offset": int(self.start_offset),
            "end_offset": int(self.end_offset),
        }


@dataclass(frozen=True, slots=True)
class MaterializedReferenceInput:
    """One run-local image copy or transcoded video shown to the model."""

    type: ReferenceInputType
    path: Path
    source_sha256: str
    output_sha256: str
    materialization: Literal["copied", "transcoded"]
    output_frame_count: int | None = None
    output_fps: int | None = None
    output_codec: str | None = None

    def media_content(self) -> dict[str, Any]:
        if self.type == "video":
            return {
                "type": "video_url",
                "video_url": {"url": self.path.resolve().as_uri()},
            }
        return {
            "type": "image_url",
            "image_url": {"url": self.path.resolve().as_uri()},
        }

    def to_manifest_dict(self, *, run_dir: Path) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "run_path": self.path.resolve().relative_to(run_dir.resolve()).as_posix(),
            "materialization": self.materialization,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
        }
        if self.output_frame_count is not None:
            data["output_frame_count"] = int(self.output_frame_count)
        if self.output_fps is not None:
            data["output_fps"] = int(self.output_fps)
        if self.output_codec is not None:
            data["output_codec"] = self.output_codec
        return data


@dataclass(frozen=True, slots=True)
class MaterializedReferenceExample:
    """One fully resolved reference example ready for conversation bootstrap."""

    index: int
    inputs: tuple[MaterializedReferenceInput, ...]
    source_description: str
    rendered_description: str
    progress_markers: tuple[ProgressMarker, ...]

    def to_message(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"### Reference Example {self.index}\n\nReference input(s) follow.",
            }
        ]
        content.extend(reference_input.media_content() for reference_input in self.inputs)
        content.append(
            {
                "type": "text",
                "text": f"Description:\n\n{self.rendered_description}",
            }
        )
        return {"role": "user", "content": content}

    def to_manifest_dict(self, *, run_dir: Path) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "inputs": [item.to_manifest_dict(run_dir=run_dir) for item in self.inputs],
            "source_description": self.source_description,
            "rendered_description": self.rendered_description,
            "progress_markers": [marker.to_dict() for marker in self.progress_markers],
        }


def parse_reference_examples(
    raw_examples: Any,
    *,
    source_path: Path,
    field_name: str,
) -> tuple[ReferenceExampleConfig, ...]:
    """Parse task-sidecar reference examples and resolve their source paths."""

    if raw_examples is None:
        return ()
    if not isinstance(raw_examples, list):
        raise TypeError(f"{source_path}: {field_name} must be a list.")

    examples: list[ReferenceExampleConfig] = []
    for example_index, raw_example in enumerate(raw_examples):
        example_field = f"{field_name}[{example_index}]"
        if not isinstance(raw_example, dict):
            raise TypeError(f"{source_path}: {example_field} must be a mapping.")
        raw_inputs = raw_example.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise ValueError(f"{source_path}: {example_field}.inputs must be a non-empty list.")
        raw_description = raw_example.get("description")
        if not isinstance(raw_description, str) or not raw_description.strip():
            raise ValueError(f"{source_path}: {example_field}.description must be a non-empty string.")

        inputs: list[ReferenceInputConfig] = []
        for input_index, raw_input in enumerate(raw_inputs):
            input_field = f"{example_field}.inputs[{input_index}]"
            if not isinstance(raw_input, dict):
                raise TypeError(f"{source_path}: {input_field} must be a mapping.")
            input_type = str(raw_input.get("type") or "").strip()
            if input_type not in _SUPPORTED_INPUT_TYPES:
                supported = ", ".join(sorted(_SUPPORTED_INPUT_TYPES))
                raise ValueError(
                    f"{source_path}: {input_field}.type={input_type!r} is unsupported; "
                    f"supported types are: {supported}."
                )
            raw_media_path = raw_input.get("path")
            if not isinstance(raw_media_path, str) or not raw_media_path.strip():
                raise ValueError(f"{source_path}: {input_field}.path must be a non-empty string.")
            media_path = Path(raw_media_path).expanduser()
            if not media_path.is_absolute():
                media_path = source_path.parent / media_path
            media_path = media_path.resolve()
            if not media_path.is_file():
                raise FileNotFoundError(f"{source_path}: {input_field}.path does not exist: {media_path}")
            inputs.append(
                ReferenceInputConfig(
                    type=input_type,  # type: ignore[arg-type]
                    path=media_path,
                )
            )

        examples.append(
            ReferenceExampleConfig(
                inputs=tuple(inputs),
                description=raw_description.strip(),
            )
        )
    return tuple(examples)


def materialize_reference_examples(
    examples: Sequence[ReferenceExampleConfig],
    *,
    run_dir: Path,
    value_max_points: int,
    video_fps: int,
    video_codec: str,
) -> tuple[MaterializedReferenceExample, ...]:
    """Create immutable run-local assets and a manifest for selected references."""

    run_root = Path(run_dir).expanduser().resolve()
    output_root = run_root / "assets" / "references"
    output_root.mkdir(parents=True, exist_ok=True)
    materialized: list[MaterializedReferenceExample] = []

    for example_index, example in enumerate(examples, start=1):
        example_dir = output_root / f"reference_{example_index:04d}"
        example_dir.mkdir(parents=True, exist_ok=True)
        resolved_inputs = tuple(
            _materialize_reference_input(
                configured_input,
                output_dir=example_dir,
                input_index=input_index,
                video_fps=video_fps,
                video_codec=video_codec,
            )
            for input_index, configured_input in enumerate(example.inputs, start=1)
        )
        rendered_description, markers = render_progress_markers(
            example.description,
            value_max_points=value_max_points,
            field_name=f"reference_examples[{example_index - 1}].description",
        )
        materialized.append(
            MaterializedReferenceExample(
                index=example_index,
                inputs=resolved_inputs,
                source_description=example.description,
                rendered_description=rendered_description,
                progress_markers=markers,
            )
        )

    manifest = {
        "value_max_points": int(value_max_points),
        "examples": [example.to_manifest_dict(run_dir=run_root) for example in materialized],
    }
    (run_root / "reference_examples.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tuple(materialized)


def render_progress_markers(
    description: str,
    *,
    value_max_points: int,
    field_name: str,
) -> tuple[str, tuple[ProgressMarker, ...]]:
    """Compile all ``[[progress:RATIO]]`` markers into configured point values."""

    max_points = int(value_max_points)
    if max_points <= 0:
        raise ValueError("value_max_points must be positive.")
    source = str(description)
    rendered_parts: list[str] = []
    markers: list[ProgressMarker] = []
    cursor = 0

    while True:
        start = source.find(_PROGRESS_MARKER_PREFIX, cursor)
        if start < 0:
            rendered_parts.append(source[cursor:])
            break
        rendered_parts.append(source[cursor:start])
        value_start = start + len(_PROGRESS_MARKER_PREFIX)
        end = source.find(_PROGRESS_MARKER_SUFFIX, value_start)
        if end < 0:
            raise ValueError(
                f"{field_name} has an unterminated progress marker at character {start}."
            )
        raw_ratio = source[value_start:end].strip()
        try:
            ratio = Decimal(raw_ratio)
        except InvalidOperation as exc:
            raise ValueError(
                f"{field_name} has an invalid progress ratio {raw_ratio!r} at character {start}."
            ) from exc
        if not ratio.is_finite() or ratio < 0 or ratio > 1:
            raise ValueError(
                f"{field_name} progress ratio must be finite and within [0, 1], got {raw_ratio!r}."
            )
        points = int((ratio * Decimal(max_points)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        rendered_parts.append(f"{points} progress points on the 0-{max_points} scale")
        marker_end = end + len(_PROGRESS_MARKER_SUFFIX)
        markers.append(
            ProgressMarker(
                ratio=ratio,
                progress_points=points,
                start_offset=start,
                end_offset=marker_end,
            )
        )
        cursor = marker_end

    # Catch malformed marker-like text that was not consumed by the strict scanner.
    rendered = "".join(rendered_parts)
    if _PROGRESS_MARKER_PREFIX in rendered:
        malformed_at = rendered.index(_PROGRESS_MARKER_PREFIX)
        raise ValueError(f"{field_name} has a malformed progress marker near character {malformed_at}.")
    return rendered, tuple(markers)


def _materialize_reference_input(
    configured_input: ReferenceInputConfig,
    *,
    output_dir: Path,
    input_index: int,
    video_fps: int,
    video_codec: str,
) -> MaterializedReferenceInput:
    source = configured_input.path.resolve()
    source_sha256 = _sha256_file(source)
    if configured_input.type == "image":
        _validate_image(source)
        suffix = source.suffix.lower() or ".png"
        output = output_dir / f"input_{input_index:04d}{suffix}"
        shutil.copy2(source, output)
        return MaterializedReferenceInput(
            type="image",
            path=output,
            source_sha256=source_sha256,
            output_sha256=_sha256_file(output),
            materialization="copied",
        )

    output = output_dir / f"input_{input_index:04d}.mp4"
    output, frame_count = transcode_reference_video(
        source_path=source,
        output_path=output,
        fps=video_fps,
        codec=video_codec,
    )
    return MaterializedReferenceInput(
        type="video",
        path=output,
        source_sha256=source_sha256,
        output_sha256=_sha256_file(output),
        materialization="transcoded",
        output_frame_count=frame_count,
        output_fps=int(video_fps),
        output_codec=str(video_codec),
    )


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - preserve the source path in a clear startup error.
        raise ValueError(f"Reference image cannot be decoded: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
