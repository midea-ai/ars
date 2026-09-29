"""Episode-source boundary for reward-agent inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import imageio.v2 as imageio
import numpy as np

from .progress import load_progress_csv
from .is_success import read_metadata_is_success
from .schemas import ProgressRow
from .visual import read_selected_frames


class RewardAgentEpisodeSource(Protocol):
    """Small interface that lets sessions avoid depending on one storage format."""

    case_id: str
    source_traj_id: str
    episode_index: int
    num_frames: int
    source_fps: float | None
    source_root: Path
    video_path: Path | None

    def read_task(self) -> str:
        """Return the natural-language robot instruction."""

    def read_metadata(self) -> dict[str, Any]:
        """Return public episode metadata used by reward-agent tools."""

    def read_progress_rows(self) -> list[ProgressRow] | None:
        """Return optional candidate progress rows in the source frame axis."""

    def read_is_success(self) -> bool:
        """Return the required terminal success ground truth."""

    def read_selected_frames(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        """Return RGB frames for the requested source frame indices."""


class CasePackEpisodeSource:
    """Adapter for the existing sanitized case-pack directory format."""

    def __init__(self, case_dir: str | Path, *, value_max_points: int) -> None:
        self.source_root = Path(case_dir).expanduser().resolve()
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"Case directory not found: {self.source_root}")
        self.case_id = self.source_root.name
        self.source_traj_id = self.case_id
        self.episode_index = int(self.read_metadata().get("episode_index", -1))
        self._value_max_points = int(value_max_points)
        self.source_fps = _positive_float_or_none(self.read_metadata().get("fps"))
        self.video_path = self._resolve_video_path()
        self.num_frames = self._resolve_num_frames()

    def read_task(self) -> str:
        path = self.source_root / "task.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Required case file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def read_metadata(self) -> dict[str, Any]:
        path = self.source_root / "metadata.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        return payload

    def read_progress_rows(self) -> list[ProgressRow] | None:
        progress_path = self.source_root / "progress.csv"
        if not progress_path.is_file():
            return None
        return load_progress_csv(
            progress_path,
            value_max_points=self._value_max_points,
        )

    def read_is_success(self) -> bool:
        """Read the required terminal success label from metadata.json."""

        metadata_path = self.source_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{metadata_path}: is_success_ground_truth.enabled=true requires metadata.json."
            )
        return read_metadata_is_success(
            self.read_metadata(),
            metadata_path=metadata_path,
        )

    def read_selected_frames(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        if self.video_path is None:
            raise FileNotFoundError(f"No video file could be resolved under {self.source_root}.")
        return read_selected_frames(self.video_path, frame_indices)

    def _resolve_video_path(self) -> Path | None:
        metadata = self.read_metadata()
        metadata_video = metadata.get("video_path") or metadata.get("video_file")
        candidates: list[Path] = []
        if metadata_video:
            candidate = Path(str(metadata_video)).expanduser()
            candidates.append(candidate if candidate.is_absolute() else self.source_root / candidate)
        candidates.extend(
            [
                self.source_root / "video.mp4",
                self.source_root / "trajectory.mp4",
            ]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        return None

    def _resolve_num_frames(self) -> int:
        metadata = self.read_metadata()
        metadata_frames = metadata.get("num_frames")
        if metadata_frames is not None:
            count = int(metadata_frames)
            if count <= 0:
                raise ValueError(f"metadata.num_frames must be positive, got {count}.")
            return count
        if self.video_path is None:
            raise ValueError("Cannot infer frame count without metadata.num_frames or a video file.")
        reader = imageio.get_reader(self.video_path)
        try:
            count = int(reader.count_frames())
        finally:
            reader.close()
        if count <= 0:
            raise ValueError(f"Video has no decodable frames: {self.video_path}")
        return count


def _positive_float_or_none(value: Any) -> float | None:
    """Return a positive float metadata value when one is present."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
