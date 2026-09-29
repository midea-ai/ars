"""LeRobot episode source for reward-agent batch serving."""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .is_success import read_lerobot_is_success
from .schemas import ProgressRow


def iter_lerobot_episode_sources(
    lerobot_root: str | Path,
    *,
    image_key: str = "image",
) -> list["LeRobotEpisodeSource"]:
    """Resolve exactly one LeRobot leaf into episode sources."""

    sources: list[LeRobotEpisodeSource] = []
    dataset_root = resolve_single_lerobot_dataset_root(lerobot_root)
    info = _load_json(dataset_root / "meta" / "info.json")
    episodes = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    if not episodes:
        total_episodes = int(info.get("total_episodes", 0))
        episodes = [{"episode_index": index} for index in range(total_episodes)]
    for episode_row in episodes:
        sources.append(
            LeRobotEpisodeSource(
                dataset_root=dataset_root,
                episode_index=int(episode_row["episode_index"]),
                info=info,
                episode_row=episode_row,
                image_key=image_key,
            )
        )
    if not sources:
        raise ValueError(f"No LeRobot episodes found under {Path(lerobot_root).expanduser()}.")
    return sources


def select_lerobot_episode_sources(
    lerobot_root: str | Path,
    episode_indices: Sequence[int] | None = None,
    limit: int | None = None,
    *,
    image_key: str = "image",
) -> list["LeRobotEpisodeSource"]:
    """Load, validate, order, and filter one LeRobot dataset leaf.

    Metadata duplicates are rejected even when the duplicate episode was not
    requested.  This avoids allowing an explicit filter to hide an ambiguous
    dataset.  Explicit requests preserve caller order; otherwise metadata is
    normalized to ascending ``episode_index`` order.
    """

    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit!r}.")

    sources = iter_lerobot_episode_sources(lerobot_root, image_key=image_key)
    by_index: dict[int, LeRobotEpisodeSource] = {}
    duplicate_metadata: set[int] = set()
    for source in sources:
        episode_index = int(source.episode_index)
        if episode_index in by_index:
            duplicate_metadata.add(episode_index)
        else:
            by_index[episode_index] = source
    if duplicate_metadata:
        raise ValueError(
            "LeRobot metadata contains duplicate episode_index values: "
            f"{sorted(duplicate_metadata)}."
        )

    if episode_indices is None:
        selected = [by_index[index] for index in sorted(by_index)]
    else:
        requested: list[int] = []
        for position, raw_index in enumerate(episode_indices):
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise TypeError(
                    f"episode_indices[{position}] must be an integer, got {raw_index!r}."
                )
            requested.append(int(raw_index))
        requested_seen: set[int] = set()
        duplicate_requested: set[int] = set()
        for index in requested:
            if index in requested_seen:
                duplicate_requested.add(index)
            requested_seen.add(index)
        if duplicate_requested:
            raise ValueError(
                "Requested episode_indices contain duplicate values: "
                f"{sorted(duplicate_requested)}."
            )
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise ValueError(
                f"Requested episode_index values are missing: {missing}. "
                f"Available episode_index values: {sorted(by_index)}."
            )
        selected = [by_index[index] for index in requested]

    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Episode selection produced no LeRobot episodes.")
    return selected


def resolve_single_lerobot_dataset_root(lerobot_root: str | Path) -> Path:
    """Resolve a run-batch root and reject collection roots with multiple leaves."""

    candidate = Path(lerobot_root).expanduser()
    roots = resolve_lerobot_dataset_roots(candidate)
    if len(roots) == 1:
        return roots[0]

    examples = "\n".join(f"  - {root}" for root in roots[:10])
    suffix = "" if len(roots) <= 10 else f"\n  ... and {len(roots) - 10} more"
    raise ValueError(
        "reward-agent run-batch currently expects exactly one LeRobot dataset leaf. "
        f"Found {len(roots)} leaves under {candidate.resolve(strict=False)}. "
        "Pass a concrete leaf path as --lerobot-root, for example:\n"
        f"{examples}{suffix}"
    )


def resolve_lerobot_dataset_roots(lerobot_root: str | Path) -> list[Path]:
    """Return concrete LeRobot leaves under a dataset or collection root."""

    candidate = Path(lerobot_root).expanduser()
    if (candidate / "meta" / "info.json").is_file():
        return [candidate.resolve()]
    matches = sorted(candidate.glob("**/meta/info.json"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any meta/info.json under {candidate.resolve(strict=False)}."
        )
    return [match.parent.parent.resolve() for match in matches]


class LeRobotEpisodeSource:
    """Read one LeRobot episode without materializing a sanitized case pack."""

    video_path: Path | None = None
    source_fps: float | None

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        episode_index: int,
        info: dict[str, Any] | None = None,
        episode_row: dict[str, Any] | None = None,
        image_key: str = "image",
        progress_rows: Sequence[ProgressRow] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.source_root = self.dataset_root
        self.episode_index = int(episode_index)
        self.image_key = str(image_key)
        self._info = dict(info) if info is not None else _load_json(self.dataset_root / "meta" / "info.json")
        self.source_fps = _positive_float_or_none(self._info.get("fps"))
        self._episode_row = (
            dict(episode_row)
            if episode_row is not None
            else self._load_episode_row(self.episode_index)
        )
        # External proposals are attached only after the LeRobot metadata has
        # been validated. Keeping them on the source lets the session
        # consume the normal ``read_progress_rows`` protocol without creating
        # temporary case-pack files.
        self._progress_rows = (
            None
            if progress_rows is None
            else tuple(progress_rows)
        )
        self.source_subset = _source_subset_for_leaf(self.dataset_root)
        self.source_traj_id = f"{_slugify_text(self.source_subset)}__episode_{self.episode_index:06d}"
        self.case_id = f"episode_{self.episode_index:06d}"
        self.episode_path = self._resolve_episode_path()
        self.num_frames = self._resolve_num_frames()
        self._is_success_cache: bool | None = None

    def read_task(self) -> str:
        tasks = self._episode_row.get("tasks")
        if isinstance(tasks, list) and tasks:
            return str(tasks[0])
        if isinstance(tasks, str) and tasks.strip():
            return tasks.strip()

        task_index = self._episode_row.get("task_index")
        if task_index is not None:
            task = self._load_task_by_index(int(task_index))
            if task:
                return task

        # Some LeRobot exports only store task_index in the parquet rows.
        try:
            task_df = pd.read_parquet(self.episode_path, columns=["task_index"])
        except Exception:
            task_df = pd.DataFrame()
        if not task_df.empty:
            task = self._load_task_by_index(int(task_df["task_index"].iloc[0]))
            if task:
                return task

        return f"episode {self.episode_index}"

    def read_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "num_frames": int(self.num_frames),
            "camera_key": self.image_key,
            "episode_index": int(self.episode_index),
            "source_traj_id": self.source_traj_id,
            "dataset_root": str(self.dataset_root),
            "frame_indexing_rule": "LeRobot transition rows, frame_index=0..T-1.",
        }
        feature = dict(self._info.get("features", {}).get(self.image_key, {}) or {})
        shape = feature.get("shape")
        if isinstance(shape, list) and shape:
            metadata["frame_shape_hwc"] = [int(item) for item in shape]
        return metadata

    def read_progress_rows(self) -> list[ProgressRow] | None:
        if self._progress_rows is None:
            return None
        return list(self._progress_rows)

    def with_progress_rows(
        self,
        progress_rows: Sequence[ProgressRow],
    ) -> "LeRobotEpisodeSource":
        """Return this episode source with an immutable external proposal attached."""

        return LeRobotEpisodeSource(
            dataset_root=self.dataset_root,
            episode_index=self.episode_index,
            info=self._info,
            episode_row=self._episode_row,
            image_key=self.image_key,
            progress_rows=progress_rows,
        )

    def read_is_success(self) -> bool:
        """Read and cache the episode-level terminal success label."""

        if self._is_success_cache is None:
            self._is_success_cache = read_lerobot_is_success(
                self.episode_path,
                expected_length=self.num_frames,
            )
        return self._is_success_cache

    def require_image_key(self, image_key: str) -> dict[str, Any]:
        """Validate and return one image feature declared by this dataset."""

        key = str(image_key).strip()
        if not key:
            raise ValueError("image_key must be non-empty.")
        raw_feature = self._info.get("features", {}).get(key)
        if not isinstance(raw_feature, dict):
            available = sorted(str(item) for item in self._info.get("features", {}))
            raise ValueError(
                f"{self.dataset_root}: episode {self.episode_index} requires image "
                f"feature {key!r}, but it is absent. Available features: {available}."
            )
        feature = dict(raw_feature)
        if feature.get("dtype") != "image":
            raise ValueError(
                f"{self.dataset_root}: feature {key!r} must have dtype='image', "
                f"got {feature.get('dtype')!r}."
            )
        shape = feature.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
            or int(shape[-1]) != 3
        ):
            raise ValueError(
                f"{self.dataset_root}: image feature {key!r} must declare HWC "
                f"three-channel shape [H, W, 3], got {shape!r}."
            )
        return feature

    def read_selected_frames(
        self,
        frame_indices: Sequence[int],
        *,
        image_key: str | None = None,
    ) -> list[np.ndarray]:
        """Decode selected rows from the primary or an explicitly named camera."""

        if not frame_indices:
            raise ValueError("frame_indices must be non-empty.")
        selected_image_key = self.image_key if image_key is None else str(image_key)
        self.require_image_key(selected_image_key)
        targets = [int(frame) for frame in frame_indices]
        if min(targets) < 0 or max(targets) >= int(self.num_frames):
            raise IndexError(
                f"Requested frames outside LeRobot episode range [0, {self.num_frames - 1}]: {targets}"
            )

        try:
            df = pd.read_parquet(
                self.episode_path,
                columns=["frame_index", selected_image_key],
            )
        except Exception as exc:
            raise ValueError(
                f"{self.episode_path}: could not read configured image feature "
                f"{selected_image_key!r}."
            ) from exc
        if len(df) != int(self.num_frames):
            raise ValueError(
                f"{self.episode_path}: expected {self.num_frames} rows, got {len(df)}."
            )
        frames_by_index: dict[int, np.ndarray] = {}
        for _, row in df.iterrows():
            frame_index = int(row["frame_index"])
            if frame_index in targets:
                frames_by_index[frame_index] = _decode_image_cell(
                    row[selected_image_key],
                    dataset_root=self.dataset_root,
                )
        missing = [frame for frame in targets if frame not in frames_by_index]
        if missing:
            raise IndexError(
                f"{self.episode_path}: image feature {selected_image_key!r} is "
                f"missing requested frame(s): {missing}"
            )
        return [frames_by_index[frame] for frame in targets]

    def _resolve_episode_path(self) -> Path:
        chunks_size = int(self._info.get("chunks_size", 1000))
        episode_chunk = int(self.episode_index) // chunks_size
        data_pattern = self._info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        candidate = self.dataset_root / str(data_pattern).format(
            episode_chunk=episode_chunk,
            episode_index=int(self.episode_index),
        )
        if candidate.is_file():
            return candidate.resolve()
        matches = sorted(self.dataset_root.glob(f"data/**/episode_{self.episode_index:06d}.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{self.dataset_root}: expected exactly one parquet for episode "
                f"{self.episode_index}, found {len(matches)}."
            )
        return matches[0].resolve()

    def _resolve_num_frames(self) -> int:
        for key in ("length", "num_frames", "episode_length"):
            if self._episode_row.get(key) is not None:
                count = int(self._episode_row[key])
                if count <= 0:
                    raise ValueError(f"{self.dataset_root}: episode {self.episode_index} has length {count}.")
                return count
        frame_df = pd.read_parquet(self.episode_path, columns=["frame_index"])
        count = int(len(frame_df))
        if count <= 0:
            raise ValueError(f"{self.episode_path}: episode has no transition rows.")
        return count

    def _load_episode_row(self, episode_index: int) -> dict[str, Any]:
        for row in _load_jsonl(self.dataset_root / "meta" / "episodes.jsonl"):
            if int(row["episode_index"]) == int(episode_index):
                return row
        return {"episode_index": int(episode_index)}

    def _load_task_by_index(self, task_index: int) -> str | None:
        tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
        if not tasks_path.is_file():
            return None
        for row in _load_jsonl(tasks_path):
            if int(row.get("task_index", -1)) == int(task_index):
                task = str(row.get("task", "")).strip()
                return task or None
        return None


def _decode_image_cell(value: Any, *, dataset_root: Path) -> np.ndarray:
    """Decode common LeRobot image cell encodings into RGB uint8 arrays."""

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _decode_image_bytes(value["bytes"])
        if value.get("path"):
            path = Path(str(value["path"])).expanduser()
            if not path.is_absolute():
                path = dataset_root / path
            return _decode_image_bytes(path.read_bytes())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _decode_image_bytes(bytes(value))
    if isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        return _decode_image_bytes(path.read_bytes())
    array = np.asarray(value)
    if array.ndim == 3:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[..., :3])
    raise ValueError(f"Unsupported LeRobot image cell type: {type(value)!r}")


def _decode_image_bytes(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object.")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}: line {line_number} must decode to a JSON object.")
            rows.append(payload)
    return rows


def _positive_float_or_none(value: Any) -> float | None:
    """Return a positive LeRobot metadata fps value when one is present."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _source_subset_for_leaf(dataset_root: Path) -> str:
    if dataset_root.parent.name.startswith("rank_"):
        return f"{dataset_root.parent.name}/{dataset_root.name}"
    return dataset_root.name


def _slugify_text(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_") or "unknown"
