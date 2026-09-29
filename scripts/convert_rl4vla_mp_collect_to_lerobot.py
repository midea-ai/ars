#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_EXPECTED_IMAGE_SHAPE = (480, 640, 3)
DEFAULT_STATE_DIM = 8
DEFAULT_ACTION_DIM = 7
NATURAL_SORT_TOKEN_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class EpisodeSummary:
    """Small per-episode summary used for validation logs."""

    path: Path
    num_frames: int
    instruction: str
    terminal_success: bool
    image_shape: tuple[int, int, int]


@dataclass(frozen=True)
class LoadedEpisode:
    """Decoded RL4VLA episode payload shared by all exported variants."""

    path: Path
    instruction: str
    states: np.ndarray
    actions: np.ndarray
    images: list[np.ndarray]
    terminal_success: bool
    image_shape: tuple[int, int, int]


@dataclass(frozen=True)
class MismatchFields:
    """Validated semantic mismatch metadata for synthetic failure variants."""

    wrong_instruction: str
    mismatch_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RL4VLA ManiSkill mp_collect .npz trajectories into a "
            "local LeRobot v2.1 dataset with embedded RGB images."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Path to one .npz file, a directory containing .npz files, or a "
            "higher-level mp_collect directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "New local LeRobot dataset directory to create; must not already exist."
        ),
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="widowx",
        help=(
            "LeRobot robot_type metadata; defaults to the collector's WidowX arm."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help=(
            "Dataset FPS. The RL4VLA collection command sets ManiSkill "
            "control_freq=5, so 5 is the default."
        ),
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of LeRobot image-writing threads.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Number of LeRobot image-writing processes; 0 uses threads only.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help=(
            "Keep terminal-failure trajectories. By default, only successful "
            "trajectories are converted."
        ),
    )
    parser.add_argument(
        "--add-mismatch-failure",
        action="store_true",
        help=(
            "For each terminal-success trajectory, also export a semantic "
            "failure episode with task=data['wrong_instruction'] and the same "
            "image/state/action sequence. Requires RL4VLA mp_collect "
            "schema with wrong_instruction and mismatch_type."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print a summary without writing LeRobot data.",
    )
    return parser.parse_args()


def discover_npz_files(input_root: Path) -> list[Path]:
    """Discover trajectory files while supporting both leaf and parent paths."""

    root = input_root.expanduser()
    if root.is_file():
        if root.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {root}")
        return [root]

    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    direct_files = sorted(
        root.glob("*.npz"),
        key=lambda path: natural_sort_path_key(path, root=root),
    )
    if direct_files:
        return direct_files
    return sorted(
        root.glob("**/*.npz"),
        key=lambda path: natural_sort_path_key(path, root=root),
    )


def natural_sort_path_key(
    path: Path,
    *,
    root: Path,
) -> tuple[tuple[tuple[int, int | str], ...], str]:
    """Sort paths by relative path text while treating digit runs as integers."""

    relative = path.relative_to(root).as_posix()
    parts: list[tuple[int, int | str]] = []
    for token in NATURAL_SORT_TOKEN_RE.split(relative):
        if not token:
            continue
        if token.isdigit():
            parts.append((1, int(token)))
        else:
            parts.append((0, token.casefold()))
    return tuple(parts), relative


def import_lerobot_dataset():
    """Load the v2.1 API only for writing; dry runs do not require LeRobot."""

    try:
        from lerobot.common.datasets.lerobot_dataset import (
            CODEBASE_VERSION,
            LeRobotDataset,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the LeRobot dependency pinned in requirements.txt before "
            "writing a dataset. Input validation is available with --dry-run."
        ) from exc
    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(
            f"Expected LeRobot dataset format v2.1, got {CODEBASE_VERSION!r}. "
            "Install the version pinned in requirements.txt."
        )
    return LeRobotDataset


def dataset_features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    """Describe collector arrays; LeRobot adds frame, episode, and task indices."""

    features = {}
    for name, width in (("state", DEFAULT_STATE_DIM), ("actions", DEFAULT_ACTION_DIM)):
        features[name] = {"dtype": "float32", "shape": (width,), "names": [name]}
    for name in ("done", "is_success"):
        features[name] = {"dtype": "bool", "shape": (1,), "names": [name]}
    features["image"] = {
        "dtype": "image",
        "shape": image_shape,
        "names": ["height", "width", "channel"],
    }
    return features


def load_npz_dict(path: Path) -> dict[str, Any]:
    """Load the RL4VLA npz payload saved as arr_0.item()."""

    with np.load(path, allow_pickle=True) as payload:
        if "arr_0" not in payload:
            raise ValueError(f"{path} does not contain arr_0.")
        data = payload["arr_0"]
        if data.shape == ():
            return data.item()
        if data.size == 1:
            return data.reshape(()).item()
    raise ValueError(f"{path} does not contain a scalar dict payload.")


def normalize_text_field(value: Any, *, field_name: str) -> str:
    """Normalize scalar/list text fields saved in RL4VLA npz payloads."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is empty.")
    return text


def normalize_instruction(value: Any) -> str:
    """Normalize RL4VLA's instruction field to the task string LeRobot expects."""

    return normalize_text_field(value, field_name="instruction")


def to_numpy_item(value: Any, *, dtype: np.dtype | type) -> np.ndarray:
    """Convert tensors, lists, and arrays to a CPU numpy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def to_2d_float_array(value: Any, *, key: str, width: int) -> np.ndarray:
    """Convert a trajectory field such as state/action into [T, width]."""

    if isinstance(value, np.ndarray) and value.dtype != object:
        array = np.asarray(value, dtype=np.float32)
    else:
        array = np.stack(
            [to_numpy_item(item, dtype=np.float32) for item in list(value)],
            axis=0,
        ).astype(np.float32)

    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Expected {key} shape [T, {width}], got {array.shape}.")
    return array


def scalar_bool(value: Any) -> bool:
    """Read numpy/tensor scalar-like values from ManiSkill info dicts."""

    array = to_numpy_item(value, dtype=np.bool_)
    if array.size != 1:
        array = array.reshape(-1)[:1]
    return bool(array.item())


def terminal_success(info: Any) -> bool:
    """Return the terminal success flag from RL4VLA's per-step info list."""

    info_list = list(info)
    if not info_list:
        raise ValueError("info is empty.")
    last_info = info_list[-1]
    if not isinstance(last_info, dict) or "success" not in last_info:
        raise ValueError("The terminal info record does not contain 'success'.")
    return scalar_bool(last_info["success"])


def decode_jpeg_bytes(value: Any) -> np.ndarray:
    """Decode JPEG bytes produced when RL4VLA stores encoded images."""

    if isinstance(value, np.ndarray):
        value = value.tobytes()
    with Image.open(io.BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_to_uint8(value: Any) -> np.ndarray:
    """Convert PIL/ndarray/JPEG-byte images to uint8 HWC RGB arrays."""

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)

    if isinstance(value, (bytes, bytearray)):
        return decode_jpeg_bytes(value)

    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == np.uint8:
        return decode_jpeg_bytes(array)

    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {array.shape}.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def validate_image_shape(
    image: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] = DEFAULT_EXPECTED_IMAGE_SHAPE,
) -> None:
    if tuple(image.shape) != expected_shape:
        raise ValueError(
            f"Expected image shape {expected_shape}, got {tuple(image.shape)}."
        )


def validate_mismatch_fields(path: Path, data: dict[str, Any]) -> MismatchFields:
    """Validate RL4VLA mismatch metadata only when a mismatch variant is needed."""

    mismatch_missing = {"wrong_instruction", "mismatch_type"} - set(data)
    if mismatch_missing:
        raise ValueError(
            f"{path} is missing required mismatch keys for "
            f"--add-mismatch-failure: {sorted(mismatch_missing)}"
        )

    wrong_instruction = normalize_text_field(
        data["wrong_instruction"],
        field_name="wrong_instruction",
    )
    mismatch_type = normalize_text_field(
        data["mismatch_type"],
        field_name="mismatch_type",
    )
    if mismatch_type not in {"object", "plate"}:
        raise ValueError(
            f"{path} has unsupported mismatch_type={mismatch_type!r}; "
            "expected 'object' or 'plate'."
        )
    return MismatchFields(
        wrong_instruction=wrong_instruction,
        mismatch_type=mismatch_type,
    )


def decode_episode_payload(
    path: Path,
    data: dict[str, Any],
) -> tuple[LoadedEpisode, EpisodeSummary]:
    """Decode the base RL4VLA fields shared by original and synthetic episodes."""

    missing = {"image", "state", "instruction", "action", "info"} - set(data)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")

    instruction = normalize_instruction(data["instruction"])
    states = to_2d_float_array(data["state"], key="state", width=DEFAULT_STATE_DIM)
    actions = to_2d_float_array(
        data["action"],
        key="action",
        width=DEFAULT_ACTION_DIM,
    )
    images = list(data["image"])
    infos = list(data["info"])

    num_frames = len(images)
    if not (num_frames == len(states) == len(actions) == len(infos)):
        raise ValueError(
            f"{path} has misaligned lengths: "
            f"image={num_frames}, state={len(states)}, "
            f"action={len(actions)}, info={len(infos)}."
        )
    if num_frames == 0:
        raise ValueError(f"{path} has no frames.")

    is_success = terminal_success(infos)
    decoded_images = [image_to_uint8(image) for image in images]
    for image in decoded_images:
        validate_image_shape(image)

    summary = EpisodeSummary(
        path=path,
        num_frames=num_frames,
        instruction=instruction,
        terminal_success=is_success,
        image_shape=tuple(decoded_images[0].shape),
    )
    episode = LoadedEpisode(
        path=path,
        instruction=instruction,
        states=states,
        actions=actions,
        images=decoded_images,
        terminal_success=is_success,
        image_shape=tuple(decoded_images[0].shape),
    )
    return episode, summary


def build_frame_sequence(
    episode: LoadedEpisode,
    *,
    task: str,
    is_success: bool,
) -> list[dict[str, Any]]:
    """Build one LeRobot episode frame sequence from a decoded payload."""

    frames: list[dict[str, Any]] = []
    num_frames = len(episode.images)
    for frame_idx in range(num_frames):
        # Original and mismatched tasks share observations but have different
        # episode-level success labels, repeated on every frame for readers.
        frames.append(
            {
                "image": episode.images[frame_idx],
                "state": episode.states[frame_idx].astype(np.float32),
                "actions": episode.actions[frame_idx].astype(np.float32),
                "task": task,
                "done": np.array([frame_idx == num_frames - 1], dtype=bool),
                "is_success": np.array([is_success], dtype=bool),
            }
        )
    return frames


def build_episode_frame_variants(
    path: Path,
    *,
    include_failures: bool,
    add_mismatch_failure: bool,
) -> tuple[list[list[dict[str, Any]]], EpisodeSummary]:
    """Load one RL4VLA episode and return all LeRobot episode variants."""

    data = load_npz_dict(path)
    episode, summary = decode_episode_payload(path, data)
    if not episode.terminal_success and not include_failures:
        return [], summary

    variants = [
        build_frame_sequence(
            episode,
            task=episode.instruction,
            is_success=episode.terminal_success,
        )
    ]
    if add_mismatch_failure and episode.terminal_success:
        mismatch = validate_mismatch_fields(path, data)
        variants.append(
            build_frame_sequence(
                episode,
                task=mismatch.wrong_instruction,
                is_success=False,
            )
        )
    return variants, summary


def print_dataset_info(output_dir: Path) -> None:
    """Print a compact summary of the produced LeRobot metadata."""

    info_path = output_dir / "meta" / "info.json"
    if not info_path.is_file():
        print(f"[convert] WARNING: missing {info_path}")
        return
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    features = sorted(info.get("features", {}).keys())
    print("[convert] wrote:", output_dir)
    print("[convert] total_episodes:", info.get("total_episodes"))
    print("[convert] total_frames:", info.get("total_frames"))
    print("[convert] fps:", info.get("fps"))
    print("[convert] features:", features)


def release_lerobot_hf_cache(dataset: Any) -> None:
    """Drop LeRobot's cumulative in-memory HF table after parquet is on disk."""

    # LeRobot appends every saved episode to dataset.hf_dataset before writing
    # the per-episode parquet. This converter only needs the parquet files and
    # metadata, so reset the table to an empty dataset to keep RSS bounded.
    dataset.hf_dataset = dataset.create_hf_dataset()
    gc.collect()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive.")
    if args.image_writer_threads < 0 or args.image_writer_processes < 0:
        raise ValueError("Image writer thread/process counts must be nonnegative.")
    files = discover_npz_files(args.input_root)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {args.input_root}.")

    output_dir = args.output_dir.expanduser()
    # Keep the output separate from input trajectories, including through symlinks.
    if any(path.resolve().is_relative_to(output_dir.resolve()) for path in files):
        raise ValueError("--output-dir must not contain the input trajectories.")
    if output_dir.is_symlink():
        raise ValueError("--output-dir must not be a symbolic link.")
    if not args.dry_run and output_dir.exists():
        raise FileExistsError(
            f"Output path already exists: {output_dir}. "
            "Choose a new --output-dir."
        )
    dataset_cls = None if args.dry_run else import_lerobot_dataset()

    dataset = None
    converted = 0
    skipped = 0
    total_frames = 0
    summaries: list[EpisodeSummary] = []

    try:
        for path in files:
            frame_variants, summary = build_episode_frame_variants(
                path,
                include_failures=args.include_failures,
                add_mismatch_failure=args.add_mismatch_failure,
            )
            summaries.append(summary)
            if not frame_variants:
                skipped += 1
                continue

            if dataset_cls is not None and dataset is None:
                # Create parents only after validating the first retained episode.
                # LeRobot creates the dataset directory itself.
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                dataset = dataset_cls.create(
                    repo_id="local/rl4vla-mp-collect",
                    root=output_dir,
                    robot_type=args.robot_type,
                    fps=args.fps,
                    features=dataset_features(summary.image_shape),
                    use_videos=False,
                    image_writer_threads=args.image_writer_threads,
                    image_writer_processes=args.image_writer_processes,
                )
            if dataset is not None:
                for frames in frame_variants:
                    for frame in frames:
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    release_lerobot_hf_cache(dataset)

            converted += len(frame_variants)
            total_frames += sum(len(frames) for frames in frame_variants)
    finally:
        if dataset is not None:
            # save_episode() persists image bytes and metadata before returning.
            # Stop background writers even if validation of a later input fails.
            dataset.stop_image_writer()

    success_count = sum(1 for summary in summaries if summary.terminal_success)
    print("[convert] input_files:", len(files))
    print("[convert] converted_episodes:", converted)
    print("[convert] skipped_episodes:", skipped)
    print("[convert] successful_inputs:", success_count)
    print("[convert] total_frames:", total_frames)
    if summaries:
        first = summaries[0]
        print("[convert] first_file:", first.path)
        print("[convert] first_instruction:", first.instruction)
        print("[convert] first_image_shape:", first.image_shape)

    if args.dry_run:
        print("[convert] dry-run only; no dataset was written.")
    elif converted:
        print_dataset_info(output_dir)
    else:
        print("[convert] no eligible episodes; no dataset was written.")


if __name__ == "__main__":
    main()
