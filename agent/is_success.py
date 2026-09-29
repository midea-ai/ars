"""Strict readers for terminal ``is_success`` ground-truth labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def read_metadata_is_success(metadata: Mapping[str, Any], *, metadata_path: Path) -> bool:
    """Read one required JSON boolean from a case-pack metadata object."""

    if "is_success" not in metadata:
        raise ValueError(
            f"{metadata_path}: is_success_ground_truth.enabled=true requires "
            "a boolean 'is_success' field."
        )
    value = metadata["is_success"]
    if not isinstance(value, bool):
        raise TypeError(
            f"{metadata_path}: 'is_success' must be a JSON boolean, "
            f"got {type(value).__name__}."
        )
    return value


def read_lerobot_is_success(episode_path: Path, *, expected_length: int) -> bool:
    """Read and validate one constant boolean label from a LeRobot episode."""

    path = Path(episode_path).expanduser().resolve()
    try:
        # This metadata read runs frequently in short-lived evaluator subprocesses.
        # PyArrow's threaded Parquet path can intermittently abort while those
        # processes are shutting down, so read the single label column serially.
        frame = pd.read_parquet(
            path,
            columns=["is_success"],
            use_threads=False,
        )
    except Exception as exc:  # noqa: BLE001 - normalize parquet/schema failures with episode context.
        raise ValueError(
            f"{path}: is_success_ground_truth.enabled=true requires a readable "
            "'is_success' column."
        ) from exc

    if len(frame) != int(expected_length):
        raise ValueError(
            f"{path}: expected {expected_length} is_success rows, got {len(frame)}."
        )
    if len(frame) == 0:
        raise ValueError(f"{path}: episode has no is_success rows.")

    values = [
        _strict_boolean_scalar(value, label=f"{path}.is_success[{index}]")
        for index, value in enumerate(frame["is_success"].tolist())
    ]
    unique_values = set(values)
    if len(unique_values) != 1:
        raise ValueError(f"{path}: is_success must be constant within an episode.")
    return values[0]


def _strict_boolean_scalar(value: Any, *, label: str) -> bool:
    """Accept scalar-shaped bool values without coercing numbers or strings."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"{label} must contain exactly one boolean scalar, got shape {array.shape}."
        )
    scalar = array.reshape(-1)[0]
    if not isinstance(scalar, (bool, np.bool_)):
        raise TypeError(
            f"{label} must be a boolean scalar, got {type(scalar).__name__}."
        )
    return bool(scalar)
