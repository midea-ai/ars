"""Shared LeRobot and reward-agent batch result indexing helpers."""

from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .is_success import read_lerobot_is_success


PROGRESS_NORMALIZATION_RULE_DIVIDE_BY_SCALE = "divide_by_scale"
PROGRESS_NORMALIZATION_RULE_LEGACY_MIN_MAX = "legacy_min_max"


class LegacyProgressNormalizationWarning(UserWarning):
    """Warn that an old artifact is being evaluated with min-max normalization."""


class ProgressNormalizationOverrideWarning(UserWarning):
    """Warn that a caller-provided scale replaces artifact normalization metadata."""


@dataclass(frozen=True, slots=True)
class LeRobotEpisodeRecord:
    """One LeRobot episode addressed by reward-agent source_traj_id."""

    dataset_root: Path
    source_subset: str
    source_traj_id: str
    episode_index: int
    length: int
    episode_path: Path
    is_success: bool | None = None


@dataclass(frozen=True, slots=True)
class RewardAgentResultEntry:
    """One completed reward-agent batch result aligned to a LeRobot episode."""

    batch_results_path: Path
    batch_results_line: int
    corrected_progress_csv: Path
    source_traj_id: str
    episode_index: int
    num_transition_rows: int
    progress_output_mode: str
    progress_value_type: str
    native_progress_value_type: str
    native_progress_value_type_source: str
    progress_value_min_points: int | float
    progress_value_max_points: int | float
    progress_normalization_rule: str
    progress_normalization_scale_points: int | float | None
    progress_normalization_source: str
    # Resolve optional audit artifacts without requiring unrelated consumers
    # to open them. Probability-space evaluation validates the file itself.
    raw_predictions_json: Path | None = None

    @property
    def progress_normalized_min(self) -> float:
        """Return the lower normalized bound implied by this artifact contract."""

        if self.progress_normalization_scale_points is None:
            return 0.0
        return float(self.progress_value_min_points) / float(
            self.progress_normalization_scale_points
        )

    @property
    def progress_normalized_max(self) -> float:
        """Return the upper normalized bound implied by this artifact contract."""

        if self.progress_normalization_scale_points is None:
            return 1.0
        return float(self.progress_value_max_points) / float(
            self.progress_normalization_scale_points
        )


@dataclass(frozen=True, slots=True)
class RewardAgentProgressCurve:
    """One full curve plus a separately typed native evaluation view."""

    frame_indices: np.ndarray
    progress_points: np.ndarray
    progress_normalized: np.ndarray
    is_model_output: np.ndarray
    sources: np.ndarray
    native_progress_points: np.ndarray

    @property
    def model_output_frame_indices(self) -> np.ndarray:
        """Original LeRobot frame indices at which the model produced values."""

        return self.frame_indices[self.is_model_output]

    @property
    def model_output_progress_points(self) -> np.ndarray:
        """Authoritative native progress values used by model-quality metrics."""

        return self.native_progress_points

    @property
    def model_output_progress_normalized(self) -> np.ndarray:
        """Return loader-validated normalized values at native output rows."""

        return self.progress_normalized[self.is_model_output]

@dataclass(frozen=True, slots=True)
class RewardAgentResultIndex:
    """Completed reward-agent rows plus loading and coverage diagnostics."""

    completed_by_source: dict[str, RewardAgentResultEntry]
    loaded_files: int
    total_rows: int
    completed_rows: int
    ignored_non_completed: int
    coverage_message: str | None = None
    ignored_out_of_scope: int = 0
    inferred_legacy_dense_integer_native_type_rows: int = 0
    inferred_legacy_normalization_rows: int = 0


def load_lerobot_episode_index(
    lerobot_root: Path,
    *,
    read_is_success: bool = False,
) -> dict[str, LeRobotEpisodeRecord]:
    """Index all LeRobot episodes under a leaf or collection root."""

    records: dict[str, LeRobotEpisodeRecord] = {}
    for dataset_root in resolve_lerobot_roots(lerobot_root):
        info = _load_json(dataset_root / "meta" / "info.json")
        episodes = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
        if not episodes:
            total_episodes = int(info.get("total_episodes", 0))
            episodes = [
                {"episode_index": episode_index, "length": None}
                for episode_index in range(total_episodes)
            ]

        source_subset = _source_subset_for_leaf(dataset_root)
        subset_slug = _slugify_text(source_subset)
        for row in episodes:
            episode_index = int(row["episode_index"])
            length = _episode_length_from_row(row, default=None)
            episode_path = _resolve_episode_path(
                dataset_root=dataset_root,
                info=info,
                episode_index=episode_index,
                require_existing=length is None or read_is_success,
            )
            if length is None:
                length = int(pd.read_parquet(episode_path, columns=["frame_index"]).shape[0])
            is_success = (
                read_lerobot_is_success(episode_path, expected_length=int(length))
                if read_is_success
                else None
            )
            source_traj_id = f"{subset_slug}__episode_{episode_index:06d}"
            if source_traj_id in records:
                raise ValueError(f"Duplicate source_traj_id under {lerobot_root}: {source_traj_id}")
            records[source_traj_id] = LeRobotEpisodeRecord(
                dataset_root=dataset_root,
                source_subset=source_subset,
                source_traj_id=source_traj_id,
                episode_index=episode_index,
                length=int(length),
                episode_path=episode_path,
                is_success=is_success,
            )

    if not records:
        raise ValueError(f"No LeRobot episodes found under {lerobot_root}.")
    return records


def load_completed_reward_agent_entries(
    *,
    batch_results: Sequence[Path],
    episodes_by_source: Mapping[str, LeRobotEpisodeRecord],
    lerobot_root: Path,
    included_source_traj_ids: set[str] | None = None,
    allow_partial: bool = False,
    label: str = "reward_agent_runs",
    strict_coverage_hint: str | None = None,
    progress_normalization_scale_points_override: float | None = None,
    require_progress_normalization_scale: bool = False,
    strict_schema: bool = False,
) -> RewardAgentResultIndex:
    """Load completed reward-agent rows and validate coverage.

    Non-completed rows are ignored so failed attempts can be superseded by
    later retry shards. When ``included_source_traj_ids`` is provided, rows for
    other sources under the same LeRobot root are counted and skipped before
    artifact validation. Completed rows in the selected scope remain strict:
    duplicates and mismatched episode metadata fail fast. When strict coverage
    fails, ``strict_coverage_hint`` is appended to the error so each caller can
    point to its actual recovery mechanism. Callers preparing normalized
    training targets may provide a positive scale override and require every
    completed row to resolve to an explicit divide-by-scale contract. The
    default remains backward compatible for evaluation callers that still
    need the historical min-max fallback.

    ``strict_schema`` is used by the live ``run-batch`` proposal path. It
    requires the non-path provenance fields used for proposal alignment
    instead of inferring missing values from legacy rows. ``dataset_root`` is
    retained as recorded provenance only and is not used for identity
    matching, so artifacts remain usable after a dataset is relocated.
    Rows outside ``included_source_traj_ids`` are ignored after their source
    identity is checked, so one proposal file may cover a larger episode set
    than the current batch.
    """

    if not batch_results:
        raise ValueError(f"{label} batch_results is required.")

    if progress_normalization_scale_points_override is None:
        normalization_scale_override = None
    else:
        raw_override = progress_normalization_scale_points_override
        if isinstance(raw_override, bool):
            raise ValueError(
                f"{label} progress_normalization_scale_points override must be "
                "a finite positive number, not a boolean."
            )
        normalization_scale_override = float(raw_override)
        if (
            not np.isfinite(normalization_scale_override)
            or normalization_scale_override <= 0.0
        ):
            raise ValueError(
                f"{label} progress_normalization_scale_points override must be "
                f"finite and greater than 0, got {raw_override!r}."
            )

    included_sources = (
        None
        if included_source_traj_ids is None
        else {str(source_traj_id) for source_traj_id in included_source_traj_ids}
    )
    if included_sources is not None:
        if not included_sources:
            raise ValueError(f"{label} included_source_traj_ids must be non-empty.")
        unknown_included = sorted(included_sources - set(episodes_by_source))
        if unknown_included:
            raise KeyError(
                f"{label} included_source_traj_ids are not present under "
                f"lerobot_root {lerobot_root}: {unknown_included[:5]}."
            )
        coverage_episodes = {
            source_traj_id: episodes_by_source[source_traj_id]
            for source_traj_id in included_sources
        }
    else:
        coverage_episodes = dict(episodes_by_source)

    completed_by_source: dict[str, RewardAgentResultEntry] = {}
    ignored_non_completed = 0
    ignored_out_of_scope = 0
    inferred_legacy_dense_integer_native_type_rows = 0
    inferred_legacy_normalization_rows = 0
    normalization_override_mismatches: list[str] = []
    total_rows = 0
    for raw_batch_results_path in batch_results:
        batch_results_path = raw_batch_results_path.expanduser().resolve()
        if not batch_results_path.is_file():
            raise FileNotFoundError(f"Reward-agent batch_results not found: {batch_results_path}")
        rows = _load_jsonl(batch_results_path)
        if not rows:
            raise ValueError(f"{batch_results_path}: expected at least one batch result row.")

        for line_number, row in enumerate(rows, start=1):
            total_rows += 1
            source_traj_id = str(row.get("source_traj_id", "")).strip()
            if not source_traj_id:
                raise ValueError(f"{batch_results_path}: line {line_number} missing source_traj_id.")
            if source_traj_id not in episodes_by_source:
                raise KeyError(
                    f"{batch_results_path}: line {line_number} source_traj_id {source_traj_id!r} "
                    f"is not present under lerobot_root {lerobot_root}."
                )
            if included_sources is not None and source_traj_id not in included_sources:
                # Scope-excluded rows must not make a selected evaluation depend
                # on unrelated progress files, attempts, or trace artifacts.
                ignored_out_of_scope += 1
                continue

            status = str(row.get("status", "")).strip().lower()
            if status != "completed":
                ignored_non_completed += 1
                continue
            if source_traj_id in completed_by_source:
                previous = completed_by_source[source_traj_id]
                raise ValueError(
                    f"duplicate completed rows for {source_traj_id!r}: "
                    f"first={previous.batch_results_path}:line {previous.batch_results_line}, "
                    f"second={batch_results_path}:line {line_number}."
                )

            episode = episodes_by_source[source_traj_id]
            episode_index = _required_row_int(
                row,
                path=batch_results_path,
                line_number=line_number,
                field_name="episode_index",
            )
            if episode_index != int(episode.episode_index):
                raise ValueError(
                    f"{batch_results_path}: line {line_number} episode_index mismatch for "
                    f"{source_traj_id}: batch={episode_index}, lerobot={episode.episode_index}."
                )

            if strict_schema:
                row_source_subset = str(row.get("source_subset", "")).strip()
                if not row_source_subset:
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} missing source_subset "
                        "required by strict proposal schema."
                    )
                if row_source_subset != episode.source_subset:
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} source_subset mismatch for "
                        f"{source_traj_id}: batch={row_source_subset!r}, "
                        f"lerobot={episode.source_subset!r}."
                    )

            num_transition_rows = _required_row_int(
                row,
                path=batch_results_path,
                line_number=line_number,
                field_name="num_transition_rows",
            )
            if num_transition_rows != int(episode.length):
                raise ValueError(
                    f"{batch_results_path}: line {line_number} num_transition_rows mismatch for "
                    f"{source_traj_id}: batch={num_transition_rows}, lerobot={episode.length}."
                )

            progress_output_mode = str(row.get("progress_output_mode", "")).strip().lower()
            if progress_output_mode not in {"dense", "sampled"}:
                raise ValueError(
                    f"{batch_results_path}: line {line_number} progress_output_mode must be "
                    f"'dense' or 'sampled', got {row.get('progress_output_mode')!r}."
                )
            progress_value_type = str(row.get("progress_value_type", "")).strip().lower()
            if progress_value_type not in {"integer", "float"}:
                raise ValueError(
                    f"{batch_results_path}: line {line_number} progress_value_type must be "
                    f"'integer' or 'float', got {row.get('progress_value_type')!r}."
            )
            if "native_progress_value_type" not in row:
                if strict_schema:
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} missing "
                        "native_progress_value_type required by strict proposal schema."
                    )
                if (
                    progress_output_mode == "dense"
                    and progress_value_type == "integer"
                ):
                    # Dense mode validates that every CSV row is native. For
                    # integer-valued dense curves, the native type can therefore
                    # be inferred when the explicit field is absent.
                    native_progress_value_type = "integer"
                    native_progress_value_type_source = (
                        "inferred_legacy_dense_integer"
                    )
                else:
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} missing "
                        "native_progress_value_type; legacy inference is supported only "
                        "for progress_output_mode='dense' with "
                        "progress_value_type='integer'."
                    )
            else:
                native_progress_value_type = str(
                    row.get("native_progress_value_type")
                ).strip().lower()
                native_progress_value_type_source = "explicit"
            if native_progress_value_type not in {"integer", "float"}:
                raise ValueError(
                    f"{batch_results_path}: line {line_number} "
                    "native_progress_value_type must be 'integer' or 'float', got "
                    f"{row.get('native_progress_value_type')!r}."
                )
            progress_value_min_points = _required_row_number(
                row,
                path=batch_results_path,
                line_number=line_number,
                field_name="progress_value_min_points",
            )
            progress_value_max_points = _required_row_number(
                row,
                path=batch_results_path,
                line_number=line_number,
                field_name="progress_value_max_points",
            )
            if progress_value_type == "integer":
                if not float(progress_value_min_points).is_integer() or not float(
                    progress_value_max_points
                ).is_integer():
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} integer progress bounds "
                        "must also be integers."
                    )
                progress_value_min_points = int(progress_value_min_points)
                progress_value_max_points = int(progress_value_max_points)
            if native_progress_value_type == "integer" and (
                not float(progress_value_min_points).is_integer()
                or not float(progress_value_max_points).is_integer()
            ):
                raise ValueError(
                    f"{batch_results_path}: line {line_number} integer native progress "
                    "requires integer-valued progress bounds."
                )
            if progress_value_max_points <= progress_value_min_points:
                raise ValueError(
                    f"{batch_results_path}: line {line_number} progress_value_max_points must be "
                    "greater than progress_value_min_points."
                )

            # Explicit normalization requires both the rule and scale. When
            # they are absent, evaluation can use min-max normalization; strict
            # callers can require a scale or provide an explicit override.
            has_normalization_rule = row.get("progress_normalization_rule") not in (
                None,
                "",
            )
            has_normalization_scale = row.get(
                "progress_normalization_scale_points"
            ) not in (None, "")
            if normalization_scale_override is not None and not has_normalization_scale:
                # A dataset override supplies the scale when it is absent,
                # whether or not the normalization rule was explicitly stored.
                if has_normalization_rule:
                    artifact_normalization_rule = str(
                        row.get("progress_normalization_rule")
                    ).strip().lower()
                    if (
                        artifact_normalization_rule
                        != PROGRESS_NORMALIZATION_RULE_DIVIDE_BY_SCALE
                    ):
                        raise ValueError(
                            f"{batch_results_path}: line {line_number} "
                            "progress_normalization_rule must be "
                            f"{PROGRESS_NORMALIZATION_RULE_DIVIDE_BY_SCALE!r} when a "
                            "config scale override supplies the missing artifact scale, got "
                            f"{row.get('progress_normalization_rule')!r}."
                        )
                progress_normalization_rule = (
                    PROGRESS_NORMALIZATION_RULE_DIVIDE_BY_SCALE
                )
                progress_normalization_scale_points = normalization_scale_override
                progress_normalization_source = "config_override_missing_artifact"
            elif not has_normalization_rule and not has_normalization_scale:
                if require_progress_normalization_scale:
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} source_traj_id "
                        f"{source_traj_id!r} has no artifact "
                        "progress_normalization_scale_points and no dataset config "
                        "override was provided. Set "
                        "progress_normalization_scale_points under this "
                        "reward_agent_runs dataset."
                    )
                else:
                    progress_normalization_rule = (
                        PROGRESS_NORMALIZATION_RULE_LEGACY_MIN_MAX
                    )
                    progress_normalization_scale_points = None
                    progress_normalization_source = "inferred_legacy_missing_scale"
            elif has_normalization_rule != has_normalization_scale:
                raise ValueError(
                    f"{batch_results_path}: line {line_number} must declare "
                    "progress_normalization_rule and "
                    "progress_normalization_scale_points together."
                )
            else:
                progress_normalization_rule = str(
                    row.get("progress_normalization_rule")
                ).strip().lower()
                if (
                    progress_normalization_rule
                    != PROGRESS_NORMALIZATION_RULE_DIVIDE_BY_SCALE
                ):
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} "
                        "progress_normalization_rule must be "
                        f"{PROGRESS_NORMALIZATION_RULE_DIVIDE_BY_SCALE!r}, got "
                        f"{row.get('progress_normalization_rule')!r}."
                    )
                if isinstance(
                    row.get("progress_normalization_scale_points"), bool
                ):
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} "
                        "progress_normalization_scale_points must be a finite number, "
                        "not a boolean."
                    )
                artifact_normalization_scale_points = _required_row_number(
                    row,
                    path=batch_results_path,
                    line_number=line_number,
                    field_name="progress_normalization_scale_points",
                )
                if artifact_normalization_scale_points <= 0.0:
                    raise ValueError(
                        f"{batch_results_path}: line {line_number} "
                        "progress_normalization_scale_points must be greater than 0."
                    )
                if normalization_scale_override is None:
                    progress_normalization_scale_points = (
                        artifact_normalization_scale_points
                    )
                    progress_normalization_source = "explicit"
                else:
                    progress_normalization_scale_points = normalization_scale_override
                    if np.isclose(
                        artifact_normalization_scale_points,
                        normalization_scale_override,
                        rtol=0.0,
                        atol=0.0,
                    ):
                        progress_normalization_source = "config_override_matches_artifact"
                    else:
                        progress_normalization_source = "config_override_replaces_artifact"
                        normalization_override_mismatches.append(
                            f"{batch_results_path}:line {line_number} "
                            f"source_traj_id={source_traj_id!r} "
                            f"artifact={artifact_normalization_scale_points:g} "
                            f"config={normalization_scale_override:g}"
                        )

            corrected_progress_csv = _resolve_batch_row_path(
                row.get("corrected_progress_csv"),
                batch_results_path=batch_results_path,
                field_name=f"line {line_number} corrected_progress_csv",
            )
            if not corrected_progress_csv.is_file():
                raise FileNotFoundError(
                    f"{batch_results_path}: line {line_number} corrected_progress_csv not found: "
                    f"{corrected_progress_csv}"
                )

            completed_by_source[source_traj_id] = RewardAgentResultEntry(
                batch_results_path=batch_results_path,
                batch_results_line=line_number,
                corrected_progress_csv=corrected_progress_csv,
                source_traj_id=source_traj_id,
                episode_index=episode_index,
                num_transition_rows=num_transition_rows,
                progress_output_mode=progress_output_mode,
                progress_value_type=progress_value_type,
                native_progress_value_type=native_progress_value_type,
                native_progress_value_type_source=(
                    native_progress_value_type_source
                ),
                progress_value_min_points=progress_value_min_points,
                progress_value_max_points=progress_value_max_points,
                progress_normalization_rule=progress_normalization_rule,
                progress_normalization_scale_points=(
                    progress_normalization_scale_points
                ),
                progress_normalization_source=progress_normalization_source,
                raw_predictions_json=(
                    None
                    if row.get("raw_predictions_json") in (None, "")
                    else _resolve_batch_row_path(
                        row["raw_predictions_json"],
                        batch_results_path=batch_results_path,
                        field_name=f"line {line_number} raw_predictions_json",
                    )
                ),
            )
            if native_progress_value_type_source == "inferred_legacy_dense_integer":
                inferred_legacy_dense_integer_native_type_rows += 1
            if progress_normalization_source == "inferred_legacy_missing_scale":
                inferred_legacy_normalization_rows += 1

    if inferred_legacy_normalization_rows:
        warnings.warn(
            f"{label}: {inferred_legacy_normalization_rows} completed result row(s) "
            "omit progress_normalization_rule and "
            "progress_normalization_scale_points; using legacy normalization "
            "(points - min_points) / (max_points - min_points).",
            LegacyProgressNormalizationWarning,
            stacklevel=2,
        )
    if normalization_override_mismatches:
        warnings.warn(
            f"{label}: config progress_normalization_scale_points replaced a different "
            f"artifact scale for {len(normalization_override_mismatches)} completed row(s). "
            "The corrected CSV must match the config-selected scale. Examples: "
            f"{normalization_override_mismatches[:3]}",
            ProgressNormalizationOverrideWarning,
            stacklevel=2,
        )

    coverage_message = _coverage_message(
        episodes_by_source=coverage_episodes,
        covered_sources=set(completed_by_source),
        allow_partial=allow_partial,
        label=label,
        strict_coverage_hint=strict_coverage_hint,
    )
    return RewardAgentResultIndex(
        completed_by_source=completed_by_source,
        loaded_files=len(batch_results),
        total_rows=total_rows,
        completed_rows=len(completed_by_source),
        ignored_non_completed=ignored_non_completed,
        ignored_out_of_scope=ignored_out_of_scope,
        inferred_legacy_dense_integer_native_type_rows=(
            inferred_legacy_dense_integer_native_type_rows
        ),
        inferred_legacy_normalization_rows=inferred_legacy_normalization_rows,
        coverage_message=coverage_message,
    )


def load_reward_agent_progress_curve(
    entry: RewardAgentResultEntry,
    episode: LeRobotEpisodeRecord,
) -> RewardAgentProgressCurve:
    """Read a canonical corrected CSV without discarding filled progress rows.

    ``progress_points`` retains the full LeRobot frame axis for RECAP and other
    downstream consumers.  Model-quality evaluators should instead use the two
    ``model_output_*`` views, which exclude rows introduced by previous-hold or
    another resolver.
    """

    with entry.corrected_progress_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [
            column
            for column in (
                "frame_index",
                "current_progress_points",
                "current_progress_normalized",
                "is_model_output",
                "source",
            )
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(f"{entry.corrected_progress_csv}: missing required columns {missing}.")
        rows = list(reader)

    if len(rows) != int(episode.length):
        raise ValueError(
            f"{entry.corrected_progress_csv}: expected {episode.length} rows for "
            f"{episode.source_traj_id}, got {len(rows)}."
        )

    progress_integer: list[int] = []
    progress_float: list[float] = []
    native_progress_integer: list[int] = []
    native_progress_float: list[float] = []
    normalized_values: list[float] = []
    model_output_mask: list[bool] = []
    source_labels: list[str] = []
    value_min = float(entry.progress_value_min_points)
    value_max = float(entry.progress_value_max_points)
    value_range = value_max - value_min
    normalized_min = entry.progress_normalized_min
    normalized_max = entry.progress_normalized_max
    for idx, row in enumerate(rows):
        frame_index = int(row["frame_index"])
        if frame_index != idx:
            raise ValueError(
                f"{entry.corrected_progress_csv}: frame_index at row {idx} must be {idx}, "
                f"got {frame_index}."
            )

        # Parse provenance before the point value so native rows can be
        # validated against their own type contract even when a linearly
        # filled full curve is floating point.
        is_model_output_raw = str(row["is_model_output"]).strip()
        if is_model_output_raw not in {"0", "1"}:
            raise ValueError(
                f"{entry.corrected_progress_csv}: is_model_output at frame {idx} must be "
                f"0 or 1, got {is_model_output_raw!r}."
            )
        is_native = is_model_output_raw == "1"
        source_label = str(row["source"]).strip()
        if not source_label:
            raise ValueError(
                f"{entry.corrected_progress_csv}: source at frame {idx} must be non-empty."
            )

        current_raw = str(row["current_progress_points"]).strip()
        if entry.progress_value_type == "integer":
            if re.fullmatch(r"[+-]?\d+", current_raw) is None:
                raise ValueError(
                    f"{entry.corrected_progress_csv}: current_progress_points at frame {idx} "
                    f"must be an integer, got {current_raw!r}."
                )
            current: int | float = int(current_raw)
            progress_integer.append(int(current))
        elif entry.progress_value_type == "float":
            current = float(current_raw)
            if not np.isfinite(current):
                raise ValueError(
                    f"{entry.corrected_progress_csv}: current_progress_points at frame {idx} "
                    f"must be finite, got {current_raw!r}."
                )
            progress_float.append(float(current))
        else:  # Defensive guard for manually constructed entries.
            raise ValueError(f"Unknown progress_value_type: {entry.progress_value_type!r}.")

        if is_native:
            if entry.native_progress_value_type == "integer":
                if re.fullmatch(r"[+-]?\d+", current_raw) is None:
                    raise ValueError(
                        f"{entry.corrected_progress_csv}: native current_progress_points "
                        f"at frame {idx} must be an integer, got {current_raw!r}."
                    )
                native_progress_integer.append(int(current_raw))
            elif entry.native_progress_value_type == "float":
                native_progress_float.append(float(current))
            else:  # Defensive guard for manually constructed entries.
                raise ValueError(
                    "Unknown native_progress_value_type: "
                    f"{entry.native_progress_value_type!r}."
                )
        if not value_min <= float(current) <= value_max:
            raise ValueError(
                f"{entry.corrected_progress_csv}: current_progress_points at frame {idx} "
                f"is outside [{entry.progress_value_min_points}, "
                f"{entry.progress_value_max_points}]: {current}."
            )
        normalized = float(row["current_progress_normalized"])
        if (
            not np.isfinite(normalized)
            or normalized < normalized_min
            or normalized > normalized_max
        ):
            raise ValueError(
                f"{entry.corrected_progress_csv}: current_progress_normalized at frame {idx} "
                f"is outside [{normalized_min}, {normalized_max}]: {normalized}."
            )
        if entry.progress_normalization_scale_points is None:
            expected_normalized = (float(current) - value_min) / value_range
        else:
            expected_normalized = float(current) / float(
                entry.progress_normalization_scale_points
            )
        if not np.isclose(normalized, expected_normalized, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{entry.corrected_progress_csv}: current_progress_normalized at frame {idx} "
                "does not match the effective progress normalization contract: "
                f"got {normalized}, expected {expected_normalized}."
            )
        normalized_values.append(normalized)
        model_output_mask.append(is_native)
        source_labels.append(source_label)

    frame_indices = np.arange(int(episode.length), dtype=np.int64)
    progress_points = (
        np.asarray(progress_integer, dtype=np.int64)
        if entry.progress_value_type == "integer"
        else np.asarray(progress_float, dtype=np.float64)
    )
    is_model_output = np.asarray(model_output_mask, dtype=np.bool_)
    native_progress_points = (
        np.asarray(native_progress_integer, dtype=np.int64)
        if entry.native_progress_value_type == "integer"
        else np.asarray(native_progress_float, dtype=np.float64)
    )
    model_output_count = int(np.sum(is_model_output))
    minimum_outputs = min(2, int(episode.length))
    if model_output_count < minimum_outputs:
        raise ValueError(
            f"{entry.corrected_progress_csv}: expected at least {minimum_outputs} model-output "
            f"rows, got {model_output_count}."
        )
    if not bool(is_model_output[0]):
        raise ValueError(f"{entry.corrected_progress_csv}: frame 0 must be a model output.")
    if not bool(is_model_output[-1]):
        raise ValueError(
            f"{entry.corrected_progress_csv}: terminal frame {episode.length - 1} must be a model output."
        )
    if entry.progress_output_mode == "dense" and not bool(np.all(is_model_output)):
        raise ValueError(
            f"{entry.corrected_progress_csv}: progress_output_mode='dense' requires every "
            "row to be marked as a model output."
        )
    if entry.progress_output_mode not in {"dense", "sampled"}:
        # Defensive guard for manually constructed entries.
        raise ValueError(f"Unknown progress_output_mode: {entry.progress_output_mode!r}.")
    return RewardAgentProgressCurve(
        frame_indices=frame_indices,
        progress_points=progress_points,
        progress_normalized=np.asarray(normalized_values, dtype=np.float64),
        is_model_output=is_model_output,
        sources=np.asarray(source_labels, dtype=np.str_),
        native_progress_points=native_progress_points,
    )


def resolve_lerobot_roots(input_root: Path) -> list[Path]:
    """Resolve either one LeRobot leaf or a collection containing leaves."""

    candidate = input_root.expanduser()
    if (candidate / "meta" / "info.json").is_file():
        return [candidate.resolve()]
    matches = sorted(candidate.glob("**/meta/info.json"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any meta/info.json under {candidate.resolve(strict=False)}."
        )
    return [match.parent.parent.resolve() for match in matches]


def _coverage_message(
    *,
    episodes_by_source: Mapping[str, LeRobotEpisodeRecord],
    covered_sources: set[str],
    allow_partial: bool,
    label: str,
    strict_coverage_hint: str | None,
) -> str | None:
    missing = sorted(set(episodes_by_source) - covered_sources)
    if not missing:
        return None
    if allow_partial:
        return f"[warning] {label}: partial coverage enabled; missing {len(missing)} episodes."
    hint = str(strict_coverage_hint or "").strip()
    hint_suffix = f" {hint}" if hint else ""
    raise ValueError(
        f"{label}: prediction indexes cover {len(covered_sources)}/{len(episodes_by_source)} episodes. "
        f"Missing examples: {missing[:5]}.{hint_suffix}"
    )


def _resolve_episode_path(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    require_existing: bool = True,
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunks_size
    data_pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    candidate = dataset_root / str(data_pattern).format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    )
    if candidate.is_file():
        return candidate.resolve()
    if not require_existing:
        return candidate.resolve()
    matches = sorted(dataset_root.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{dataset_root}: expected exactly one parquet for episode {episode_index}, "
            f"found {len(matches)}."
        )
    return matches[0].resolve()


def _episode_length_from_row(row: dict[str, Any], *, default: int | None) -> int | None:
    for key in ("length", "num_frames", "episode_length"):
        if row.get(key) is not None:
            return int(row[key])
    return default


def _resolve_batch_row_path(
    raw_value: Any,
    *,
    batch_results_path: Path,
    field_name: str,
) -> Path:
    if raw_value in (None, ""):
        raise ValueError(f"{batch_results_path}: {field_name} is required.")
    candidate = Path(str(raw_value)).expanduser()
    if not candidate.is_absolute():
        candidate = batch_results_path.parent / candidate
    return candidate.resolve()


def _required_row_int(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    field_name: str,
) -> int:
    if row.get(field_name) in (None, ""):
        raise ValueError(f"{path}: line {line_number} missing {field_name!r}.")
    return int(row[field_name])


def _required_row_number(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    field_name: str,
) -> float:
    if row.get(field_name) in (None, ""):
        raise ValueError(f"{path}: line {line_number} missing {field_name!r}.")
    value = float(row[field_name])
    if not np.isfinite(value):
        raise ValueError(f"{path}: line {line_number} {field_name} must be finite.")
    return value


def _source_subset_for_leaf(dataset_root: Path) -> str:
    if dataset_root.parent.name.startswith("rank_"):
        return f"{dataset_root.parent.name}/{dataset_root.name}"
    return dataset_root.name


def _slugify_text(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_") or "unknown"


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
