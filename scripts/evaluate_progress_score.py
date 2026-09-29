#!/usr/bin/env python3
"""Evaluate the controlled wrong-object progress metrics.

The input is a YAML job file that groups each LeRobot root with one or more
reward-agent batch_results.jsonl files.  Each dataset group must be covered
exactly once by completed reward-agent rows.  Primary metrics use each model's
native output points. Optional sensitivity analysis reports the native failure
upper tail and evaluates configurable, endpoint-preserving original frame indices
from those same native outputs; filled rows are never evaluated.
FPSR measures duration without increases above epsilon
and is reported alongside success-positive coverage over the same tolerance
grid. TOPReward can be evaluated
directly in True-token probability space using its raw prediction artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.result_index import (  # noqa: E402
    LeRobotEpisodeRecord,
    RewardAgentResultEntry,
    load_completed_reward_agent_entries,
    load_lerobot_episode_index,
    load_reward_agent_progress_curve,
)
from agent.progress import build_progress_sample_indices  # noqa: E402


DEFAULT_FAILURE_TOP_FRAC = 0.05
FAILURE_TOP_MIN_COUNT = 1
DEFAULT_SENSITIVITY_MAX_FRAMES = 8
DEFAULT_MAX_DIAGNOSTIC_EPISODES = 10
DEFAULT_MAX_INTERVALS_PER_EPISODE = 5
DEFAULT_FPSR_EPSILONS = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.02)
PROGRESS_SOURCES = ("artifact_normalized", "topreward_probability")
# This numerical guard only decides whether normalized VOC is effectively
# perfect. Adjacent progress changes use exact zero; FPSR sweep tolerances
# have no effect on success diagnostics.
DEFAULT_VOC_DEGRADED_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """One LeRobot root and all result shards evaluated in this invocation."""

    lerobot_root: Path
    batch_results: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class JobConfig:
    """Resolved evaluator configuration."""

    config_path: Path
    datasets: tuple[DatasetConfig, ...]
    fpsr_epsilons: tuple[float, ...]
    progress_source: str
    sensitivity_enabled: bool
    sensitivity_max_frames: int
    failure_top_frac: float


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    """Computed scalar summaries for one evaluated episode."""

    source_traj_id: str
    is_success: bool
    voc: float | None
    normalized_voc: float | None
    failure_top_progress: float | None
    fpsr_by_epsilon: tuple[float, ...] | None
    success_increase_by_epsilon: tuple[float, ...] | None
    progress_final: float
    progress_min: float
    progress_max: float
    progress_mean: float
    metric_frame_count: int
    failure_top_count: int | None
    failure_top_effective_frac: float | None
    is_constant_success_progress: bool
    voc_diagnostic: dict[str, Any] | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate native-output VOC, failure suppression AUC, failure "
            "progress-increase duration ratio, FPSR tolerance curves, and progress gaps "
            "from LeRobot datasets plus reward-agent batch_results.jsonl files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "YAML config with datasets[].lerobot_root and datasets[].batch_results."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Results are always printed to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    job_cfg = load_job_config(args.config)
    result = evaluate_job(job_cfg)
    # The evaluator builds the result in reader-facing semantic order: primary
    # metrics first, followed by appendix results, diagnostics, and provenance.
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


def load_job_config(config_path: Path) -> JobConfig:
    """Load a YAML job config and resolve paths relative to that file."""

    resolved_config = config_path.expanduser().resolve()
    with resolved_config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{resolved_config}: expected a YAML mapping.")
    allowed_fields = {
        "progress_source", "fpsr_epsilons", "sensitivity", "datasets"
    }
    unknown_fields = sorted(str(key) for key in payload if key not in allowed_fields)
    if unknown_fields:
        raise ValueError(f"{resolved_config}: unknown config fields: {unknown_fields}.")

    progress_source = payload.get("progress_source", "artifact_normalized")
    if progress_source not in PROGRESS_SOURCES:
        raise ValueError(
            f"{resolved_config}: progress_source must be one of {PROGRESS_SOURCES}."
        )
    # The grid is the only FPSR threshold setting. Its exact-zero first point
    # also defines the primary metric, including for tiny raw probabilities.
    raw_epsilons = payload.get("fpsr_epsilons", list(DEFAULT_FPSR_EPSILONS))
    if not isinstance(raw_epsilons, list) or not raw_epsilons:
        raise ValueError(f"{resolved_config}: fpsr_epsilons must be a non-empty list.")
    fpsr_epsilons = tuple(
        _require_config_number(
            value, config_path=resolved_config, field_name=f"fpsr_epsilons[{index}]"
        )
        for index, value in enumerate(raw_epsilons)
    )
    if fpsr_epsilons[0] != 0.0 or any(
        right <= left for left, right in zip(fpsr_epsilons, fpsr_epsilons[1:])
    ):
        raise ValueError(
            f"{resolved_config}: fpsr_epsilons must start at zero and strictly increase."
        )

    raw_sensitivity = payload.get("sensitivity", {})
    if raw_sensitivity is None:
        raw_sensitivity = {}
    if not isinstance(raw_sensitivity, dict):
        raise TypeError(f"{resolved_config}: sensitivity must be a mapping.")
    sensitivity_enabled = _require_config_bool(
        raw_sensitivity.get("enabled", True),
        config_path=resolved_config,
        field_name="sensitivity.enabled",
    )
    sensitivity_max_frames = _require_config_int(
        raw_sensitivity.get("max_frames", DEFAULT_SENSITIVITY_MAX_FRAMES),
        config_path=resolved_config,
        field_name="sensitivity.max_frames",
    )
    if sensitivity_max_frames < 2:
        raise ValueError(
            f"{resolved_config}: sensitivity.max_frames must be at least 2, "
            f"got {sensitivity_max_frames}."
        )

    # The upper-tail fraction belongs to optional sensitivity analysis; primary
    # failure metrics always use the native maximum, independent of this value.
    failure_top_frac = _require_config_number(
        raw_sensitivity.get("failure_top_frac", DEFAULT_FAILURE_TOP_FRAC),
        config_path=resolved_config,
        field_name="sensitivity.failure_top_frac",
    )
    if not 0.0 < failure_top_frac <= 1.0:
        raise ValueError(
            f"{resolved_config}: sensitivity.failure_top_frac must be in (0, 1], "
            f"got {failure_top_frac}."
        )

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(f"{resolved_config}: datasets must be a non-empty list.")

    base_dir = resolved_config.parent
    datasets: list[DatasetConfig] = []
    for index, raw_dataset in enumerate(raw_datasets):
        if not isinstance(raw_dataset, dict):
            raise TypeError(f"{resolved_config}: datasets[{index}] must be a mapping.")
        lerobot_root = _resolve_config_path(
            raw_dataset.get("lerobot_root"),
            base_dir=base_dir,
            field_name=f"datasets[{index}].lerobot_root",
        )
        raw_batch_results = raw_dataset.get("batch_results")
        if not isinstance(raw_batch_results, list) or not raw_batch_results:
            raise ValueError(
                f"{resolved_config}: datasets[{index}].batch_results must be a non-empty list."
            )
        batch_results = tuple(
            _resolve_config_path(
                raw_path,
                base_dir=base_dir,
                field_name=f"datasets[{index}].batch_results[{path_index}]",
            )
            for path_index, raw_path in enumerate(raw_batch_results)
        )
        datasets.append(
            DatasetConfig(
                lerobot_root=lerobot_root,
                batch_results=batch_results,
            )
        )

    return JobConfig(
        config_path=resolved_config,
        datasets=tuple(datasets),
        failure_top_frac=failure_top_frac,
        fpsr_epsilons=fpsr_epsilons,
        progress_source=progress_source,
        sensitivity_enabled=sensitivity_enabled,
        sensitivity_max_frames=sensitivity_max_frames,
    )


def evaluate_job(job_cfg: JobConfig) -> dict[str, Any]:
    """Evaluate all configured datasets and return a JSON-serializable summary."""

    all_scores: list[EpisodeScore] = []
    sensitivity_scores: list[EpisodeScore] = []
    dataset_summaries: list[dict[str, Any]] = []
    total_ignored_non_completed = 0
    value_specs: set[tuple[str, str, float, float]] = set()
    effective_normalization_specs: set[tuple[float, float, float, float]] = set()
    artifact_normalization_contracts: set[
        tuple[str, float | None, str]
    ] = set()
    total_inferred_legacy_normalization_rows = 0
    probability_underflow_count = 0
    output_modes: set[str] = set()
    for dataset_cfg in job_cfg.datasets:
        episodes = load_lerobot_episode_index(dataset_cfg.lerobot_root, read_is_success=True)
        result_index = load_completed_reward_agent_entries(
            batch_results=dataset_cfg.batch_results,
            episodes_by_source=episodes,
            lerobot_root=dataset_cfg.lerobot_root,
            allow_partial=False,
            label=f"progress_score:{dataset_cfg.lerobot_root}",
        )
        entries = result_index.completed_by_source
        total_ignored_non_completed += result_index.ignored_non_completed
        total_inferred_legacy_normalization_rows += (
            result_index.inferred_legacy_normalization_rows
        )

        scores: list[EpisodeScore] = []
        dataset_sensitivity_scores: list[EpisodeScore] = []
        for source_traj_id in sorted(episodes):
            episode = episodes[source_traj_id]
            entry = entries[source_traj_id]
            output_modes.add(entry.progress_output_mode)
            value_specs.add(
                (
                    entry.progress_value_type,
                    entry.native_progress_value_type,
                    float(entry.progress_value_min_points),
                    float(entry.progress_value_max_points),
                )
            )
            if entry.progress_normalization_scale_points is None:
                normalization_offset = float(entry.progress_value_min_points)
                normalization_divisor = float(
                    entry.progress_value_max_points
                ) - float(entry.progress_value_min_points)
            else:
                normalization_offset = 0.0
                normalization_divisor = float(
                    entry.progress_normalization_scale_points
                )
            effective_normalization_specs.add(
                (
                    normalization_offset,
                    normalization_divisor,
                    entry.progress_normalized_min,
                    entry.progress_normalized_max,
                )
            )
            artifact_normalization_contracts.add(
                (
                    entry.progress_normalization_rule,
                    (
                        None
                        if entry.progress_normalization_scale_points is None
                        else float(entry.progress_normalization_scale_points)
                    ),
                    entry.progress_normalization_source,
                )
            )
            curve = load_reward_agent_progress_curve(entry, episode)
            # Select one value space before computing any metric. Probability-
            # space FPSR and CMPG must use the same native values.
            progress_values = curve.model_output_progress_normalized
            # Preserve integer-native values through both evaluation views.
            # Subtracting already-normalized floats can misclassify equality.
            integer_scale = native_integer_progress_scale(
                entry, progress_source=job_cfg.progress_source
            )
            integer_points = (
                curve.model_output_progress_points if integer_scale is not None else None
            )
            if job_cfg.progress_source == "topreward_probability":
                progress_values, underflow_count = load_topreward_probabilities(
                    entry, frame_indices=curve.model_output_frame_indices
                )
                probability_underflow_count += underflow_count
            scores.append(
                score_episode_from_values(
                    episode=episode,
                    frame_indices=curve.model_output_frame_indices,
                    progress_normalized=progress_values,
                    integer_progress_points=integer_points,
                    integer_progress_scale=integer_scale,
                    failure_top_frac=(
                        job_cfg.failure_top_frac if job_cfg.sensitivity_enabled else None
                    ),
                    fpsr_epsilons=job_cfg.fpsr_epsilons,
                )
            )

            # Sensitivity evaluates a canonical subset of this raw result set.
            # Every requested index must be a model output; extra native points
            # are allowed, while interpolation/fill rows never enter metrics.
            if job_cfg.sensitivity_enabled:
                expected_indices = build_fixed_frame_indices(
                    num_frames=int(episode.length),
                    max_frames=job_cfg.sensitivity_max_frames,
                )
                sensitivity_indices, sensitivity_progress = select_fixed_frame_native_outputs(
                    source_traj_id=source_traj_id,
                    expected_indices=expected_indices,
                    actual_indices=curve.model_output_frame_indices,
                    actual_progress_values=progress_values,
                )
                # These indices were validated as native above. Slice the exact
                # integer values at the same positions, never the filled curve.
                sensitivity_integer_points = None
                if integer_points is not None:
                    native_positions = np.searchsorted(
                        curve.model_output_frame_indices, sensitivity_indices
                    )
                    sensitivity_integer_points = integer_points[native_positions]
                dataset_sensitivity_scores.append(
                    score_episode_from_values(
                        episode=episode,
                        frame_indices=sensitivity_indices,
                        progress_normalized=sensitivity_progress,
                        integer_progress_points=sensitivity_integer_points,
                        integer_progress_scale=integer_scale,
                        # Upper-tail summaries use the full native output set.
                        failure_top_frac=None,
                        fpsr_epsilons=job_cfg.fpsr_epsilons,
                    )
                )

        all_scores.extend(scores)
        sensitivity_scores.extend(dataset_sensitivity_scores)
        dataset_summaries.append(
            {
                "lerobot_root": str(dataset_cfg.lerobot_root),
                "batch_results": [str(path) for path in dataset_cfg.batch_results],
                "num_episodes": len(scores),
                "num_success": sum(1 for score in scores if score.is_success),
                "num_failure": sum(1 for score in scores if not score.is_success),
                "num_fixed_frame_episodes": len(dataset_sensitivity_scores),
                "ignored_non_completed": result_index.ignored_non_completed,
                "inferred_legacy_normalization_rows": (
                    result_index.inferred_legacy_normalization_rows
                ),
            }
        )

    if len(output_modes) != 1:
        raise ValueError(
            "All completed entries in one evaluation job must share one progress output "
            f"mode, got {sorted(output_modes)}."
        )
    if len(value_specs) != 1:
        raise ValueError(
            "All completed entries in one evaluation job must share one progress value "
            f"type/range, got {sorted(value_specs)}."
        )
    if len(effective_normalization_specs) != 1:
        raise ValueError(
            "All completed entries in one evaluation job must use an equivalent "
            "progress normalization mapping, got "
            f"{sorted(effective_normalization_specs)}."
        )
    native_summary = summarize_scores(
        all_scores,
        include_top_fraction=job_cfg.sensitivity_enabled,
        fpsr_epsilons=job_cfg.fpsr_epsilons,
    )
    output_mode = next(iter(output_modes))
    curve_value_type, native_value_type, value_min, value_max = next(
        iter(value_specs)
    )
    (
        normalization_offset,
        normalization_divisor,
        normalized_min,
        normalized_max,
    ) = next(iter(effective_normalization_specs))

    # Keep paper-facing metrics at the front of the document. The
    # appendix always retains the components of the primary progress gap.
    # Native upper-tail quantities are added only when sensitivity is enabled.
    result: dict[str, Any] = {
        "primary_metrics": native_summary["primary_metrics"],
        "appendix_metrics": dict(native_summary["metric_components"]),
        "fpsr_curve": native_summary["fpsr_curve"],
    }

    if job_cfg.sensitivity_enabled:
        result["appendix_metrics"].update(native_summary["appendix_metrics"])

    fixed_frame_summary: dict[str, Any] | None = None
    if sensitivity_scores:
        fixed_frame_summary = summarize_scores(
            sensitivity_scores,
            include_top_fraction=False,
            fpsr_epsilons=job_cfg.fpsr_epsilons,
        )
        result["fixed_frame_analysis"] = {
            "metrics": fixed_frame_summary["primary_metrics"],
            "metric_components": fixed_frame_summary["metric_components"],
            "fpsr_curve": fixed_frame_summary["fpsr_curve"],
            "metadata": {
                "max_frames": job_cfg.sensitivity_max_frames,
                "sample_rule": "linspace_include_start_end",
                "source": "current_batch_native_model_outputs",
                "includes_first_frame": True,
                "includes_terminal_frame": True,
                "num_episodes": len(sensitivity_scores),
            },
        }

    # Detailed curve-quality information is intentionally separate from the
    # scores a reader would copy into a result table.
    diagnostics: dict[str, Any] = {
        "native_output": native_summary["diagnostics"],
    }
    if fixed_frame_summary is not None:
        diagnostics["fixed_frame"] = fixed_frame_summary["diagnostics"]
    result["diagnostics"] = diagnostics

    # Counts, value contracts, and source paths describe how the evaluation was
    # assembled; they are useful provenance but are not model-quality metrics.
    result["evaluation_context"] = {
        "num_datasets": len(job_cfg.datasets),
        "num_episodes": len(all_scores),
        "num_success": sum(1 for score in all_scores if score.is_success),
        "num_failure": sum(1 for score in all_scores if not score.is_success),
        "fpsr_primary_epsilon": job_cfg.fpsr_epsilons[0],
        "success_diagnostic_epsilon": 0.0,
        "fpsr_epsilons": list(job_cfg.fpsr_epsilons),
        "success_positive_time_retention": {
            "baseline_epsilon": 0.0,
            "aggregation": "ratio_of_group_mean_positive_time_coverage",
            "zero_baseline": "undefined_null",
        },
        "evaluation_progress": {
            "source": job_cfg.progress_source,
            "units": (
                "True-token probability"
                if job_cfg.progress_source == "topreward_probability"
                else "artifact-normalized progress"
            ),
            "min": (
                0.0 if job_cfg.progress_source == "topreward_probability" else normalized_min
            ),
            "max": (
                1.0 if job_cfg.progress_source == "topreward_probability" else normalized_max
            ),
        },
        "failure_progress_increase_time_ratio": {
            "definition_version": 2,
            "timeline": "original_frame_indices",
            "measure": "interval_duration",
            "interval_weight": "end_frame_index - start_frame_index",
            "denominator": "episode_length - 1",
            "increase_rule": "delta_progress > epsilon",
            "threshold_comparison": (
                "integer_deltas_exact_decimal_thresholds"
                if (
                    native_value_type == "integer"
                    and job_cfg.progress_source == "artifact_normalized"
                )
                else "normalized_float64_differences"
            ),
            "trajectory_aggregation": "unweighted_mean",
        },
        "ignored_non_completed": total_ignored_non_completed,
        "inferred_legacy_normalization_rows": (
            total_inferred_legacy_normalization_rows
        ),
        "progress_output_mode": output_mode,
        "progress_value_spec": {
            "full_curve_type": curve_value_type,
            "native_type": native_value_type,
            "min_points": value_min,
            "max_points": value_max,
        },
        "progress_normalization": {
            "applied_to_metrics": job_cfg.progress_source == "artifact_normalized",
            "effective_offset_points": normalization_offset,
            "effective_divisor_points": normalization_divisor,
            "normalized_min": normalized_min,
            "normalized_max": normalized_max,
            "artifact_contracts": [
                {
                    "rule": rule,
                    "scale_points": scale,
                    "source": source,
                }
                for rule, scale, source in sorted(
                    artifact_normalization_contracts,
                    key=lambda item: (item[0], str(item[1]), item[2]),
                )
            ],
        },
        "datasets": dataset_summaries,
    }
    if job_cfg.sensitivity_enabled:
        result["evaluation_context"].update(
            {
                "failure_top_frac": job_cfg.failure_top_frac,
                "failure_top_min_count": FAILURE_TOP_MIN_COUNT,
            }
        )
    if job_cfg.progress_source == "topreward_probability":
        result["diagnostics"]["topreward_probability"] = {
            "native_underflow_count": probability_underflow_count,
            "value_field": "probability_true",
            "validated_against": "exp(raw_logprob_true)",
        }
    return result


def summarize_scores(
    scores: Sequence[EpisodeScore],
    *,
    include_top_fraction: bool,
    fpsr_epsilons: Sequence[float],
) -> dict[str, Any]:
    """Aggregate metrics, their components, and operational diagnostics."""

    success_scores = [score for score in scores if score.is_success]
    failure_scores = [score for score in scores if not score.is_success]
    if not success_scores:
        raise ValueError("Cannot compute VOC without at least one successful episode.")
    if not failure_scores:
        raise ValueError("Cannot compute failure metrics without at least one failure episode.")

    # A constant success curve has undefined Spearman correlation and provides
    # no evidence of progress increasing over time.  The paper-facing VOC
    # aggregate therefore assigns that trajectory zero credit without
    # pretending its raw Spearman value is zero.  Diagnostics retain the raw
    # undefined value, the defined-only mean, and the number of constant curves.
    aggregate_success_vocs = np.asarray(
        [0.0 if score.voc is None else float(score.voc) for score in success_scores],
        dtype=np.float64,
    )
    voc = float(np.mean(aggregate_success_vocs))
    normalized_success_vocs = np.asarray(
        [float(score.normalized_voc) for score in success_scores],
        dtype=np.float64,
    )
    n_voc = float(np.mean(normalized_success_vocs))
    defined_success_vocs = np.asarray(
        [float(score.voc) for score in success_scores if score.voc is not None],
        dtype=np.float64,
    )
    defined_only_voc_mean = (
        float(np.mean(defined_success_vocs)) if len(defined_success_vocs) else None
    )
    success_final_mean = float(np.mean([score.progress_final for score in success_scores]))
    failure_max_mean = float(np.mean([score.progress_max for score in failure_scores]))
    # Average trajectories equally, then reuse the zero-threshold curve point
    # as the primary FPSR. There is no separately computed primary tolerance.
    if len(fpsr_epsilons) == 0 or fpsr_epsilons[0] != 0.0:
        raise ValueError("The FPSR grid must start at zero for the primary metric.")
    if any(
        score.fpsr_by_epsilon is None
        or len(score.fpsr_by_epsilon) != len(fpsr_epsilons)
        for score in failure_scores
    ):
        raise ValueError("Failure scores are missing the configured FPSR curve.")
    fpsr_curve_mean = np.mean(
        np.asarray([score.fpsr_by_epsilon for score in failure_scores]), axis=0
    )
    # Pair suppression with the success group's surviving positive duration.
    # Successful episodes can contain pauses/regressions, so this is coverage,
    # not a ground-truth TPR. Keep flat episodes in the group mean.
    if any(
        score.success_increase_by_epsilon is None
        or len(score.success_increase_by_epsilon) != len(fpsr_epsilons)
        for score in success_scores
    ):
        raise ValueError("Success scores are missing the configured positive-time curve.")
    success_increase_mean = np.mean(
        np.asarray([score.success_increase_by_epsilon for score in success_scores]), axis=0
    )
    success_zero_coverage = float(success_increase_mean[0])
    metric_counts = [int(score.metric_frame_count) for score in scores]
    result: dict[str, Any] = {
        "primary_metrics": {
            "voc": voc,
            "fpsr": float(fpsr_curve_mean[0]),
            "cmpg": float(success_final_mean - failure_max_mean),
            "success_progress_increase_time_ratio": success_zero_coverage,
            "failure_suppression_auc": float(1.0 - failure_max_mean),
        },
        "fpsr_curve": [
            {
                "epsilon": float(epsilon),
                "fpsr": float(fpsr),
                "failure_progress_increase_time_ratio": float(1.0 - fpsr),
                "success_progress_increase_time_ratio": float(success_coverage),
                # A model with no positive success intervals has no retention
                # baseline; it must not receive a fabricated perfect ratio.
                "success_positive_time_retention": (
                    float(success_coverage / success_zero_coverage)
                    if success_zero_coverage > 0.0 else None
                ),
            }
            for epsilon, fpsr, success_coverage in zip(
                fpsr_epsilons, fpsr_curve_mean, success_increase_mean, strict=True
            )
        ],
        "metric_components": {
            "success_final_progress_mean": success_final_mean,
            "failure_max_progress_mean": failure_max_mean,
        },
        "diagnostics": {
            "nVOC": n_voc,
            "defined_only_voc_mean": defined_only_voc_mean,
            "num_defined_success_voc": int(len(defined_success_vocs)),
            "num_constant_success_voc": sum(
                1 for score in success_scores if score.is_constant_success_progress
            ),
            "progress_points_per_trajectory": {
                "min": int(min(metric_counts)),
                "max": int(max(metric_counts)),
                "mean": float(np.mean(metric_counts)),
            },
            "success_voc_diagnostics": summarize_success_voc_diagnostics(
                success_scores,
            ),
        },
    }
    if include_top_fraction:
        failure_top_scores = [
            float(score.failure_top_progress)
            for score in failure_scores
            if score.failure_top_progress is not None
        ]
        failure_top_counts = [
            int(score.failure_top_count)
            for score in failure_scores
            if score.failure_top_count is not None
        ]
        failure_effective_fracs = [
            float(score.failure_top_effective_frac)
            for score in failure_scores
            if score.failure_top_effective_frac is not None
        ]
        if len(failure_top_scores) != len(failure_scores):
            raise ValueError("Native failure scores are missing top-fraction summaries.")
        failure_top_mean = float(np.mean(failure_top_scores))
        result["appendix_metrics"] = {
            "success_final_progress_mean": success_final_mean,
            "failure_max_progress_mean": failure_max_mean,
            "failure_top_fraction_progress_mean": failure_top_mean,
            "success_failure_progress_gap_top_fraction": float(
                success_final_mean - failure_top_mean
            ),
        }
        result["diagnostics"].update(
            {
                "failure_top_fraction_count_min": int(min(failure_top_counts)),
                "failure_top_fraction_count_max": int(max(failure_top_counts)),
                "failure_top_fraction_effective_fraction_mean": float(
                    np.mean(failure_effective_fracs)
                ),
            }
        )
    return result


def load_topreward_probabilities(
    entry: RewardAgentResultEntry,
    *,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Validate raw TOPReward probabilities against native frame positions.

    The CSV is the clipped/shifted log-probability view and cannot in general
    recover the raw probability. Its native frame positions still define the
    evaluation sampling protocol. Missing raw data is an error, not a request
    to switch silently back to the linear view.
    """

    path = entry.raw_predictions_json
    if path is None:
        raise ValueError(
            f"{entry.source_traj_id}: topreward_probability requires raw_predictions_json."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("backend") != "topreward":
        raise ValueError(f"{path}: expected a TOPReward raw artifact (backend=topreward).")

    # Exact alignment excludes fill rows, reordered points, and truncated raw
    # artifacts; checking length alone would miss all three failure modes.
    raw_indices = payload.get("sampled_frame_indices")
    if (
        not isinstance(raw_indices, list)
        or any(type(value) is not int for value in raw_indices)
        or raw_indices != np.asarray(frame_indices).tolist()
    ):
        raise ValueError(f"{path}: sampled_frame_indices must match native CSV frame indices.")

    def numeric_array(field_name: str) -> np.ndarray:
        raw_values = payload.get(field_name)
        if (
            not isinstance(raw_values, list)
            or len(raw_values) != len(raw_indices)
            or any(type(value) not in (int, float) for value in raw_values)
        ):
            raise ValueError(f"{path}: {field_name} must contain one number per native frame.")
        values = np.asarray(raw_values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {field_name} must be finite.")
        return values

    probabilities = numeric_array("probability_true")
    logprobs = numeric_array("raw_logprob_true")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{path}: probability_true must lie in [0, 1].")
    if np.any(logprobs > 0.0):
        raise ValueError(f"{path}: raw_logprob_true must be <= 0.")
    # An absolute tolerance would wrongly accept zero in place of a small but
    # representable probability. Use the backend's float64 exp convention.
    derived_probabilities = np.asarray([math.exp(value) for value in logprobs])
    if not np.allclose(probabilities, derived_probabilities, rtol=1e-12, atol=0.0):
        raise ValueError(f"{path}: probability_true must equal exp(raw_logprob_true).")
    underflows = derived_probabilities == 0.0
    if "probability_underflow" in payload:
        flags = payload["probability_underflow"]
        if (
            not isinstance(flags, list)
            or any(type(value) is not bool for value in flags)
            or flags != underflows.tolist()
        ):
            raise ValueError(f"{path}: probability_underflow does not match float64 exp.")
    return probabilities, int(np.count_nonzero(underflows))


def native_integer_progress_scale(
    entry: RewardAgentResultEntry,
    *,
    progress_source: str,
) -> Fraction | None:
    """Return the exact affine divisor only for integer-native artifact values.

    Offsets cancel in adjacent differences. Fraction reads the decimal spelling
    of scale metadata so multiplying it by a decimal epsilon introduces no
    binary rounding. Probability-space and float-native values retain their
    original comparison path, even if their CSV values happen to be integral.
    """

    if (
        progress_source != "artifact_normalized"
        or entry.native_progress_value_type != "integer"
    ):
        return None
    if entry.progress_normalization_scale_points is not None:
        return Fraction(str(entry.progress_normalization_scale_points))
    return (
        Fraction(str(entry.progress_value_max_points))
        - Fraction(str(entry.progress_value_min_points))
    )


def score_episode_from_values(
    *,
    episode: LeRobotEpisodeRecord,
    frame_indices: np.ndarray,
    progress_normalized: np.ndarray,
    failure_top_frac: float | None = None,
    fpsr_epsilons: Sequence[float] = DEFAULT_FPSR_EPSILONS,
    integer_progress_points: np.ndarray | None = None,
    integer_progress_scale: Fraction | None = None,
) -> EpisodeScore:
    """Score a progress view; a None fraction skips optional upper-tail work."""

    if episode.is_success is None:
        raise ValueError(f"{episode.source_traj_id}: is_success was not loaded.")
    metric_frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    progress = np.asarray(progress_normalized, dtype=np.float64).reshape(-1)
    if len(metric_frames) != len(progress) or len(progress) == 0:
        raise ValueError(
            f"{episode.source_traj_id}: metric frame/value lengths must match and be non-empty."
        )
    if not np.all(np.isfinite(progress)):
        raise ValueError(
            f"{episode.source_traj_id}: normalized progress contains non-finite values."
        )
    final_progress = float(progress[-1])
    increase_ratios = progress_increase_time_ratios(
        progress,
        frame_indices=metric_frames,
        episode_length=int(episode.length),
        fpsr_epsilons=fpsr_epsilons,
        integer_progress_points=integer_progress_points,
        integer_progress_scale=integer_progress_scale,
    )
    if episode.is_success:
        voc, is_constant = spearman_voc(progress, frame_indices=metric_frames)
        normalized_voc = (
            0.0
            if is_constant
            else float(np.clip((float(voc) + 1.0) / 2.0, 0.0, 1.0))
        )
        failure_top_progress = None
        fpsr_by_epsilon = None
        success_increase_by_epsilon = tuple(float(value) for value in increase_ratios)
        failure_top_count = None
        failure_top_effective_frac = None
        voc_diagnostic = build_success_voc_diagnostic(
            source_traj_id=episode.source_traj_id,
            voc=voc,
            normalized_voc=normalized_voc,
            frame_indices=metric_frames,
            progress=progress,
            is_constant=is_constant,
        )
    else:
        voc = None
        normalized_voc = None
        is_constant = False
        # Skip computation as well as reporting when sensitivity is disabled or
        # this is the fixed-frame view, whose metrics do not use the upper tail.
        failure_top_progress = None
        failure_top_count = None
        failure_top_effective_frac = None
        if failure_top_frac is not None:
            failure_top_progress, failure_top_count = top_fraction_mean(
                progress,
                fraction=failure_top_frac,
                min_count=FAILURE_TOP_MIN_COUNT,
            )
            failure_top_effective_frac = float(failure_top_count / len(progress))
        fpsr_by_epsilon = tuple(float(1.0 - ratio) for ratio in increase_ratios)
        success_increase_by_epsilon = None
        voc_diagnostic = None
    return EpisodeScore(
        source_traj_id=episode.source_traj_id,
        is_success=episode.is_success,
        voc=voc,
        normalized_voc=normalized_voc,
        failure_top_progress=(
            None if failure_top_progress is None else float(failure_top_progress)
        ),
        fpsr_by_epsilon=fpsr_by_epsilon,
        success_increase_by_epsilon=success_increase_by_epsilon,
        progress_final=final_progress,
        progress_min=float(np.min(progress)),
        progress_max=float(np.max(progress)),
        progress_mean=float(np.mean(progress)),
        metric_frame_count=int(len(progress)),
        failure_top_count=failure_top_count,
        failure_top_effective_frac=failure_top_effective_frac,
        is_constant_success_progress=is_constant,
        voc_diagnostic=voc_diagnostic,
    )


def failure_progress_increase_time_ratio(
    progress: np.ndarray,
    *,
    frame_indices: np.ndarray,
    episode_length: int,
    integer_progress_points: np.ndarray | None = None,
    integer_progress_scale: Fraction | None = None,
) -> float:
    """Return the strictly positive fraction of original-frame duration."""

    ratios = progress_increase_time_ratios(
        progress,
        frame_indices=frame_indices,
        episode_length=episode_length,
        fpsr_epsilons=(0.0,),
        integer_progress_points=integer_progress_points,
        integer_progress_scale=integer_progress_scale,
    )
    return float(ratios[0])


def progress_increase_time_ratios(
    progress: np.ndarray,
    *,
    frame_indices: np.ndarray,
    episode_length: int,
    fpsr_epsilons: Sequence[float],
    integer_progress_points: np.ndarray | None = None,
    integer_progress_scale: Fraction | None = None,
) -> np.ndarray:
    """Measure growing interval duration at each tolerance on the same curve.

    Adjacent native predictions define disjoint time spans with lengths equal
    to their frame-index differences. Their shared endpoints have zero duration.
    Under the fixed-frame-rate protocol, total duration is episode_length - 1
    frame steps; this gives sparse and dense outputs the same time axis. A
    positive endpoint difference marks the whole span, not its unobserved path.
    """

    values = np.asarray(progress, dtype=np.float64).reshape(-1)
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    epsilons = np.asarray(fpsr_epsilons, dtype=np.float64).reshape(-1)
    if len(values) != len(frames) or len(values) == 0:
        raise ValueError("frame_indices must match progress and both must be non-empty.")
    if int(episode_length) < 2:
        raise ValueError("Interval-duration metrics require at least two frames.")
    if len(epsilons) == 0 or np.any(~np.isfinite(epsilons)) or np.any(epsilons < 0.0):
        raise ValueError("fpsr_epsilons thresholds must be finite and non-negative.")
    if np.any(~np.isfinite(values)):
        raise ValueError("progress values must be finite.")
    durations = np.diff(frames)
    if np.any(durations <= 0):
        raise ValueError("frame_indices must be strictly increasing.")
    if int(frames[0]) != 0 or int(frames[-1]) != int(episode_length) - 1:
        raise ValueError(
            "frame_indices must include original trajectory endpoints 0 and "
            f"{int(episode_length) - 1}, got {frames.tolist()}."
        )

    # Require both integer points and their scale to preserve exact comparisons
    # at epsilon boundaries; a partial pair must not trigger float subtraction.
    if (integer_progress_points is None) != (integer_progress_scale is None):
        raise ValueError("Integer progress points and scale must be supplied together.")
    if integer_progress_points is not None:
        points = np.asarray(integer_progress_points)
        if points.shape != values.shape or not np.issubdtype(points.dtype, np.integer):
            raise ValueError(
                "Integer progress points must be a native integer vector matching progress."
            )
        if not isinstance(integer_progress_scale, Fraction) or integer_progress_scale <= 0:
            raise ValueError("Integer progress scale must be a positive Fraction.")
        # For integer d, d > epsilon * scale iff d > floor(epsilon * scale).
        # Exact decimal products also avoid bugs such as 0.29 * 100 rounding
        # below 29. Python integers keep signed differences free of overflow.
        cutoffs = [
            math.floor(Fraction(str(epsilon)) * integer_progress_scale)
            for epsilon in epsilons
        ]
        deltas = [int(right) - int(left) for left, right in zip(points, points[1:])]
        increasing = np.asarray(
            [[delta > cutoff for cutoff in cutoffs] for delta in deltas], dtype=np.bool_
        )
    else:
        # Float-native predictions, including raw TOPReward probabilities, keep
        # strict float64 comparisons. Zero adds no hidden numerical tolerance.
        increasing = np.diff(values)[:, None] > epsilons[None, :]
    return np.sum(durations[:, None] * increasing, axis=0) / float(episode_length - 1)


def spearman_voc(
    progress: np.ndarray,
    *,
    frame_indices: np.ndarray | None = None,
) -> tuple[float | None, bool]:
    """Return rank-correlation VOC against frame index.

    This is the rank-order version of VOC: a perfect score requires progress
    values to induce the same ordering as chronological frame indices.  Ties are
    assigned average ranks, matching the standard Spearman convention, so a
    plateau can reduce the score.  A constant success progress curve has
    undefined rank correlation; the evaluator returns ``None`` here so the
    aggregation layer can apply the explicit zero-credit convention while
    diagnostics preserve the distinction between undefined and zero.
    """

    if len(progress) < 2 or float(np.max(progress)) == float(np.min(progress)):
        return None, True
    progress_ranks = average_ranks(progress.astype(np.float64))
    frames = (
        np.arange(len(progress), dtype=np.float64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.float64).reshape(-1)
    )
    if len(frames) != len(progress) or np.any(np.diff(frames) <= 0):
        raise ValueError("frame_indices must match progress and be strictly increasing.")
    frame_ranks = average_ranks(frames)
    corr = float(np.corrcoef(progress_ranks, frame_ranks)[0, 1])
    if not math.isfinite(corr):
        return None, True
    return float(np.clip(corr, -1.0, 1.0)), False


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to equal values, like Spearman rank correlation.

    The exact rank origin does not matter for correlation.  We use zero-based
    ranks to keep the arithmetic simple: tied values spanning sorted positions
    ``start..end-1`` all receive ``(start + end - 1) / 2``.
    """

    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def summarize_success_voc_diagnostics(
    success_scores: Sequence[EpisodeScore],
) -> dict[str, Any]:
    """Summarize the worst successful episodes whose rank VOC is below 1."""

    degraded = [
        score
        for score in success_scores
        if score.normalized_voc is not None
        and score.normalized_voc < 1.0 - DEFAULT_VOC_DEGRADED_EPS
        and score.voc_diagnostic is not None
    ]
    degraded.sort(key=lambda score: (float(score.normalized_voc), score.source_traj_id))
    worst = degraded[:DEFAULT_MAX_DIAGNOSTIC_EPISODES]
    return {
        "num_degraded_success": len(degraded),
        "max_episodes": DEFAULT_MAX_DIAGNOSTIC_EPISODES,
        "worst_episodes": [score.voc_diagnostic for score in worst],
    }


def build_success_voc_diagnostic(
    *,
    source_traj_id: str,
    voc: float | None,
    normalized_voc: float,
    frame_indices: np.ndarray,
    progress: np.ndarray,
    is_constant: bool,
) -> dict[str, Any]:
    """Build a compact explanation for rank-VOC loss on one success episode."""

    diffs = np.diff(progress)
    # Diagnose literal drops and ties at zero, independent of the FPSR grid.
    # A large sensitivity threshold must not hide a success-curve regression.
    drop_intervals = summarize_transition_intervals(
        progress,
        frame_indices=frame_indices,
        mask=diffs < 0.0,
        interval_kind="drop",
    )
    # Plateaus are not strict regressions, but average-rank Spearman VOC treats
    # tied progress values as shared ranks, so long plateaus can still reduce
    # the score.  Reporting them separately keeps this distinction visible.
    plateau_intervals = summarize_transition_intervals(
        progress,
        frame_indices=frame_indices,
        mask=diffs == 0.0,
        interval_kind="plateau",
    )
    max_adjacent_drop = None
    if len(diffs) > 0 and float(np.min(diffs)) < 0.0:
        drop_index = int(np.argmin(diffs))
        max_adjacent_drop = {
            "start_frame": int(frame_indices[drop_index]),
            "end_frame": int(frame_indices[drop_index + 1]),
            "from_progress": float(progress[drop_index]),
            "to_progress": float(progress[drop_index + 1]),
            "drop": float(progress[drop_index] - progress[drop_index + 1]),
        }

    unique_values, counts = np.unique(progress, return_counts=True)
    tied_frame_count = int(np.sum(counts[counts > 1]))
    return {
        "source_traj_id": source_traj_id,
        "voc": None if voc is None else float(voc),
        "nVOC": float(normalized_voc),
        "is_constant": bool(is_constant),
        "num_frames": int(len(progress)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "progress_final": float(progress[-1]),
        "progress_min": float(np.min(progress)),
        "progress_max": float(np.max(progress)),
        "progress_mean": float(np.mean(progress)),
        "num_unique_progress_values": int(len(unique_values)),
        "tie_fraction": float(tied_frame_count / len(progress)) if len(progress) else 0.0,
        "num_adjacent_drops": int(np.sum(diffs < 0.0)),
        "num_adjacent_plateaus": int(np.sum(diffs == 0.0)),
        "max_adjacent_drop": max_adjacent_drop,
        "drop_intervals": drop_intervals[:DEFAULT_MAX_INTERVALS_PER_EPISODE],
        "plateau_intervals": plateau_intervals[:DEFAULT_MAX_INTERVALS_PER_EPISODE],
    }


def summarize_transition_intervals(
    progress: np.ndarray,
    *,
    frame_indices: np.ndarray,
    mask: np.ndarray,
    interval_kind: str,
) -> list[dict[str, Any]]:
    """Compress adjacent transition masks into frame intervals for diagnostics."""

    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for transition_index, enabled in enumerate(mask):
        if bool(enabled) and start is None:
            start = transition_index
        elif not bool(enabled) and start is not None:
            intervals.append((start, transition_index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask) - 1))

    rendered: list[dict[str, Any]] = []
    for start_transition, end_transition in intervals:
        start_position = start_transition
        end_position = end_transition + 1
        start_frame = int(frame_indices[start_position])
        end_frame = int(frame_indices[end_position])
        segment = progress[start_position : end_position + 1]
        if interval_kind == "drop":
            rendered.append(
                {
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "length_frames": int(end_frame - start_frame + 1),
                    "num_model_outputs": int(end_position - start_position + 1),
                    "from_progress": float(progress[start_position]),
                    "to_progress": float(progress[end_position]),
                    "drop": float(progress[start_position] - progress[end_position]),
                    "min_progress": float(np.min(segment)),
                    "max_progress": float(np.max(segment)),
                }
            )
        elif interval_kind == "plateau":
            rendered.append(
                {
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "length_frames": int(end_frame - start_frame + 1),
                    "num_model_outputs": int(end_position - start_position + 1),
                    "progress": float(np.mean(segment)),
                }
            )
        else:
            raise ValueError(f"Unknown interval kind: {interval_kind}")

    if interval_kind == "drop":
        rendered.sort(key=lambda item: (item["drop"], item["length_frames"]), reverse=True)
    else:
        rendered.sort(key=lambda item: item["length_frames"], reverse=True)
    return rendered


def top_fraction_mean(
    values: np.ndarray,
    *,
    fraction: float,
    min_count: int = FAILURE_TOP_MIN_COUNT,
) -> tuple[float, int]:
    """Mean the upper tail while keeping at least ``min_count`` observations."""

    if len(values) == 0:
        raise ValueError("Cannot compute top-fraction mean for an empty progress curve.")
    if min_count <= 0:
        raise ValueError(f"min_count must be positive, got {min_count}.")
    count = min(
        len(values),
        max(int(min_count), int(math.ceil(len(values) * float(fraction)))),
    )
    largest = np.partition(values.astype(np.float64), -count)[-count:]
    return float(np.mean(largest)), int(count)


def build_fixed_frame_indices(*, num_frames: int, max_frames: int) -> np.ndarray:
    """Build and verify the endpoint-preserving fixed-frame appendix view."""

    indices = build_progress_sample_indices(
        num_frames=int(num_frames),
        max_frames=int(max_frames),
    )
    expected_count = min(int(num_frames), int(max_frames))
    if len(indices) != expected_count:
        raise ValueError(
            "Fixed-frame sampling must return exactly "
            f"{expected_count} indices, got {indices.tolist()}."
        )
    if int(indices[0]) != 0 or int(indices[-1]) != int(num_frames) - 1:
        raise ValueError(
            "Fixed-frame sampling must include the first and terminal frames, "
            f"got {indices.tolist()} for num_frames={num_frames}."
        )
    return indices


def select_fixed_frame_native_outputs(
    *,
    source_traj_id: str,
    expected_indices: np.ndarray,
    actual_indices: np.ndarray,
    actual_progress_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select canonical indices from raw outputs without consulting filled rows."""

    expected = np.asarray(expected_indices, dtype=np.int64).reshape(-1)
    actual = np.asarray(actual_indices, dtype=np.int64).reshape(-1)
    progress = np.asarray(actual_progress_values).reshape(-1)
    if len(actual) != len(progress):
        raise ValueError(
            f"{source_traj_id}: model-output frame/value lengths must match, got "
            f"{len(actual)} and {len(progress)}."
        )

    positions_by_frame = {
        int(frame_index): position for position, frame_index in enumerate(actual)
    }
    missing = [
        int(frame_index)
        for frame_index in expected
        if int(frame_index) not in positions_by_frame
    ]
    if missing:
        raise ValueError(
            f"{source_traj_id}: fixed-frame analysis is missing native model outputs "
            f"at canonical frame indices {missing}; expected {expected.tolist()}, got "
            f"{actual.tolist()}. Interpolation/fill rows are not accepted."
        )
    selected_positions = np.asarray(
        [positions_by_frame[int(frame_index)] for frame_index in expected],
        dtype=np.int64,
    )
    return expected, progress[selected_positions]


def _resolve_config_path(raw_value: Any, *, base_dir: Path, field_name: str) -> Path:
    if raw_value in (None, ""):
        raise ValueError(f"{field_name} is required.")
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require_config_int(
    raw_value: Any,
    *,
    config_path: Path,
    field_name: str,
) -> int:
    """Require an explicit YAML integer without truncating floats or booleans."""

    if type(raw_value) is not int:
        raise TypeError(
            f"{config_path}: {field_name} must be an integer, got {raw_value!r}."
        )
    return raw_value


def _require_config_number(
    raw_value: Any,
    *,
    config_path: Path,
    field_name: str,
) -> float:
    """Require a finite YAML number without accepting booleans or strings."""

    if type(raw_value) not in (int, float):
        raise TypeError(
            f"{config_path}: {field_name} must be a number, got {raw_value!r}."
        )
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(
            f"{config_path}: {field_name} must be finite, got {raw_value!r}."
        )
    return value


def _require_config_bool(
    raw_value: Any,
    *,
    config_path: Path,
    field_name: str,
) -> bool:
    """Require an explicit YAML boolean instead of applying Python truthiness."""

    if type(raw_value) is not bool:
        raise TypeError(
            f"{config_path}: {field_name} must be true or false, got {raw_value!r}."
        )
    return raw_value


if __name__ == "__main__":
    main()
