"""Batch scheduler for running reward-agent sessions over LeRobot episodes."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .batch_journal import BatchJournal
from .config import RewardAgentConfig
from .lerobot_source import LeRobotEpisodeSource, iter_lerobot_episode_sources
from .result_index import (
    load_completed_reward_agent_entries,
    load_lerobot_episode_index,
    load_reward_agent_progress_curve,
)
from .runner import ChatClientProtocol, RewardAgentRunner
from .schemas import ProgressRow
from .serving import ResolvedEndpoint, ServingManager
from .session import RewardAgentSession
from .task_context import (
    ResolvedTaskContext,
    TaskContextCoverage,
    TaskContextResolver,
    summarize_task_context_coverage,
)


PROGRESS_NORMALIZATION_RULE = "divide_by_scale"


@dataclass(frozen=True, slots=True)
class BatchRunSummary:
    """Filesystem outputs produced by one reward-agent batch run."""

    batch_dir: Path
    batch_results: Path
    batch_events: Path
    batch_summary: Path
    serving_resolved: Path
    total_episodes: int


@dataclass(frozen=True, slots=True)
class BatchWorkspace:
    """One atomically reserved output directory owned by a single batch."""

    batch_id: str
    batch_dir: Path
    servers_dir: Path
    batch_results: Path
    batch_events: Path
    batch_summary: Path
    serving_resolved: Path

    @classmethod
    def reserve(
        cls,
        *,
        output_root: str | Path,
        batch_id: str,
    ) -> "BatchWorkspace":
        """Claim a fresh batch directory before any external side effect."""

        resolved_batch_id = _validate_batch_id(batch_id)
        batch_dir = reserve_batch_directory(
            output_root=output_root,
            batch_id=resolved_batch_id,
        )
        return cls(
            batch_id=resolved_batch_id,
            batch_dir=batch_dir,
            servers_dir=batch_dir / "servers",
            batch_results=batch_dir / "batch_results.jsonl",
            batch_events=batch_dir / "batch_events.jsonl",
            batch_summary=batch_dir / "batch_summary.json",
            serving_resolved=batch_dir / "serving_resolved.yaml",
        )


@dataclass(frozen=True, slots=True)
class BatchTaskContextPlan:
    """Resolved task-bound text and references for every scheduled episode."""

    by_source_traj_id: dict[str, ResolvedTaskContext]
    coverage: TaskContextCoverage


@dataclass(frozen=True, slots=True)
class ExternalProposalPlan:
    """Validated progress proposals selected from one external batch artifact."""

    batch_results_path: Path
    sha256: str
    rows_by_source_traj_id: dict[str, tuple[ProgressRow, ...]]
    total_rows: int
    selected_rows: int
    ignored_rows: int
    non_completed_rows: int
    target_scale_points: int

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return provenance and scope counts without serializing curve data."""

        return {
            "batch_results": str(self.batch_results_path),
            "sha256": self.sha256,
            "total_rows": int(self.total_rows),
            "selected_rows": int(self.selected_rows),
            "ignored_rows": int(self.ignored_rows),
            "non_completed_rows": int(self.non_completed_rows),
            "target_scale_points": int(self.target_scale_points),
        }


@dataclass(frozen=True, slots=True)
class PreparedBatchRun:
    """Validated immutable inputs plus the workspace reserved for their run."""

    workspace: BatchWorkspace
    episodes: tuple[LeRobotEpisodeSource, ...]
    task_context: BatchTaskContextPlan
    external_proposal: ExternalProposalPlan | None = None


class RewardAgentBatchRunner:
    """Run reward-agent over many LeRobot episodes with endpoint leases."""

    def __init__(
        self,
        *,
        lerobot_root: str | Path,
        output_root: str | Path,
        config: RewardAgentConfig,
        serving_manager: ServingManager,
        batch_id: str | None = None,
        max_retries: int | None = None,
        proposal_batch_results: str | Path | None = None,
        check_server_ready: bool = False,
        print_context: bool = False,
        client_factory: Callable[[ResolvedEndpoint], ChatClientProtocol] | None = None,
    ) -> None:
        self.lerobot_root = Path(lerobot_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.config = config
        self.serving_manager = serving_manager
        self.batch_id = (
            batch_id
            if batch_id is not None
            else datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        )
        # The scheduler treats max_retries as total attempts per episode:
        # max_retries=3 means attempt_00, attempt_01, and attempt_02 at most.
        configured_max_retries = config.batch.max_retries if max_retries is None else max_retries
        self.max_retries = max(1, int(configured_max_retries))
        self.check_server_ready = bool(check_server_ready)
        self.print_context = bool(print_context)
        self.client_factory = client_factory
        self.proposal_batch_results = (
            None
            if proposal_batch_results is None
            else Path(proposal_batch_results).expanduser().resolve()
        )
        self._external_proposal: ExternalProposalPlan | None = None
        self._prepared: PreparedBatchRun | None = None
        self._has_run = False

    def prepare(
        self,
        *,
        sources: Iterable[LeRobotEpisodeSource] | None = None,
    ) -> PreparedBatchRun:
        """Validate all local inputs, then atomically reserve the batch directory."""

        if self._prepared is not None:
            raise RuntimeError(
                f"Batch runner for batch_id={self.batch_id!r} has already been prepared."
            )

        episodes = list(sources) if sources is not None else iter_lerobot_episode_sources(self.lerobot_root)
        if not episodes:
            raise ValueError(f"No LeRobot episodes found under {self.lerobot_root}.")
        episodes = _filter_episodes_by_indices(
            episodes,
            episode_indices=self.config.batch.episode_indices,
        )
        external_proposal: ExternalProposalPlan | None = None
        if self.proposal_batch_results is not None:
            external_proposal = _load_external_proposal_plan(
                self.proposal_batch_results,
                lerobot_root=self.lerobot_root,
                episodes=episodes,
                target_scale_points=int(self.config.progress.value_max_points),
            )
            episodes = [
                _attach_external_proposal(
                    source,
                    external_proposal.rows_by_source_traj_id[source.source_traj_id],
                )
                for source in episodes
            ]
        self._external_proposal = external_proposal
        if self.config.is_success_ground_truth.enabled:
            # Validate and cache every selected label before workers create
            # per-attempt outputs or send any model requests.
            for source in episodes:
                source.read_is_success()

        task_context_plan = _build_task_context_plan(episodes, self.config)
        endpoint_queue = _build_endpoint_queue(self.serving_manager.endpoints)
        if endpoint_queue.qsize() <= 0:
            raise ValueError("No endpoint capacity is available for batch serving.")

        prepared = PreparedBatchRun(
            workspace=BatchWorkspace.reserve(
                output_root=self.output_root,
                batch_id=self.batch_id,
            ),
            episodes=tuple(episodes),
            task_context=task_context_plan,
            external_proposal=external_proposal,
        )
        self._prepared = prepared
        return prepared

    def run(self, prepared: PreparedBatchRun) -> BatchRunSummary:
        """Consume one prepared batch and write its durable result artifacts."""

        if prepared is not self._prepared:
            raise ValueError("Prepared batch does not belong to this runner instance.")
        if self._has_run:
            raise RuntimeError(
                f"Batch runner for batch_id={self.batch_id!r} has already run."
            )
        self._has_run = True

        workspace = prepared.workspace
        episodes = list(prepared.episodes)
        task_context_plan = prepared.task_context
        external_proposal = prepared.external_proposal
        journal = BatchJournal(workspace.batch_dir)
        serving_resolved = self.serving_manager.write_resolved(
            workspace.serving_resolved
        )

        endpoint_queue = _build_endpoint_queue(self.serving_manager.endpoints)
        max_workers = endpoint_queue.qsize()

        final_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._run_episode_with_retries,
                    source,
                    task_context=task_context_plan.by_source_traj_id[source.source_traj_id],
                    endpoint_queue=endpoint_queue,
                    batch_dir=workspace.batch_dir,
                    journal=journal,
                )
                for source in episodes
            ]
            for future in as_completed(futures):
                # Re-raise unexpected scheduler errors after the per-episode
                # failure row has had a chance to be written.
                final_rows.append(future.result())

        summary_payload: dict[str, Any] = {
            "batch_id": self.batch_id,
            "batch_dir": _path_relative_to(
                workspace.batch_dir,
                base_dir=workspace.batch_dir,
            ),
            "total_episodes": len(episodes),
            "episode_indices": [int(source.episode_index) for source in episodes],
            "status_counts": _status_counts(final_rows),
            "nonfatal_artifact_warning_counts": _nonfatal_artifact_warning_counts(final_rows),
            "nonfatal_artifact_warnings": _batch_nonfatal_artifact_warnings(final_rows),
            "task_context": task_context_plan.coverage.to_dict(),
            "progress_output_mode": "dense",
            "progress_value_type": "integer",
            "native_progress_value_type": "integer",
            "progress_value_min_points": 0,
            "progress_value_max_points": int(self.config.progress.value_max_points),
            "progress_normalization_rule": PROGRESS_NORMALIZATION_RULE,
            "progress_normalization_scale_points": int(
                self.config.progress.value_max_points
            ),
        }
        if external_proposal is not None:
            summary_payload["external_proposal"] = external_proposal.to_manifest_dict()
        journal.refresh_summary(summary_payload)

        return BatchRunSummary(
            batch_dir=workspace.batch_dir,
            batch_results=workspace.batch_results,
            batch_events=workspace.batch_events,
            batch_summary=workspace.batch_summary,
            serving_resolved=serving_resolved,
            total_episodes=len(episodes),
        )

    def _run_episode_with_retries(
        self,
        source: LeRobotEpisodeSource,
        *,
        task_context: ResolvedTaskContext,
        endpoint_queue: "queue.Queue[ResolvedEndpoint]",
        batch_dir: Path,
        journal: BatchJournal,
    ) -> dict[str, Any]:
        final_row: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt_index in range(self.max_retries):
            endpoint = endpoint_queue.get()
            try:
                attempt_row = self._run_one_attempt(
                    source,
                    task_context=task_context,
                    endpoint=endpoint,
                    attempt_index=attempt_index,
                    batch_dir=batch_dir,
                    journal=journal,
                )
            except Exception as exc:
                last_error = str(exc)
                attempt_row = self._failed_attempt_row(
                    source,
                    task_context=task_context,
                    endpoint=endpoint,
                    attempt_index=attempt_index,
                    batch_dir=batch_dir,
                    error=last_error,
                    traceback_text=traceback.format_exc(),
                )
                journal.append_event({"event": "attempt_failed", **attempt_row})
            finally:
                endpoint_queue.put(endpoint)

            if attempt_row.get("status") == "completed":
                final_row = attempt_row
                break
            final_row = attempt_row

        if final_row is None:
            final_row = self._final_failed_row(
                source,
                task_context=task_context,
                batch_dir=batch_dir,
                error=last_error or "unknown failure",
            )
        journal.append_terminal_result(_batch_result_row(final_row))
        return final_row

    def _run_one_attempt(
        self,
        source: LeRobotEpisodeSource,
        *,
        task_context: ResolvedTaskContext,
        endpoint: ResolvedEndpoint,
        attempt_index: int,
        batch_dir: Path,
        journal: BatchJournal,
    ) -> dict[str, Any]:
        episode_dir = batch_dir / f"episode_{source.episode_index:06d}"
        attempt_dir = episode_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = _run_manifest_payload(
            source,
            endpoint=endpoint,
            attempt_index=attempt_index,
            config=self.config,
            task_context=task_context,
            external_proposal=self._external_proposal,
        )
        run_manifest["status"] = "running"
        _write_json(attempt_dir / "run_manifest.json", run_manifest)
        journal.append_event(
            {
                "event": "attempt_started",
                "source_traj_id": source.source_traj_id,
                "episode_index": int(source.episode_index),
                "attempt_index": int(attempt_index),
                "endpoint": endpoint.to_dict(),
                "attempt_dir": _path_relative_to(attempt_dir, base_dir=batch_dir),
                "task_context_matched": bool(task_context.matched),
            },
        )

        endpoint_config = _config_for_endpoint(self.config, endpoint)
        session = RewardAgentSession(
            episode_source=source,
            run_dir=attempt_dir,
            config=endpoint_config,
            base_robot_instruction=task_context.base_instruction,
            task_context=task_context,
        )
        runner = RewardAgentRunner(
            session=session,
            config=endpoint_config,
            client=None if self.client_factory is None else self.client_factory(endpoint),
            print_context=self.print_context,
            check_server_ready=self.check_server_ready,
        )
        status = runner.run()
        result_manifest_path = attempt_dir / "result_manifest.json"
        corrected_progress_csv = _corrected_progress_csv_from_manifest(result_manifest_path)
        nonfatal_artifact_warnings = _nonfatal_artifact_warnings_from_manifest(result_manifest_path)
        completed = bool(status.get("completed")) and corrected_progress_csv is not None
        attempt_status = (
            "completed"
            if completed
            else _incomplete_attempt_status(
                runner_status=status,
                corrected_progress_csv=corrected_progress_csv,
                result_manifest_path=result_manifest_path,
            )
        )

        run_manifest["status"] = attempt_status
        _write_json(attempt_dir / "run_manifest.json", run_manifest)
        row = {
            **_batch_identity_fields(source),
            "attempt_index": int(attempt_index),
            "attempt_dir": _path_relative_to(attempt_dir, base_dir=batch_dir),
            "endpoint_base_url": endpoint.base_url,
            "status": attempt_status,
            "runner_status": dict(status),
            # Batch rows intentionally keep only the scalar routing signal
            # needed for later aggregation. The concrete review focus remains
            # in the per-attempt final_result.json artifact.
            "external_review_need": (
                int(session.external_review_need)
                if completed and session.external_review_need is not None
                else None
            ),
            "corrected_progress_csv": _path_relative_to(corrected_progress_csv, base_dir=batch_dir)
            if corrected_progress_csv is not None
            else None,
            "nonfatal_artifact_warnings": nonfatal_artifact_warnings,
            "progress_output_mode": "dense",
            "progress_value_type": "integer",
            "native_progress_value_type": "integer",
            "progress_value_min_points": 0,
            "progress_value_max_points": int(endpoint_config.progress.value_max_points),
            "progress_normalization_rule": PROGRESS_NORMALIZATION_RULE,
            "progress_normalization_scale_points": int(
                endpoint_config.progress.value_max_points
            ),
            "task_context_matched": bool(task_context.matched),
        }
        journal.append_event({"event": "attempt_finished", **row})
        return row

    def _failed_attempt_row(
        self,
        source: LeRobotEpisodeSource,
        *,
        task_context: ResolvedTaskContext,
        endpoint: ResolvedEndpoint,
        attempt_index: int,
        batch_dir: Path,
        error: str,
        traceback_text: str,
    ) -> dict[str, Any]:
        attempt_dir = batch_dir / f"episode_{source.episode_index:06d}" / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            attempt_dir / "run_manifest.json",
            {
                **_run_manifest_payload(
                    source,
                    endpoint=endpoint,
                    attempt_index=attempt_index,
                    config=self.config,
                    task_context=task_context,
                    external_proposal=self._external_proposal,
                ),
                "status": "failed",
                "error": error,
            },
        )
        _write_json(
            attempt_dir / "error.json",
            {
                "error": error,
                "traceback": traceback_text,
            },
        )
        return {
            **_batch_identity_fields(source),
            "attempt_index": int(attempt_index),
            "attempt_dir": _path_relative_to(attempt_dir, base_dir=batch_dir),
            "endpoint_base_url": endpoint.base_url,
            "status": "failed",
            "error": error,
            "corrected_progress_csv": None,
            "progress_output_mode": "dense",
            "progress_value_type": "integer",
            "native_progress_value_type": "integer",
            "progress_value_min_points": 0,
            "progress_value_max_points": int(self.config.progress.value_max_points),
            "progress_normalization_rule": PROGRESS_NORMALIZATION_RULE,
            "progress_normalization_scale_points": int(
                self.config.progress.value_max_points
            ),
            "task_context_matched": bool(task_context.matched),
        }

    def _final_failed_row(
        self,
        source: LeRobotEpisodeSource,
        *,
        task_context: ResolvedTaskContext,
        batch_dir: Path,
        error: str,
    ) -> dict[str, Any]:
        return {
            **_batch_identity_fields(source),
            "attempt_index": None,
            "attempt_dir": None,
            "endpoint_base_url": None,
            "status": "failed",
            "error": error,
            "corrected_progress_csv": None,
            "progress_output_mode": "dense",
            "progress_value_type": "integer",
            "native_progress_value_type": "integer",
            "progress_value_min_points": 0,
            "progress_value_max_points": int(self.config.progress.value_max_points),
            "progress_normalization_rule": PROGRESS_NORMALIZATION_RULE,
            "progress_normalization_scale_points": int(
                self.config.progress.value_max_points
            ),
            "task_context_matched": bool(task_context.matched),
        }

def _build_endpoint_queue(endpoints: list[ResolvedEndpoint]) -> "queue.Queue[ResolvedEndpoint]":
    endpoint_queue: "queue.Queue[ResolvedEndpoint]" = queue.Queue()
    for endpoint in endpoints:
        for _ in range(max(1, int(endpoint.max_concurrent))):
            endpoint_queue.put(endpoint)
    return endpoint_queue


def _filter_episodes_by_indices(
    episodes: list[LeRobotEpisodeSource],
    *,
    episode_indices: tuple[int, ...] | None,
) -> list[LeRobotEpisodeSource]:
    """Apply an explicit episode_index schedule after LeRobot metadata is loaded."""

    if episode_indices is None:
        return episodes

    episodes_by_index: dict[int, LeRobotEpisodeSource] = {}
    duplicate_available: set[int] = set()
    for source in episodes:
        episode_index = int(source.episode_index)
        if episode_index in episodes_by_index:
            duplicate_available.add(episode_index)
            continue
        episodes_by_index[episode_index] = source
    if duplicate_available:
        raise ValueError(
            "LeRobot metadata contains duplicate episode_index values: "
            f"{sorted(duplicate_available)}."
        )

    missing = [index for index in episode_indices if index not in episodes_by_index]
    if missing:
        available = sorted(episodes_by_index)
        raise ValueError(
            "Configured batch.episode_indices include missing episode_index values: "
            f"{missing}. Available episode_index values: {_format_episode_index_sample(available)}."
        )

    return [episodes_by_index[int(index)] for index in episode_indices]


def _format_episode_index_sample(indices: list[int]) -> str:
    """Keep missing-index errors readable for large LeRobot datasets."""

    if not indices:
        return "[]"
    if len(indices) <= 20:
        return str(indices)
    return (
        f"{indices[0]}..{indices[-1]} "
        f"({len(indices)} total; first={indices[:5]}, last={indices[-5:]})"
    )


def reserve_batch_directory(
    *,
    output_root: str | Path,
    batch_id: str,
) -> Path:
    """Atomically claim one batch id and reject every kind of stale directory."""

    resolved_batch_id = _validate_batch_id(batch_id)
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = resolved_output_root / resolved_batch_id
    try:
        batch_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Batch workspace already exists for batch_id={resolved_batch_id!r}: "
            f"{batch_dir}. Choose a new --batch-id or output root."
        ) from exc
    return batch_dir


def _validate_batch_id(batch_id: str) -> str:
    """Keep a batch id inside its configured output root as one path component."""

    value = str(batch_id)
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(
            "batch_id must be a non-empty single path component without path separators, "
            f"got {value!r}."
        )
    return value


def _build_task_context_plan(
    episodes: list[LeRobotEpisodeSource],
    config: RewardAgentConfig,
) -> BatchTaskContextPlan:
    """Resolve all task-bound reward-agent context before launching workers."""

    resolver = TaskContextResolver.from_config(config.task_context)
    by_source_traj_id: dict[str, ResolvedTaskContext] = {}
    for source in episodes:
        resolved = resolver.resolve(source.read_task())
        by_source_traj_id[source.source_traj_id] = resolved
    resolved_values = list(by_source_traj_id.values())
    resolver.require_coverage(resolved_values)
    return BatchTaskContextPlan(
        by_source_traj_id=by_source_traj_id,
        coverage=summarize_task_context_coverage(resolved_values),
    )


def _load_external_proposal_plan(
    batch_results_path: Path,
    *,
    lerobot_root: Path,
    episodes: list[LeRobotEpisodeSource],
    target_scale_points: int,
) -> ExternalProposalPlan:
    """Validate selected proposal rows and convert them into session rows.

    The shared result index performs all artifact-level checks against the
    LeRobot metadata. This adapter then projects each validated normalized
    curve into the reward-agent integer value space consumed by ``ProgressRow``.
    """

    resolved_path = Path(batch_results_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"External proposal batch_results not found: {resolved_path}"
        )
    if not episodes:
        raise ValueError("Cannot load an external proposal without selected episodes.")
    if isinstance(target_scale_points, bool) or int(target_scale_points) <= 0:
        raise ValueError(
            "External proposal target scale must be a positive integer, got "
            f"{target_scale_points!r}."
        )

    episodes_by_source = load_lerobot_episode_index(lerobot_root)
    selected_source_ids = {str(source.source_traj_id) for source in episodes}
    result_index = load_completed_reward_agent_entries(
        batch_results=[resolved_path],
        episodes_by_source=episodes_by_source,
        lerobot_root=lerobot_root,
        included_source_traj_ids=selected_source_ids,
        allow_partial=False,
        label="external proposal",
        strict_coverage_hint=(
            "Ensure --proposal-batch-results contains one completed row for "
            "every selected LeRobot episode."
        ),
        require_progress_normalization_scale=True,
        strict_schema=True,
    )

    rows_by_source: dict[str, tuple[ProgressRow, ...]] = {}
    target_scale = int(target_scale_points)
    for source in episodes:
        source_traj_id = str(source.source_traj_id)
        entry = result_index.completed_by_source.get(source_traj_id)
        if entry is None:  # Defensive guard; strict coverage should catch this.
            raise ValueError(
                "external proposal coverage is missing selected source_traj_id "
                f"{source_traj_id!r}."
            )
        episode = episodes_by_source[source_traj_id]
        curve = load_reward_agent_progress_curve(entry, episode)
        normalized = np.asarray(curve.progress_normalized, dtype=np.float64)
        if len(normalized) != int(source.num_frames):
            raise ValueError(
                f"{resolved_path}: proposal curve for {source_traj_id!r} has "
                f"{len(normalized)} rows, expected {source.num_frames}."
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError(
                f"{resolved_path}: proposal curve for {source_traj_id!r} contains "
                "non-finite normalized values."
            )
        if np.any(normalized < 0.0) or np.any(normalized > 1.0):
            minimum = float(np.min(normalized))
            maximum = float(np.max(normalized))
            raise ValueError(
                f"{resolved_path}: proposal curve for {source_traj_id!r} has "
                f"normalized range [{minimum:g}, {maximum:g}], but run-batch "
                "proposals must lie in [0, 1]."
            )

        proposal_points = np.rint(normalized * float(target_scale)).astype(np.int64)
        source_rows = tuple(
            ProgressRow(
                frame_index=int(frame_index),
                progress_points=int(progress_point),
                source="external_proposal",
            )
            for frame_index, progress_point in enumerate(proposal_points.tolist())
        )
        rows_by_source[source_traj_id] = source_rows

    return ExternalProposalPlan(
        batch_results_path=resolved_path,
        sha256=_sha256_file(resolved_path),
        rows_by_source_traj_id=rows_by_source,
        total_rows=int(result_index.total_rows),
        selected_rows=int(result_index.completed_rows),
        ignored_rows=int(result_index.ignored_out_of_scope),
        non_completed_rows=int(result_index.ignored_non_completed),
        target_scale_points=target_scale,
    )


def _attach_external_proposal(
    source: LeRobotEpisodeSource,
    progress_rows: tuple[ProgressRow, ...],
) -> LeRobotEpisodeSource:
    """Attach validated proposal rows without materializing a case pack."""

    attach = getattr(source, "with_progress_rows", None)
    if not callable(attach):
        raise TypeError(
            "External proposals require LeRobot episode sources that support "
            "with_progress_rows()."
        )
    return attach(progress_rows)


def _config_for_endpoint(config: RewardAgentConfig, endpoint: ResolvedEndpoint) -> RewardAgentConfig:
    payload = config.to_dict()
    vllm_payload = dict(payload.get("vllm") or {})
    vllm_payload["base_url"] = endpoint.base_url
    payload["vllm"] = vllm_payload
    return RewardAgentConfig.from_dict(payload)


def _batch_identity_fields(source: LeRobotEpisodeSource) -> dict[str, Any]:
    return {
        "source_traj_id": source.source_traj_id,
        "episode_index": int(source.episode_index),
        "dataset_root": str(source.dataset_root),
        "source_subset": source.source_subset,
        "num_transition_rows": int(source.num_frames),
        "frame_axis": "lerobot_transition_rows",
        "terminal_state_policy": "ignored",
    }


def _run_manifest_payload(
    source: LeRobotEpisodeSource,
    *,
    endpoint: ResolvedEndpoint,
    attempt_index: int,
    config: RewardAgentConfig,
    task_context: ResolvedTaskContext,
    external_proposal: ExternalProposalPlan | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **_batch_identity_fields(source),
        "lerobot_root": str(source.dataset_root),
        "attempt_index": int(attempt_index),
        "endpoint": endpoint.to_dict(),
        "progress_output_mode": "dense",
        "progress_value_type": "integer",
        "native_progress_value_type": "integer",
        "progress_value_min_points": 0,
        "progress_value_max_points": int(config.progress.value_max_points),
        "progress_normalization_rule": PROGRESS_NORMALIZATION_RULE,
        "progress_normalization_scale_points": int(
            config.progress.value_max_points
        ),
        "base_robot_instruction": task_context.base_instruction,
        "robot_instruction": task_context.final_instruction,
        "task_context": task_context.to_manifest_dict(),
    }
    if external_proposal is not None:
        payload["external_proposal"] = external_proposal.to_manifest_dict()
    return payload


def _corrected_progress_csv_from_manifest(manifest_path: Path) -> Path | None:
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = payload.get("progress") if isinstance(payload, dict) else None
    if not isinstance(progress, dict):
        return None
    raw_path = progress.get("corrected_progress_csv")
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve() if path.is_file() else None


def _path_relative_to(path: str | Path, *, base_dir: Path) -> str:
    """Serialize a path relative to the JSON file directory that will contain it."""

    resolved_path = Path(path).expanduser().resolve()
    resolved_base = Path(base_dir).expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, start=resolved_base))
    return relative.as_posix() if relative.as_posix() else "."


def _nonfatal_artifact_warnings_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    warnings = payload.get("nonfatal_artifact_warnings") if isinstance(payload, dict) else None
    if not isinstance(warnings, list):
        return []
    return [dict(warning) for warning in warnings if isinstance(warning, dict)]


def _incomplete_attempt_status(
    *,
    runner_status: dict[str, Any],
    corrected_progress_csv: Path | None,
    result_manifest_path: Path,
) -> str:
    raw_status = str(runner_status.get("status", "failed"))
    if raw_status != "completed":
        return raw_status
    if not result_manifest_path.is_file():
        return "failed_missing_result_manifest"
    if corrected_progress_csv is None:
        return "failed_missing_corrected_progress_csv"
    return "failed_incomplete_finalization"


def _batch_result_row(attempt_row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_traj_id",
        "episode_index",
        "dataset_root",
        "source_subset",
        "attempt_dir",
        "status",
        "external_review_need",
        "corrected_progress_csv",
        "nonfatal_artifact_warnings",
        "num_transition_rows",
        "frame_axis",
        "terminal_state_policy",
        "progress_output_mode",
        "progress_value_type",
        "native_progress_value_type",
        "progress_value_min_points",
        "progress_value_max_points",
        "progress_normalization_rule",
        "progress_normalization_scale_points",
        "task_context_matched",
    ]
    return {key: attempt_row.get(key) for key in keys}


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _batch_nonfatal_artifact_warnings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for row in rows:
        row_warnings = row.get("nonfatal_artifact_warnings")
        if not isinstance(row_warnings, list):
            continue
        for warning in row_warnings:
            if not isinstance(warning, dict):
                continue
            warnings.append(
                {
                    "source_traj_id": row.get("source_traj_id"),
                    "episode_index": row.get("episode_index"),
                    "attempt_dir": row.get("attempt_dir"),
                    **dict(warning),
                }
            )
    return warnings


def _nonfatal_artifact_warning_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for warning in _batch_nonfatal_artifact_warnings(rows):
        artifact = str(warning.get("artifact", "unknown"))
        counts[artifact] = counts.get(artifact, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    """Hash an immutable input artifact for reproducible batch manifests."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
