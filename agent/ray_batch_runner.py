"""Ray orchestration for statically sharded multi-node reward-agent batches."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .batch_runner import (
    _load_external_proposal_plan,
    reserve_batch_directory,
)
from .config import RewardAgentConfig
from .lerobot_source import LeRobotEpisodeSource, select_lerobot_episode_sources
from .ray_worker import RewardAgentRayNodeWorker
from .serving import (
    derive_client_endpoints_from_server,
    load_server_config,
    resolve_client_endpoint_groups,
    resolve_distributed_server_gpu_ids,
    resolve_server_groups,
    resolve_server_startup_config,
    validate_client_server_alignment,
)
from .task_context import (
    TaskContextResolver,
    summarize_task_context_coverage,
)


DISTRIBUTION_RULE = "contiguous_balanced_first_remainder"


@dataclass(frozen=True, slots=True)
class RayNodeSpec:
    """One alive Ray node selected for a reward-agent shard."""

    node_id: str
    node_ip: str
    gpu_capacity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ray_node_id": self.node_id,
            "node_ip": self.node_ip,
            "gpu_capacity": self.gpu_capacity,
        }


@dataclass(frozen=True, slots=True)
class RayShardPlan:
    """One deterministic contiguous episode slice assigned to a Ray node."""

    node_rank: int
    node: RayNodeSpec
    episode_indices: tuple[int, ...]

    @property
    def node_name(self) -> str:
        return f"node_{self.node_rank:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_rank": self.node_rank,
            "node_name": self.node_name,
            **self.node.to_dict(),
            "episode_count": len(self.episode_indices),
            "episode_indices": list(self.episode_indices),
            "batch_results": f"nodes/{self.node_name}/batch_results.jsonl",
        }


@dataclass(frozen=True, slots=True)
class RayBatchRunSummary:
    """Global shared-storage outputs from one Ray reward-agent batch."""

    batch_dir: Path
    distributed_manifest: Path
    distributed_summary: Path
    batch_results: tuple[Path, ...]
    total_episodes: int
    num_nodes: int
    status: str


def split_contiguous_episode_indices(
    episode_indices: Sequence[int],
    *,
    num_nodes: int,
) -> tuple[tuple[int, ...], ...]:
    """Split one ordered schedule into contiguous shards differing by at most one."""

    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError(f"num_nodes must be a positive integer, got {num_nodes!r}.")
    ordered = tuple(int(index) for index in episode_indices)
    if len(ordered) < num_nodes:
        raise ValueError(
            "Ray run-batch requires at least one episode per node: "
            f"episodes={len(ordered)}, num_nodes={num_nodes}."
        )

    base_size, remainder = divmod(len(ordered), num_nodes)
    shards: list[tuple[int, ...]] = []
    start = 0
    for node_rank in range(num_nodes):
        shard_size = base_size + (1 if node_rank < remainder else 0)
        stop = start + shard_size
        shards.append(ordered[start:stop])
        start = stop
    return tuple(shards)


def eligible_ray_nodes(
    raw_nodes: Sequence[Mapping[str, Any]],
    *,
    required_gpus_per_node: int,
) -> list[RayNodeSpec]:
    """Filter alive GPU-capable nodes and return them in stable IP order."""

    selected_by_id: dict[str, RayNodeSpec] = {}
    for raw_node in raw_nodes:
        if raw_node.get("Alive") is not True:
            continue
        resources = raw_node.get("Resources") or {}
        if not isinstance(resources, Mapping):
            continue
        gpu_capacity = float(resources.get("GPU", 0.0))
        if gpu_capacity < float(required_gpus_per_node):
            continue
        node_id = str(raw_node.get("NodeID") or "").strip()
        node_ip = str(raw_node.get("NodeManagerAddress") or "").strip()
        if not node_id or not node_ip:
            continue
        selected_by_id[node_id] = RayNodeSpec(
            node_id=node_id,
            node_ip=node_ip,
            gpu_capacity=gpu_capacity,
        )
    return sorted(selected_by_id.values(), key=_ray_node_sort_key)


def wait_for_ray_nodes(
    ray_module: Any,
    *,
    num_nodes: int,
    required_gpus_per_node: int,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> list[RayNodeSpec]:
    """Wait until the platform-started Ray cluster exposes enough GPU nodes."""

    deadline = time.monotonic() + float(timeout_seconds)
    last_eligible: list[RayNodeSpec] = []
    while True:
        last_eligible = eligible_ray_nodes(
            ray_module.nodes(),
            required_gpus_per_node=required_gpus_per_node,
        )
        if len(last_eligible) >= num_nodes:
            return last_eligible[:num_nodes]
        if time.monotonic() >= deadline:
            visible = [node.to_dict() for node in last_eligible]
            raise TimeoutError(
                "Timed out waiting for Ray GPU nodes: "
                f"required_nodes={num_nodes}, "
                f"required_gpus_per_node={required_gpus_per_node}, "
                f"eligible_nodes={visible}."
            )
        time.sleep(float(poll_interval_seconds))


def run_ray_batch(
    *,
    lerobot_root: str | Path,
    output_root: str | Path,
    config: RewardAgentConfig,
    server_config: str | Path,
    num_nodes: int,
    batch_id: str | None,
    proposal_batch_results: str | Path | None = None,
    max_concurrent_per_endpoint: int,
    cluster_ready_timeout_seconds: float,
    print_context: bool,
) -> RayBatchRunSummary:
    """Connect to Ray 2.31+, run fixed node shards, and write global metadata."""

    resolved_lerobot_root = Path(lerobot_root).expanduser().resolve()
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_server_config = Path(server_config).expanduser().resolve()
    server_payload = load_server_config(resolved_server_config)
    startup_config = resolve_server_startup_config(server_payload.get("startup"))
    raw_server_groups = server_payload.get("server_groups", [])
    logical_gpu_ids = resolve_distributed_server_gpu_ids(raw_server_groups)

    # Validate server argv, local ports, and configured client URLs before a
    # Ray connection or shared output directory is created.
    logical_server_endpoints = resolve_server_groups(raw_server_groups)
    _require_unique_server_urls(logical_server_endpoints)
    if config.vllm.endpoint_groups:
        client_endpoints = resolve_client_endpoint_groups(
            list(config.vllm.endpoint_groups)
        )
    else:
        client_endpoints = derive_client_endpoints_from_server(
            logical_server_endpoints,
            max_concurrent=max_concurrent_per_endpoint,
        )
    validate_client_server_alignment(
        client_endpoints=client_endpoints,
        server_endpoints=logical_server_endpoints,
    )

    selected_sources, task_context_coverage = _preflight_sources(
        resolved_lerobot_root,
        config=config,
    )
    external_proposal = None
    resolved_proposal_path = (
        None
        if proposal_batch_results is None
        else Path(proposal_batch_results).expanduser().resolve()
    )
    if resolved_proposal_path is not None:
        external_proposal = _load_external_proposal_plan(
            resolved_proposal_path,
            lerobot_root=resolved_lerobot_root,
            episodes=selected_sources,
            target_scale_points=int(config.progress.value_max_points),
        )
    ordered_episode_indices = [
        int(source.episode_index) for source in selected_sources
    ]
    episode_shards = split_contiguous_episode_indices(
        ordered_episode_indices,
        num_nodes=num_nodes,
    )
    resolved_batch_id = (
        batch_id
        if batch_id is not None
        else datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )

    # Claim the global ID before connecting to Ray or scheduling any actor.
    # Node workspaces are claimed later by their owning actors immediately
    # before those actors start managed servers.
    batch_dir = reserve_batch_directory(
        output_root=resolved_output_root,
        batch_id=resolved_batch_id,
    )
    nodes_output_root = batch_dir / "nodes"
    nodes_output_root.mkdir(parents=False, exist_ok=False)
    manifest_path = batch_dir / "distributed_manifest.json"
    summary_path = batch_dir / "distributed_summary.json"

    ray_module = _import_ray()
    ray_module.init(address="auto")
    actors: list[tuple[RayShardPlan, Any]] = []
    try:
        # Node discovery reports total node capacity, while the placement
        # barrier below proves that each hard-affinity actor has actually
        # acquired its GPU reservation.  Both phases share one cluster-ready
        # deadline so resource contention cannot leave actors pending forever.
        cluster_ready_deadline = time.monotonic() + float(
            cluster_ready_timeout_seconds
        )
        selected_nodes = wait_for_ray_nodes(
            ray_module,
            num_nodes=num_nodes,
            required_gpus_per_node=len(logical_gpu_ids),
            timeout_seconds=_remaining_deadline_seconds(cluster_ready_deadline),
        )
        shard_plans = tuple(
            RayShardPlan(
                node_rank=node_rank,
                node=selected_nodes[node_rank],
                episode_indices=episode_shards[node_rank],
            )
            for node_rank in range(num_nodes)
        )

        _atomic_write_json(
            manifest_path,
            {
                "schema_version": 3,
                "batch_id": resolved_batch_id,
                "lerobot_root": str(selected_sources[0].dataset_root),
                "output_root": str(resolved_output_root),
                "num_nodes": num_nodes,
                "total_episodes": len(selected_sources),
                "episode_indices": ordered_episode_indices,
                "distribution_rule": DISTRIBUTION_RULE,
                "logical_gpu_ids_per_node": list(logical_gpu_ids),
                "gpus_per_node": len(logical_gpu_ids),
                "server_config": str(resolved_server_config),
                "server_config_sha256": _sha256_file(resolved_server_config),
                "server_startup": startup_config.to_dict(),
                "ray_version": str(getattr(ray_module, "__version__", "unknown")),
                "task_context": task_context_coverage,
                "external_proposal": (
                    None
                    if external_proposal is None
                    else external_proposal.to_manifest_dict()
                ),
                "shards": [plan.to_dict() for plan in shard_plans],
            },
        )

        config_payload = _config_payload_for_ray(config)
        remote_worker_class, node_affinity_class = _ray_actor_dependencies(ray_module)
        orchestration_failures: list[dict[str, Any]] = []
        placement_by_rank: dict[int, dict[str, Any]] = {}
        port_preflight_by_rank: dict[int, dict[str, Any]] = {}
        prepared_by_rank: dict[int, dict[str, Any]] = {}
        results_by_rank: dict[int, dict[str, Any]] = {}
        try:
            for plan in shard_plans:
                actor = remote_worker_class.options(
                    num_gpus=len(logical_gpu_ids),
                    scheduling_strategy=node_affinity_class(
                        node_id=plan.node.node_id,
                        soft=False,
                    ),
                    max_restarts=0,
                    max_task_retries=0,
                ).remote(
                    node_rank=plan.node_rank,
                    expected_ray_node_id=plan.node.node_id,
                    episode_indices=list(plan.episode_indices),
                    lerobot_root=str(selected_sources[0].dataset_root),
                    nodes_output_root=str(nodes_output_root),
                    config_payload=config_payload,
                    server_payload=server_payload,
                    shard_plan=plan.to_dict(),
                    proposal_batch_results=(
                        None
                        if resolved_proposal_path is None
                        else str(resolved_proposal_path)
                    ),
                    max_concurrent_per_endpoint=max_concurrent_per_endpoint,
                    print_context=print_context,
                )
                actors.append((plan, actor))

            placement_by_rank, placement_failures = _collect_remote_results(
                ray_module,
                [
                    (plan.node_rank, actor.check_placement.remote())
                    for plan, actor in actors
                ],
                stage="actor_scheduling",
                timeout_seconds=_remaining_deadline_seconds(
                    cluster_ready_deadline
                ),
            )
            for failure in placement_failures:
                node_rank = failure.get("node_rank")
                if node_rank is not None:
                    plan = shard_plans[int(node_rank)]
                    failure["ray_node_id"] = plan.node.node_id
                    failure["node_ip"] = plan.node.node_ip
            orchestration_failures.extend(placement_failures)

            # Reserve every fixed HTTP port and one free TP1 candidate from
            # each replica-local block before any actor starts vLLM.  This
            # keeps transient internal collisions self-healing while retaining
            # a strict all-node barrier for genuinely exhausted layouts.
            if not orchestration_failures:
                port_preflight_by_rank, port_preflight_failures = (
                    _collect_remote_results(
                        ray_module,
                        [
                            (plan.node_rank, actor.check_server_ports.remote())
                            for plan, actor in actors
                        ],
                        stage="server_port_preflight",
                    )
                )
                for failure in port_preflight_failures:
                    node_rank = failure.get("node_rank")
                    if node_rank is not None:
                        plan = shard_plans[int(node_rank)]
                        failure["ray_node_id"] = plan.node.node_id
                        failure["node_ip"] = plan.node.node_ip
                orchestration_failures.extend(port_preflight_failures)

                # An expected bind collision is returned as structured actor
                # data instead of an opaque Ray exception.  Keep that result
                # on the shard while also promoting it to the global failures
                # list that gates prepare() on every node.
                for node_rank, preflight in sorted(port_preflight_by_rank.items()):
                    raw_failure = preflight.get("failure")
                    if preflight.get("status") == "passed":
                        continue
                    plan = shard_plans[node_rank]
                    failure = (
                        dict(raw_failure)
                        if isinstance(raw_failure, Mapping)
                        else {
                            "error_type": "RuntimeError",
                            "error": "Malformed server port preflight result.",
                        }
                    )
                    failure.update(
                        {
                            "stage": "server_port_preflight",
                            "node_rank": node_rank,
                            "node_name": plan.node_name,
                            "ray_node_id": plan.node.node_id,
                            "node_ip": plan.node.node_ip,
                        }
                    )
                    orchestration_failures.append(failure)

            # No node-local server process is started until every actor has
            # acquired its reservation, passed node-ID validation, and passed
            # the port preflight above.
            if not orchestration_failures:
                prepared_by_rank, prepare_failures = _collect_remote_results(
                    ray_module,
                    [
                        (plan.node_rank, actor.prepare.remote())
                        for plan, actor in actors
                    ],
                    stage="prepare",
                )
                orchestration_failures.extend(prepare_failures)

            # A strict readiness barrier prevents faster nodes from writing
            # episode artifacts while another node is still failing startup.
            if not orchestration_failures:
                results_by_rank, run_failures = _collect_remote_results(
                    ray_module,
                    [
                        (plan.node_rank, actor.run_shard.remote())
                        for plan, actor in actors
                    ],
                    stage="run_shard",
                )
                orchestration_failures.extend(run_failures)
        except Exception as exc:
            orchestration_failures.append(
                {
                    "stage": "orchestration",
                    "node_rank": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            cleanup_warnings = _cleanup_ray_actors(
                ray_module,
                actors,
                graceful_shutdown_ranks=set(prepared_by_rank),
            )

        global_status = "failed" if orchestration_failures else "completed"
        distributed_summary_payload = _distributed_summary_payload(
            batch_id=resolved_batch_id,
            shard_plans=shard_plans,
            placement_by_rank=placement_by_rank,
            port_preflight_by_rank=port_preflight_by_rank,
            prepared_by_rank=prepared_by_rank,
            results_by_rank=results_by_rank,
            failures=orchestration_failures,
            cleanup_warnings=cleanup_warnings,
        )
        _atomic_write_json(summary_path, distributed_summary_payload)
        batch_results = tuple(
            Path(results_by_rank[node_rank]["batch_results"]).resolve()
            for node_rank in sorted(results_by_rank)
        )
        return RayBatchRunSummary(
            batch_dir=batch_dir,
            distributed_manifest=manifest_path,
            distributed_summary=summary_path,
            batch_results=batch_results,
            total_episodes=len(selected_sources),
            num_nodes=num_nodes,
            status=global_status,
        )
    finally:
        # ray.shutdown disconnects this submitted driver; it does not stop the
        # platform-owned Ray head or worker daemons.
        ray_module.shutdown()


def _preflight_sources(
    lerobot_root: Path,
    *,
    config: RewardAgentConfig,
) -> tuple[list[LeRobotEpisodeSource], dict[str, Any]]:
    """Validate the full global selection before any node writes outputs."""

    sources = select_lerobot_episode_sources(
        lerobot_root,
        episode_indices=config.batch.episode_indices,
    )
    if config.is_success_ground_truth.enabled:
        for source in sources:
            source.read_is_success()

    resolver = TaskContextResolver.from_config(config.task_context)
    resolved_contexts = [resolver.resolve(source.read_task()) for source in sources]
    resolver.require_coverage(resolved_contexts)
    return sources, summarize_task_context_coverage(resolved_contexts).to_dict()


def _config_payload_for_ray(config: RewardAgentConfig) -> dict[str, Any]:
    """Serialize config and make actor-opened task-context paths unambiguous."""

    payload = config.to_dict()
    task_context = dict(payload.get("task_context") or {})
    raw_path = task_context.get("path")
    if raw_path not in (None, ""):
        task_context["path"] = str(Path(raw_path).expanduser().resolve())
    payload["task_context"] = task_context
    return payload


def _collect_remote_results(
    ray_module: Any,
    refs_by_rank: Sequence[tuple[int, Any]],
    *,
    stage: str,
    timeout_seconds: float | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Collect all actor calls so one failed node does not hide other outcomes."""

    pending = {object_ref: int(node_rank) for node_rank, object_ref in refs_by_rank}
    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    deadline = (
        None
        if timeout_seconds is None
        else time.monotonic() + max(0.0, float(timeout_seconds))
    )
    while pending:
        wait_kwargs: dict[str, Any] = {"num_returns": 1}
        if deadline is not None:
            remaining_seconds = _remaining_deadline_seconds(deadline)
            if remaining_seconds <= 0.0:
                ready = []
            else:
                wait_kwargs["timeout"] = remaining_seconds
                ready, _remaining = ray_module.wait(list(pending), **wait_kwargs)
        else:
            ready, _remaining = ray_module.wait(list(pending), **wait_kwargs)
        if not ready:
            for _object_ref, node_rank in sorted(
                pending.items(),
                key=lambda item: item[1],
            ):
                failures.append(
                    {
                        "stage": stage,
                        "node_rank": node_rank,
                        "error_type": "TimeoutError",
                        "error": (
                            f"Timed out waiting for Ray stage {stage!r} after "
                            f"{float(timeout_seconds or 0.0):.3f} seconds."
                        ),
                    }
                )
            break
        object_ref = ready[0]
        node_rank = pending.pop(object_ref)
        try:
            result = ray_module.get(object_ref)
        except Exception as exc:
            failures.append(
                {
                    "stage": stage,
                    "node_rank": node_rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            results[node_rank] = dict(result)
    return results, failures


def _cleanup_ray_actors(
    ray_module: Any,
    actors: Sequence[tuple[RayShardPlan, Any]],
    *,
    graceful_shutdown_ranks: set[int],
) -> list[dict[str, Any]]:
    """Gracefully stop prepared actors, then release every Ray GPU lease."""

    cleanup_warnings: list[dict[str, Any]] = []
    shutdown_refs: list[tuple[int, Any]] = []
    for plan, actor in actors:
        if plan.node_rank not in graceful_shutdown_ranks:
            # Unprepared actors have no child servers.  In particular, never
            # queue shutdown() behind a placement method that is still pending
            # for resources; kill the actor directly in the loop below.
            continue
        try:
            shutdown_refs.append((plan.node_rank, actor.shutdown.remote()))
        except Exception as exc:
            cleanup_warnings.append(
                {
                    "node_rank": plan.node_rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if shutdown_refs:
        _shutdown_results, shutdown_failures = _collect_remote_results(
            ray_module,
            shutdown_refs,
            stage="shutdown",
        )
        cleanup_warnings.extend(shutdown_failures)

    for plan, actor in actors:
        try:
            ray_module.kill(actor, no_restart=True)
        except Exception as exc:
            cleanup_warnings.append(
                {
                    "node_rank": plan.node_rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return cleanup_warnings


def _distributed_summary_payload(
    *,
    batch_id: str,
    shard_plans: Sequence[RayShardPlan],
    placement_by_rank: Mapping[int, Mapping[str, Any]],
    port_preflight_by_rank: Mapping[int, Mapping[str, Any]],
    prepared_by_rank: Mapping[int, Mapping[str, Any]],
    results_by_rank: Mapping[int, Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    cleanup_warnings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one global summary without rewriting shard-relative result rows."""

    failed_ranks = {
        int(failure["node_rank"])
        for failure in failures
        if failure.get("node_rank") is not None
    }
    status_counts: dict[str, int] = {}
    terminal_result_count = 0
    shard_rows: list[dict[str, Any]] = []
    for plan in shard_plans:
        result = results_by_rank.get(plan.node_rank)
        if result is not None:
            shard_status = "completed"
            terminal_result_count += int(result.get("terminal_result_count", 0))
            for status, count in dict(result.get("status_counts") or {}).items():
                status_counts[str(status)] = status_counts.get(str(status), 0) + int(count)
        elif plan.node_rank in failed_ranks:
            shard_status = "failed"
        else:
            shard_status = "not_run"
        shard_rows.append(
            {
                **plan.to_dict(),
                "status": shard_status,
                "scheduled": plan.node_rank in placement_by_rank,
                "placement": placement_by_rank.get(plan.node_rank),
                "port_preflight": port_preflight_by_rank.get(plan.node_rank),
                "prepared": plan.node_rank in prepared_by_rank,
                "runtime": prepared_by_rank.get(plan.node_rank),
                "result": result,
            }
        )

    return {
        "schema_version": 3,
        "batch_id": batch_id,
        "status": "failed" if failures else "completed",
        "num_nodes": len(shard_plans),
        "total_episodes": sum(len(plan.episode_indices) for plan in shard_plans),
        "terminal_result_count": terminal_result_count,
        "status_counts": dict(sorted(status_counts.items())),
        "batch_results": [
            plan.to_dict()["batch_results"]
            for plan in shard_plans
            if plan.node_rank in results_by_rank
        ],
        "failures": [dict(failure) for failure in failures],
        "cleanup_warnings": [dict(warning) for warning in cleanup_warnings],
        "shards": shard_rows,
    }


def _ray_actor_dependencies(ray_module: Any) -> tuple[Any, Any]:
    """Resolve Ray-only actor APIs after the multi-node branch is selected."""

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return ray_module.remote(RewardAgentRayNodeWorker), NodeAffinitySchedulingStrategy


def _import_ray() -> Any:
    """Import Ray lazily and report the cluster image requirement clearly."""

    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "run-batch --num-nodes > 1 requires Ray in the submitted job "
            "environment (the target cluster uses Ray 2.31.0)."
        ) from exc
    return ray


def _require_unique_server_urls(server_endpoints: Sequence[Any]) -> None:
    """Reject local port collisions before launching the same layout on N nodes."""

    urls = [str(endpoint.base_url).rstrip("/") for endpoint in server_endpoints]
    duplicates = sorted({url for url in urls if urls.count(url) > 1})
    if duplicates:
        raise ValueError(
            "Ray run-batch server_groups resolve to duplicate local endpoint URLs: "
            f"{duplicates}. Use distinct port ranges across server groups."
        )


def _ray_node_sort_key(node: RayNodeSpec) -> tuple[int, int | str, str]:
    """Sort numeric IP addresses naturally and retain a deterministic fallback."""

    try:
        address = ipaddress.ip_address(node.node_ip)
    except ValueError:
        return (99, node.node_ip, node.node_id)
    return (address.version, int(address), node.node_id)


def _remaining_deadline_seconds(deadline: float) -> float:
    """Return a non-negative duration suitable for Ray and polling APIs."""

    return max(0.0, float(deadline) - time.monotonic())


def _sha256_file(path: Path) -> str:
    """Record the exact server lifecycle config shared by all workers."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one driver-owned distributed metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "DISTRIBUTION_RULE",
    "RayBatchRunSummary",
    "RayNodeSpec",
    "RayShardPlan",
    "eligible_ray_nodes",
    "run_ray_batch",
    "split_contiguous_episode_indices",
    "wait_for_ray_nodes",
]
