"""One Ray node worker for a statically assigned reward-agent shard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .batch_runner import PreparedBatchRun, RewardAgentBatchRunner
from .config import RewardAgentConfig
from .serving import (
    ManagedPortReservations,
    ManagedServerPortUnavailableError,
    ResolvedEndpoint,
    ServingManager,
    derive_client_endpoints_from_server,
    managed_port_system_context,
    remap_server_groups_to_assigned_gpus,
    reserve_managed_server_ports,
    resolve_client_endpoint_groups,
    resolve_server_groups,
    resolve_server_startup_config,
    validate_client_server_alignment,
)


class RewardAgentRayNodeWorker:
    """Own node-local vLLM processes and run one immutable episode shard."""

    def __init__(
        self,
        *,
        node_rank: int,
        expected_ray_node_id: str,
        episode_indices: list[int],
        lerobot_root: str,
        nodes_output_root: str,
        config_payload: Mapping[str, Any],
        server_payload: Mapping[str, Any],
        shard_plan: Mapping[str, Any],
        proposal_batch_results: str | None = None,
        max_concurrent_per_endpoint: int,
        print_context: bool,
    ) -> None:
        self.node_rank = int(node_rank)
        self.node_name = f"node_{self.node_rank:03d}"
        self.expected_ray_node_id = str(expected_ray_node_id)
        self.episode_indices = [int(index) for index in episode_indices]
        self.lerobot_root = Path(lerobot_root).expanduser().resolve()
        self.nodes_output_root = Path(nodes_output_root).expanduser().resolve()
        self.config_payload = dict(config_payload)
        self.server_payload = dict(server_payload)
        self.shard_plan = dict(shard_plan)
        self.proposal_batch_results = (
            None
            if proposal_batch_results is None
            else str(Path(proposal_batch_results).expanduser().resolve())
        )
        self.max_concurrent_per_endpoint = max(
            1, int(max_concurrent_per_endpoint)
        )
        self.print_context = bool(print_context)
        self._manager: ServingManager | None = None
        self._gpu_id_map: dict[int, int] | None = None
        self._preflight_server_endpoints: list[ResolvedEndpoint] | None = None
        self._preflight_gpu_id_map: dict[int, int] | None = None
        self._port_reservations: ManagedPortReservations | None = None
        self._port_preflight_payload: dict[str, Any] | None = None
        self._batch_runner: RewardAgentBatchRunner | None = None
        self._prepared_batch: PreparedBatchRun | None = None

    @property
    def node_dir(self) -> Path:
        """Return this actor's exclusive shared-storage output directory."""

        return self.nodes_output_root / self.node_name

    def check_placement(self) -> dict[str, Any]:
        """Confirm this actor owns the expected node-local GPU reservation.

        This method deliberately does not start vLLM.  The driver first waits
        for every actor to complete this lightweight barrier, so an actor that
        remains pending for GPU resources can be killed without leaving child
        server processes behind.
        """

        actual_node_id, assigned_gpu_ids = self._runtime_assignment()
        return {
            "node_rank": self.node_rank,
            "node_name": self.node_name,
            "expected_ray_node_id": self.expected_ray_node_id,
            "actual_ray_node_id": actual_node_id,
            "ray_assigned_gpu_ids": [int(gpu_id) for gpu_id in assigned_gpu_ids],
        }

    def check_server_ports(self) -> dict[str, Any]:
        """Preflight this node's managed ports without starting vLLM.

        The driver runs this method on every placed actor before calling
        :meth:`prepare`.  A port collision therefore fails in seconds instead
        of allowing the other nodes to spend many minutes compiling models
        before a global readiness barrier reports the failure.
        """

        if self._port_preflight_payload is not None:
            return dict(self._port_preflight_payload)

        actual_node_id, assigned_gpu_ids = self._runtime_assignment()
        server_endpoints, gpu_id_map = self._resolve_server_layout(
            assigned_gpu_ids
        )
        base_payload = {
            "node_rank": self.node_rank,
            "node_name": self.node_name,
            "expected_ray_node_id": self.expected_ray_node_id,
            "actual_ray_node_id": actual_node_id,
            "ray_assigned_gpu_ids": [int(gpu_id) for gpu_id in assigned_gpu_ids],
            **managed_port_system_context(),
        }
        try:
            reservations = reserve_managed_server_ports(server_endpoints)
        except ManagedServerPortUnavailableError as exc:
            # Expected bind collisions are returned as data so the driver can
            # preserve the exact endpoint and port in distributed_summary.json.
            payload = {
                **base_payload,
                "status": "failed",
                "server_port_allocations": [
                    endpoint.port_allocation_dict()
                    for endpoint in server_endpoints
                ],
                "failure": exc.to_dict(),
            }
            self._port_preflight_payload = payload
            return dict(payload)

        self._preflight_server_endpoints = server_endpoints
        self._preflight_gpu_id_map = gpu_id_map
        self._port_reservations = reservations
        payload = {
            **base_payload,
            "status": "passed",
            "server_port_allocations": [
                endpoint.port_allocation_dict() for endpoint in server_endpoints
            ],
            "failure": None,
        }
        self._port_preflight_payload = payload
        return dict(payload)

    def prepare(self) -> dict[str, Any]:
        """Map Ray GPUs, start every configured local server, and wait ready."""

        if self._manager is not None:
            raise RuntimeError(f"{self.node_name} has already been prepared.")

        actual_node_id, assigned_gpu_ids = self._runtime_assignment()
        if (
            self._preflight_server_endpoints is None
            or self._preflight_gpu_id_map is None
            or self._port_reservations is None
        ):
            raise RuntimeError(
                f"{self.node_name} must pass check_server_ports() before prepare()."
            )
        server_endpoints = self._preflight_server_endpoints
        gpu_id_map = self._preflight_gpu_id_map
        startup_config = resolve_server_startup_config(
            self.server_payload.get("startup")
        )

        config_payload = dict(self.config_payload)
        batch_payload = dict(config_payload.get("batch") or {})
        batch_payload["episode_indices"] = list(self.episode_indices)
        config_payload["batch"] = batch_payload
        shard_config = RewardAgentConfig.from_dict(config_payload)
        if shard_config.vllm.endpoint_groups:
            client_endpoints = resolve_client_endpoint_groups(
                list(shard_config.vllm.endpoint_groups)
            )
        else:
            client_endpoints = derive_client_endpoints_from_server(
                server_endpoints,
                max_concurrent=self.max_concurrent_per_endpoint,
            )
        validate_client_server_alignment(
            client_endpoints=client_endpoints,
            server_endpoints=server_endpoints,
        )

        manager = ServingManager(
            client_endpoints,
            server_endpoints=server_endpoints,
            startup_config=startup_config,
            port_reservations=self._port_reservations,
        )
        # Ownership transfers to ServingManager even if startup later fails.
        self._port_reservations = None
        batch_runner = RewardAgentBatchRunner(
            lerobot_root=self.lerobot_root,
            output_root=self.nodes_output_root,
            config=shard_config,
            serving_manager=manager,
            batch_id=self.node_name,
            proposal_batch_results=self.proposal_batch_results,
            check_server_ready=False,
            print_context=self.print_context,
        )
        try:
            prepared_batch = batch_runner.prepare()
            _write_json(
                prepared_batch.workspace.batch_dir / "shard_plan.json",
                self.shard_plan,
            )
            manager.set_log_dir(prepared_batch.workspace.servers_dir)
        except Exception:
            # Input/workspace preflight happens before Popen, but the manager
            # already owns the cross-node port reservations and must release them.
            manager.shutdown()
            raise

        try:
            manager.start_managed()
        except Exception as exc:
            try:
                _write_json(
                    self.node_dir / "server_failure.json",
                    {
                        "node_rank": self.node_rank,
                        "node_name": self.node_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "startup": startup_config.to_dict(),
                        "server_endpoints": [
                            endpoint.to_dict() for endpoint in server_endpoints
                        ],
                    },
                )
            finally:
                manager.shutdown()
            raise

        self._manager = manager
        self._gpu_id_map = gpu_id_map
        self._batch_runner = batch_runner
        self._prepared_batch = prepared_batch
        try:
            runtime_payload = {
                "node_rank": self.node_rank,
                "node_name": self.node_name,
                "expected_ray_node_id": self.expected_ray_node_id,
                "actual_ray_node_id": actual_node_id,
                "ray_assigned_gpu_ids": [int(gpu_id) for gpu_id in assigned_gpu_ids],
                "logical_to_assigned_gpu_id": {
                    str(logical): int(assigned)
                    for logical, assigned in sorted(gpu_id_map.items())
                },
                "episode_indices": list(self.episode_indices),
                "server_startup": startup_config.to_dict(),
                "server_endpoints": [
                    endpoint.to_dict() for endpoint in server_endpoints
                ],
            }
            _write_json(self.node_dir / "ray_runtime.json", runtime_payload)
            _write_json(
                self.node_dir / "server_status.json",
                {
                    "status": "ready",
                    "startup": startup_config.to_dict(),
                    "server_endpoints": [
                        endpoint.to_dict() for endpoint in server_endpoints
                    ],
                },
            )
        except Exception:
            # Once the manager is attached to the actor, any metadata failure
            # must still stop the already-ready node-local server processes.
            self._shutdown_manager()
            raise
        return runtime_payload

    def run_shard(self) -> dict[str, Any]:
        """Run the existing node-local batch scheduler for this fixed shard."""

        if (
            self._manager is None
            or self._batch_runner is None
            or self._prepared_batch is None
        ):
            raise RuntimeError(f"{self.node_name} must be prepared before run_shard().")

        try:
            batch = self._batch_runner.run(self._prepared_batch)
            shard_summary = json.loads(
                batch.batch_summary.read_text(encoding="utf-8")
            )
            return {
                "node_rank": self.node_rank,
                "node_name": self.node_name,
                "status": "completed",
                "batch_dir": str(batch.batch_dir),
                "batch_results": str(batch.batch_results),
                "batch_events": str(batch.batch_events),
                "batch_summary": str(batch.batch_summary),
                "serving_resolved": str(batch.serving_resolved),
                "total_episodes": int(batch.total_episodes),
                "terminal_result_count": int(
                    shard_summary.get("terminal_result_count", 0)
                ),
                "status_counts": dict(shard_summary.get("status_counts") or {}),
            }
        finally:
            self._shutdown_manager()

    def shutdown(self) -> dict[str, Any]:
        """Best-effort explicit cleanup used by the driver on every exit path."""

        self._shutdown_manager()
        return {
            "node_rank": self.node_rank,
            "node_name": self.node_name,
            "shutdown": True,
        }

    def _shutdown_manager(self) -> None:
        """Release reservations and local server processes exactly once."""

        reservations = self._port_reservations
        self._port_reservations = None
        if reservations is not None:
            reservations.close()
        manager = self._manager
        self._manager = None
        self._batch_runner = None
        self._prepared_batch = None
        if manager is not None:
            manager.shutdown()

    def _runtime_assignment(self) -> tuple[str, list[Any]]:
        """Return and validate the Ray node/GPU assignment for this actor."""

        # Ray remains optional for the normal local run-batch path because it
        # is imported only inside methods executed by a Ray actor.
        import ray

        runtime_context = ray.get_runtime_context()
        actual_node_id = str(runtime_context.get_node_id())
        if actual_node_id != self.expected_ray_node_id:
            raise RuntimeError(
                f"{self.node_name} was scheduled on Ray node {actual_node_id}, "
                f"expected {self.expected_ray_node_id}."
            )
        return actual_node_id, list(ray.get_gpu_ids())

    def _resolve_server_layout(
        self,
        assigned_gpu_ids: list[Any],
    ) -> tuple[list[Any], dict[int, int]]:
        """Resolve logical server slots onto this actor's physical GPUs."""

        remapped_groups, gpu_id_map = remap_server_groups_to_assigned_gpus(
            self.server_payload.get("server_groups", []),
            assigned_gpu_ids=assigned_gpu_ids,
        )
        return resolve_server_groups(remapped_groups), gpu_id_map


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one node-owned metadata file with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = ["RewardAgentRayNodeWorker"]
