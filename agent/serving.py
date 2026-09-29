"""Endpoint-group resolution and lifecycle helpers for reward-agent batches."""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml


_RESERVED_MANAGED_ENV_KEYS = frozenset({"CUDA_VISIBLE_DEVICES", "VLLM_PORT"})
_TCP_STATE_NAMES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
    "0C": "NEW_SYN_RECV",
}


@dataclass(frozen=True, slots=True)
class ServerStartupConfig:
    """Node-local policy for starting and stopping managed vLLM replicas."""

    max_parallel_per_node: int
    all_ready_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_grace_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResolvedEndpoint:
    """One concrete OpenAI-compatible endpoint usable by the batch scheduler."""

    group_name: str
    mode: str
    replica_index: int
    base_url: str
    max_concurrent: int
    gpu_id: int | None = None
    port: int | None = None
    vllm_port_block_start: int | None = None
    vllm_port_block_end: int | None = None
    selected_vllm_port: int | None = None
    vllm_port_selection: str | None = None
    rejected_vllm_ports: list[dict[str, Any]] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    pid: int | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def port_allocation_dict(self) -> dict[str, Any]:
        """Return the node-local fields relevant to managed port allocation."""

        return {
            "group_name": self.group_name,
            "replica_index": self.replica_index,
            "gpu_id": self.gpu_id,
            "http_port": self.port,
            "vllm_port_block_start": self.vllm_port_block_start,
            "vllm_port_block_end": self.vllm_port_block_end,
            "selected_vllm_port": self.selected_vllm_port,
            "selection": self.vllm_port_selection,
            "rejected_candidates": copy.deepcopy(self.rejected_vllm_ports),
        }


class ManagedServerPortUnavailableError(RuntimeError):
    """Report an exhausted fixed port or replica-local vLLM candidate block."""

    def __init__(
        self,
        *,
        endpoint: ResolvedEndpoint,
        port_kind: str,
        attempted_ports: Sequence[int],
        conflicts: Sequence[dict[str, Any]],
    ) -> None:
        self.port_kind = str(port_kind)
        self.attempted_ports = [int(port) for port in attempted_ports]
        self.conflicts = [dict(conflict) for conflict in conflicts]
        self.endpoint = endpoint.port_allocation_dict()
        if self.port_kind == "http":
            message = (
                f"Managed server HTTP port {self.attempted_ports[0]} for endpoint "
                f"{_endpoint_label(endpoint)} is unavailable on this node."
            )
        else:
            message = (
                "Managed server vLLM candidate block "
                f"{endpoint.vllm_port_block_start}-{endpoint.vllm_port_block_end} "
                f"for endpoint {_endpoint_label(endpoint)} has no available port."
            )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Return stable fields suitable for a distributed failure summary."""

        return {
            "error_type": type(self).__name__,
            "error": str(self),
            "port_kind": self.port_kind,
            "attempted_ports": list(self.attempted_ports),
            "endpoint": dict(self.endpoint),
            "conflicts": copy.deepcopy(self.conflicts),
        }


@dataclass(slots=True)
class ManagedPortReservations:
    """Hold successful preflight binds until their endpoint begins startup."""

    _sockets_by_endpoint: dict[int, list[Any]] = field(default_factory=dict)

    def add(self, endpoint: ResolvedEndpoint, probe: Any) -> None:
        self._sockets_by_endpoint.setdefault(id(endpoint), []).append(probe)

    def release_endpoint(self, endpoint: ResolvedEndpoint) -> None:
        """Release only the sockets needed by the endpoint about to start."""

        for probe in self._sockets_by_endpoint.pop(id(endpoint), []):
            probe.close()

    def close(self) -> None:
        """Release every unconsumed reservation after failure or cancellation."""

        for probes in self._sockets_by_endpoint.values():
            for probe in probes:
                probe.close()
        self._sockets_by_endpoint.clear()


@dataclass(slots=True)
class _ManagedProcess:
    """Runtime handles needed to monitor and clean up one server process."""

    endpoint: ResolvedEndpoint
    process: subprocess.Popen[bytes]
    log_handle: Any | None = None


class ServingManager:
    """Own managed vLLM processes while leaving external endpoints alone."""

    def __init__(
        self,
        endpoints: list[ResolvedEndpoint],
        *,
        server_endpoints: list[ResolvedEndpoint] | None = None,
        startup_config: ServerStartupConfig | None = None,
        log_dir: str | Path | None = None,
        port_reservations: ManagedPortReservations | None = None,
    ) -> None:
        self.endpoints = endpoints
        self.server_endpoints = list(server_endpoints or [])
        self.startup_config = startup_config
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir is not None else None
        self._port_reservations = port_reservations
        self._processes: list[_ManagedProcess] = []
        self._pending_log_handles: dict[int, Any] = {}
        self._managed_start_attempted = False

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "ServingManager":
        return cls(resolve_client_endpoint_groups(payload.get("endpoint_groups", [])))

    def set_log_dir(self, log_dir: str | Path) -> None:
        """Bind managed logs to the workspace reserved by the batch runner."""

        if self._managed_start_attempted:
            raise RuntimeError("Cannot change managed log directory after startup begins.")
        self.log_dir = Path(log_dir).expanduser().resolve()

    def start_managed(self) -> None:
        """Start managed endpoints through a bounded rolling readiness window."""

        if not self.server_endpoints:
            return
        if self.startup_config is None:
            raise ValueError("Managed server startup requires a top-level startup config.")
        if self._managed_start_attempted:
            raise RuntimeError("Managed servers have already been started by this manager.")
        self._managed_start_attempted = True

        try:
            if self._port_reservations is None:
                self._port_reservations = reserve_managed_server_ports(
                    self.server_endpoints
                )
            _validate_selected_vllm_ports(self.server_endpoints)
            self._validate_managed_endpoints_for_startup()
            self._prepare_managed_logs()

            queued = list(self.server_endpoints)
            starting: list[_ManagedProcess] = []
            deadline = time.monotonic() + self.startup_config.all_ready_timeout_seconds
            while queued or starting:
                while queued and len(starting) < self.startup_config.max_parallel_per_node:
                    starting.append(self._start_endpoint(queued.pop(0)))

                ready_records: list[_ManagedProcess] = []
                for record in starting:
                    return_code = record.process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            _managed_process_exit_message(
                                record.endpoint,
                                return_code=int(return_code),
                            )
                        )
                    if _endpoint_ready(record.endpoint.base_url):
                        ready_records.append(record)
                        print(
                            "[reward-agent serving] ready "
                            f"endpoint={_endpoint_label(record.endpoint)} "
                            f"pid={record.process.pid}",
                            flush=True,
                        )

                for record in ready_records:
                    starting.remove(record)
                if not queued and not starting:
                    return
                if time.monotonic() >= deadline:
                    pending = [
                        _endpoint_label(record.endpoint) for record in starting
                    ]
                    pending.extend(_endpoint_label(endpoint) for endpoint in queued)
                    raise TimeoutError(
                        "Timed out waiting for managed endpoints: "
                        f"{sorted(pending)}"
                    )
                if ready_records:
                    # Refill newly available startup slots immediately. The
                    # polling delay is only needed when no state changed.
                    continue
                time.sleep(self.startup_config.poll_interval_seconds)
        except Exception:
            self.shutdown()
            raise

    def _validate_managed_endpoints_for_startup(self) -> None:
        """Reject incomplete endpoint records before creating logs or processes."""

        for endpoint in self.server_endpoints:
            if not endpoint.argv:
                raise ValueError(
                    f"Managed endpoint {_endpoint_label(endpoint)} has empty argv."
                )
            if endpoint.selected_vllm_port is None:
                raise ValueError(
                    f"Managed endpoint {_endpoint_label(endpoint)} has no selected "
                    "vLLM port."
                )

    def _prepare_managed_logs(self) -> None:
        """Exclusively create every replica log before the first subprocess."""

        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=False, exist_ok=False)
        try:
            for endpoint in self.server_endpoints:
                log_path = self.log_dir / _endpoint_log_name(endpoint)
                endpoint.log_path = str(log_path)
                self._pending_log_handles[id(endpoint)] = log_path.open("xb")
        except Exception:
            self._close_pending_log_handles()
            raise

    def _start_endpoint(self, endpoint: ResolvedEndpoint) -> _ManagedProcess:
        """Start one isolated process group and retain its optional log handle."""

        if not endpoint.argv:
            raise ValueError(
                f"Managed endpoint {_endpoint_label(endpoint)} has empty argv."
            )
        if endpoint.selected_vllm_port is None:
            raise ValueError(
                f"Managed endpoint {_endpoint_label(endpoint)} has no selected "
                "vLLM port."
            )
        endpoint.env["VLLM_PORT"] = str(endpoint.selected_vllm_port)
        if self._port_reservations is not None:
            # The subprocess cannot inherit a pre-bound TCPStore socket, so
            # release this endpoint's reservation immediately before Popen.
            self._port_reservations.release_endpoint(endpoint)
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in endpoint.env.items()})
        log_handle = self._pending_log_handles.pop(id(endpoint), None)
        popen_kwargs: dict[str, Any] = {
            "env": env,
            "start_new_session": True,
        }
        if self.log_dir is not None:
            if log_handle is None:
                raise RuntimeError(
                    "Managed endpoint log was not reserved before subprocess startup: "
                    f"{_endpoint_label(endpoint)}."
                )
            popen_kwargs["stdout"] = log_handle
            popen_kwargs["stderr"] = subprocess.STDOUT
        try:
            process = subprocess.Popen(endpoint.argv, **popen_kwargs)
        except Exception:
            if log_handle is not None:
                log_handle.close()
            raise
        endpoint.pid = int(process.pid)
        record = _ManagedProcess(
            endpoint=endpoint,
            process=process,
            log_handle=log_handle,
        )
        self._processes.append(record)
        print(
            "[reward-agent serving] starting "
            f"endpoint={_endpoint_label(endpoint)} pid={process.pid}",
            flush=True,
        )
        return record

    def wait_ready(self, *, timeout_seconds: float = 3600.0, poll_interval_seconds: float = 2.0) -> None:
        """Wait until every resolved endpoint responds to /models."""

        deadline = time.monotonic() + float(timeout_seconds)
        pending = {endpoint.base_url.rstrip("/") for endpoint in self.endpoints}
        while pending:
            ready_now: set[str] = set()
            for base_url in sorted(pending):
                if _endpoint_ready(base_url):
                    ready_now.add(base_url)
            pending -= ready_now
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for endpoints: {sorted(pending)}")
            time.sleep(float(poll_interval_seconds))

    def shutdown(self) -> None:
        """Terminate every process group started by this manager."""

        if self._port_reservations is not None:
            self._port_reservations.close()
            self._port_reservations = None
        self._close_pending_log_handles()
        if not self._processes:
            return
        grace_seconds = (
            self.startup_config.shutdown_grace_seconds
            if self.startup_config is not None
            else 30.0
        )
        records = list(reversed(self._processes))
        for record in records:
            _signal_process_group(record, signal.SIGTERM)

        deadline = time.monotonic() + grace_seconds
        for record in records:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                record.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        for record in records:
            if record.process.poll() is None:
                _signal_process_group(record, signal.SIGKILL)
                try:
                    record.process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    # The process handle is retained for diagnostics.  There is
                    # no broader process target that is safe to signal here.
                    pass
            if record.log_handle is not None and not record.log_handle.closed:
                record.log_handle.close()
        self._processes.clear()

    def _close_pending_log_handles(self) -> None:
        """Close logs reserved for endpoints that have not reached Popen."""

        for handle in self._pending_log_handles.values():
            if not handle.closed:
                handle.close()
        self._pending_log_handles.clear()

    def write_resolved(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "startup": self.startup_config.to_dict()
                    if self.startup_config is not None
                    else None,
                    "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
                    "server_endpoints": [endpoint.to_dict() for endpoint in self.server_endpoints],
                },
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        return output_path


def load_server_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML server lifecycle config mapping."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return payload


def resolve_server_startup_config(raw_startup: Any) -> ServerStartupConfig:
    """Validate the required node-local managed-server startup policy."""

    if not isinstance(raw_startup, dict):
        raise TypeError("server.yaml startup must be a mapping.")
    required = {
        "max_parallel_per_node",
        "all_ready_timeout_seconds",
        "poll_interval_seconds",
        "shutdown_grace_seconds",
    }
    missing = sorted(required - set(raw_startup))
    if missing:
        raise ValueError(f"server.yaml startup is missing required keys: {missing}.")
    return ServerStartupConfig(
        max_parallel_per_node=_normalize_positive_int(
            raw_startup["max_parallel_per_node"],
            field_name="startup.max_parallel_per_node",
        ),
        all_ready_timeout_seconds=_normalize_positive_float(
            raw_startup["all_ready_timeout_seconds"],
            field_name="startup.all_ready_timeout_seconds",
        ),
        poll_interval_seconds=_normalize_positive_float(
            raw_startup["poll_interval_seconds"],
            field_name="startup.poll_interval_seconds",
        ),
        shutdown_grace_seconds=_normalize_positive_float(
            raw_startup["shutdown_grace_seconds"],
            field_name="startup.shutdown_grace_seconds",
        ),
    )


def resolve_distributed_server_gpu_ids(raw_groups: Any) -> tuple[int, ...]:
    """Return globally unique logical GPU slots for one Ray node.

    The existing server schema starts one managed server process for every
    ``server_groups[*].gpu_ids`` entry.  Sharing one ID across groups would
    therefore launch multiple full vLLM processes on the same device.  Ray
    batches reject that ambiguous/high-risk layout and reserve one Ray GPU for
    every configured logical slot.
    """

    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("server_groups must be a non-empty list.")

    logical_gpu_ids: list[int] = []
    owner_by_gpu_id: dict[int, str] = {}
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise TypeError(f"server_groups[{group_index}] must be a mapping.")
        group_name = str(raw_group.get("name") or f"group_{group_index}").strip()
        raw_gpu_ids = raw_group.get("gpu_ids")
        if not isinstance(raw_gpu_ids, list) or not raw_gpu_ids:
            raise ValueError(f"server group {group_name!r} must set gpu_ids.")
        for gpu_position, raw_gpu_id in enumerate(raw_gpu_ids):
            gpu_id = _normalize_gpu_id(
                raw_gpu_id,
                field_name=(
                    f"server_groups[{group_index}].gpu_ids[{gpu_position}]"
                ),
            )
            previous_owner = owner_by_gpu_id.get(gpu_id)
            if previous_owner is not None:
                raise ValueError(
                    "Ray run-batch requires server_groups[*].gpu_ids to be "
                    f"globally unique; GPU ID {gpu_id} appears in both "
                    f"{previous_owner!r} and {group_name!r}."
                )
            owner_by_gpu_id[gpu_id] = group_name
            logical_gpu_ids.append(gpu_id)

    return tuple(logical_gpu_ids)


def remap_server_groups_to_assigned_gpus(
    raw_groups: Any,
    *,
    assigned_gpu_ids: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Map server.yaml logical slots onto GPUs reserved by a Ray actor.

    Ray decides which physical devices satisfy ``num_gpus``.  A child vLLM
    process must use those exact devices rather than assuming that logical
    slots such as 0 and 1 are also the allocated physical IDs.
    """

    logical_gpu_ids = resolve_distributed_server_gpu_ids(raw_groups)
    normalized_assigned = tuple(
        _normalize_gpu_id(raw_gpu_id, field_name=f"Ray assigned_gpu_ids[{index}]")
        for index, raw_gpu_id in enumerate(assigned_gpu_ids)
    )
    if len(normalized_assigned) != len(logical_gpu_ids):
        raise ValueError(
            "Ray assigned GPU count does not match server.yaml: "
            f"assigned={list(normalized_assigned)}, "
            f"configured_logical_slots={list(logical_gpu_ids)}."
        )
    if len(set(normalized_assigned)) != len(normalized_assigned):
        raise ValueError(
            f"Ray assigned duplicate GPU IDs: {list(normalized_assigned)}."
        )

    gpu_id_map = dict(zip(logical_gpu_ids, normalized_assigned, strict=True))
    remapped_groups: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(raw_groups):
        group = copy.deepcopy(raw_group)
        raw_env = group.get("env") or {}
        if not isinstance(raw_env, dict):
            raise TypeError(f"server_groups[{group_index}].env must be a mapping.")
        if "CUDA_VISIBLE_DEVICES" in raw_env:
            raise ValueError(
                "Ray run-batch derives CUDA_VISIBLE_DEVICES from Ray GPU "
                f"reservations; remove it from server_groups[{group_index}].env."
            )
        group["gpu_ids"] = [gpu_id_map[int(gpu_id)] for gpu_id in group["gpu_ids"]]
        remapped_groups.append(group)
    return remapped_groups, gpu_id_map


def resolve_client_endpoint_groups(raw_groups: Any) -> list[ResolvedEndpoint]:
    """Expand client-side vLLM endpoint groups into concrete endpoints."""

    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("endpoint_groups must be a non-empty list.")
    endpoints: list[ResolvedEndpoint] = []
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise TypeError(f"endpoint_groups[{group_index}] must be a mapping.")
        name = str(raw_group.get("name") or f"group_{group_index}").strip()
        max_concurrent = int(raw_group.get("max_concurrent", 1))
        if max_concurrent <= 0:
            raise ValueError(f"endpoint_groups[{group_index}].max_concurrent must be positive.")
        if raw_group.get("endpoints") is not None:
            endpoints.extend(_resolve_explicit_client_group(raw_group, name=name, max_concurrent=max_concurrent))
        elif raw_group.get("ports") is not None:
            endpoints.extend(_resolve_port_template_client_group(raw_group, name=name, max_concurrent=max_concurrent))
        else:
            raise ValueError(
                f"endpoint_groups[{group_index}] must set either endpoints or ports."
            )
    return endpoints


def resolve_server_groups(raw_groups: Any) -> list[ResolvedEndpoint]:
    """Expand server-side vLLM process groups into managed endpoints."""

    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("server_groups must be a non-empty list.")
    endpoints: list[ResolvedEndpoint] = []
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise TypeError(f"server_groups[{group_index}] must be a mapping.")
        name = str(raw_group.get("name") or f"group_{group_index}").strip()
        endpoints.extend(_resolve_server_group(raw_group, name=name))
    _validate_managed_endpoint_ports(endpoints)
    return endpoints


def derive_client_endpoints_from_server(
    server_endpoints: list[ResolvedEndpoint],
    *,
    max_concurrent: int = 1,
) -> list[ResolvedEndpoint]:
    """Use resolved server URLs as client endpoints when config omits them."""

    return [
        ResolvedEndpoint(
            group_name=endpoint.group_name,
            mode="external",
            replica_index=endpoint.replica_index,
            base_url=endpoint.base_url,
            max_concurrent=max(1, int(max_concurrent)),
            gpu_id=endpoint.gpu_id,
            port=endpoint.port,
        )
        for endpoint in server_endpoints
    ]


def validate_client_server_alignment(
    *,
    client_endpoints: list[ResolvedEndpoint],
    server_endpoints: list[ResolvedEndpoint],
) -> None:
    """Require exact client/server URL alignment for managed serving."""

    client_url_list = [endpoint.base_url.rstrip("/") for endpoint in client_endpoints]
    server_url_list = [endpoint.base_url.rstrip("/") for endpoint in server_endpoints]
    duplicate_clients = sorted(
        {url for url in client_url_list if client_url_list.count(url) > 1}
    )
    if duplicate_clients:
        raise ValueError(
            f"Client endpoint_groups contain duplicate URL(s): {duplicate_clients}."
        )
    client_urls = set(client_url_list)
    server_urls = set(server_url_list)
    missing_servers = sorted(client_urls - server_urls)
    unused_servers = sorted(server_urls - client_urls)
    if missing_servers or unused_servers:
        raise ValueError(
            "Managed client/server endpoint URLs must match exactly: "
            f"client_urls_without_server={missing_servers}, "
            f"server_urls_without_client={unused_servers}."
        )


def _resolve_explicit_client_group(
    raw_group: dict[str, Any],
    *,
    name: str,
    max_concurrent: int,
) -> list[ResolvedEndpoint]:
    raw_endpoints = raw_group.get("endpoints")
    if not isinstance(raw_endpoints, list) or not raw_endpoints:
        raise ValueError(f"client endpoint group {name!r} must set a non-empty endpoints list.")
    endpoints: list[ResolvedEndpoint] = []
    for index, raw_endpoint in enumerate(raw_endpoints):
        if isinstance(raw_endpoint, dict):
            base_url = str(raw_endpoint.get("base_url") or "").strip()
            endpoint_concurrency = int(raw_endpoint.get("max_concurrent", max_concurrent))
        else:
            base_url = str(raw_endpoint).strip()
            endpoint_concurrency = max_concurrent
        if not base_url:
            raise ValueError(f"client endpoint group {name!r} endpoint {index} is missing base_url.")
        endpoints.append(
            ResolvedEndpoint(
                group_name=name,
                mode="external",
                replica_index=index,
                base_url=base_url.rstrip("/"),
                max_concurrent=endpoint_concurrency,
            )
        )
    return endpoints


def _resolve_port_template_client_group(
    raw_group: dict[str, Any],
    *,
    name: str,
    max_concurrent: int,
) -> list[ResolvedEndpoint]:
    ports = [
        _normalize_tcp_port(
            item,
            field_name=f"client endpoint group {name!r} ports[{index}]",
        )
        for index, item in enumerate(raw_group.get("ports", []))
    ]
    if not ports:
        raise ValueError(f"client endpoint group {name!r} must set a non-empty ports list.")
    base_url_template = str(raw_group.get("base_url_template", "http://127.0.0.1:{port}/v1"))
    return [
        ResolvedEndpoint(
            group_name=name,
            mode="external",
            replica_index=index,
            port=port,
            base_url=base_url_template.format(
                group_name=name,
                replica_index=index,
                port=port,
            ).rstrip("/"),
            max_concurrent=max_concurrent,
        )
        for index, port in enumerate(ports)
    ]


def _resolve_server_group(raw_group: dict[str, Any], *, name: str) -> list[ResolvedEndpoint]:
    gpu_ids = [
        _normalize_gpu_id(item, field_name=f"server group {name!r} gpu_ids[{index}]")
        for index, item in enumerate(raw_group.get("gpu_ids", []))
    ]
    if not gpu_ids:
        raise ValueError(f"server group {name!r} must set gpu_ids.")
    for required_key in (
        "http_port_start",
        "vllm_port_base",
        "vllm_port_block_size",
    ):
        if required_key not in raw_group:
            raise ValueError(
                f"server group {name!r} must set {required_key}."
            )
    http_port_start = _normalize_tcp_port(
        raw_group["http_port_start"],
        field_name=f"server group {name!r} http_port_start",
    )
    vllm_port_base = _normalize_tcp_port(
        raw_group["vllm_port_base"],
        field_name=f"server group {name!r} vllm_port_base",
    )
    vllm_port_block_size = _normalize_positive_int(
        raw_group["vllm_port_block_size"],
        field_name=f"server group {name!r} vllm_port_block_size",
    )
    raw_env = raw_group.get("env") or {}
    if not isinstance(raw_env, dict):
        raise TypeError(f"server group {name!r} env must be a mapping.")
    reserved_env_keys = sorted(_RESERVED_MANAGED_ENV_KEYS & set(raw_env))
    if reserved_env_keys:
        raise ValueError(
            f"server group {name!r} env contains manager-owned key(s) "
            f"{reserved_env_keys}; configure GPU and internal ports through "
            "gpu_ids/vllm_port_base/vllm_port_block_size."
        )

    endpoints: list[ResolvedEndpoint] = []
    for replica_index, gpu_id in enumerate(gpu_ids):
        port = _normalize_tcp_port(
            http_port_start + replica_index,
            field_name=f"server group {name!r} replica {replica_index} HTTP port",
        )
        replica_vllm_port_block_start = _normalize_tcp_port(
            vllm_port_base + replica_index * vllm_port_block_size,
            field_name=(
                f"server group {name!r} replica {replica_index} vLLM block start"
            ),
        )
        replica_vllm_port_block_end = _normalize_tcp_port(
            replica_vllm_port_block_start + vllm_port_block_size - 1,
            field_name=(
                f"server group {name!r} replica {replica_index} vLLM block end"
            ),
        )
        values = {
            "group_name": name,
            "replica_index": replica_index,
            "gpu_id": gpu_id,
            "port": port,
            "http_port": port,
            "vllm_port_block_start": replica_vllm_port_block_start,
            "vllm_port_block_end": replica_vllm_port_block_end,
            "vllm_port_block_size": vllm_port_block_size,
        }
        env = {
            str(key): _format_known_placeholders(str(value), values)
            for key, value in raw_env.items()
        }
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        argv = _resolve_server_argv(raw_group, group_name=name, values=values)
        endpoints.append(
            ResolvedEndpoint(
                group_name=name,
                mode="managed",
                replica_index=replica_index,
                gpu_id=gpu_id,
                port=port,
                vllm_port_block_start=replica_vllm_port_block_start,
                vllm_port_block_end=replica_vllm_port_block_end,
                base_url=f"http://127.0.0.1:{port}/v1",
                max_concurrent=1,
                argv=argv,
                env=env,
            )
        )
    return endpoints


def _resolve_server_argv(
    raw_group: dict[str, Any],
    *,
    group_name: str,
    values: dict[str, Any],
) -> list[str]:
    """Resolve either compact kind/model_path/args config or raw argv."""

    if raw_group.get("argv") is not None:
        argv_template = raw_group.get("argv")
        if not isinstance(argv_template, list) or not argv_template:
            raise ValueError(f"server group {group_name!r} argv must be a non-empty list.")
        return [_format_known_placeholders(str(item), values) for item in argv_template]

    kind = str(raw_group.get("kind", "vllm")).strip().lower()
    if kind != "vllm":
        raise ValueError(
            f"server group {group_name!r} kind must be 'vllm' when argv is omitted, got {kind!r}."
        )
    model_path = raw_group.get("model_path")
    if model_path in (None, ""):
        raise ValueError(f"server group {group_name!r} must set model_path when argv is omitted.")

    argv = ["vllm", "serve", _format_known_placeholders(str(model_path), values)]
    args = raw_group.get("args", {}) or {}
    if not isinstance(args, dict):
        raise TypeError(f"server group {group_name!r} args must be a mapping.")
    argv.extend(_server_args_to_argv(args, values=values))
    return argv


def _server_args_to_argv(args: dict[str, Any], *, values: dict[str, Any]) -> list[str]:
    """Expand a YAML mapping of CLI arguments into subprocess argv tokens."""

    argv: list[str] = []
    for raw_key, raw_value in args.items():
        key = str(raw_key)
        if not key.startswith("--"):
            key = f"--{key}"
        if raw_value is None or raw_value is False:
            continue
        argv.append(key)
        if raw_value is True:
            continue
        argv.append(_format_server_arg_value(raw_value, values=values))
    return argv


def _format_server_arg_value(value: Any, *, values: dict[str, Any]) -> str:
    """Convert scalar or structured YAML arg values into one argv string."""

    if isinstance(value, str):
        return _format_known_placeholders(value, values)
    if isinstance(value, (dict, list)):
        return _format_known_placeholders(
            json.dumps(value, ensure_ascii=False),
            values,
        )
    return _format_known_placeholders(str(value), values)


def _format_known_placeholders(value: str, values: dict[str, Any]) -> str:
    """Replace only reward-agent placeholders so JSON argv strings stay intact."""

    formatted = str(value)
    for key, replacement in values.items():
        formatted = formatted.replace("{" + key + "}", str(replacement))
    return formatted


def _normalize_gpu_id(value: Any, *, field_name: str) -> int:
    """Normalize integer-like GPU IDs without accepting booleans/fractions."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a non-negative integer, not bool.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must be a non-negative integer, got {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(
            f"{field_name} must be a non-negative integer, got {value!r}."
        )
    if normalized < 0:
        raise ValueError(
            f"{field_name} must be a non-negative integer, got {value!r}."
        )
    return normalized


def _normalize_positive_int(value: Any, *, field_name: str) -> int:
    """Normalize a strictly positive integer without accepting booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a positive integer, not bool.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must be a positive integer, got {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(
            f"{field_name} must be a positive integer, got {value!r}."
        )
    if normalized <= 0:
        raise ValueError(
            f"{field_name} must be a positive integer, got {value!r}."
        )
    return normalized


def _normalize_positive_float(value: Any, *, field_name: str) -> float:
    """Normalize a finite strictly positive duration."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be positive, not bool.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be positive, got {value!r}.") from exc
    if normalized <= 0.0 or normalized == float("inf") or normalized != normalized:
        raise ValueError(f"{field_name} must be finite and positive, got {value!r}.")
    return normalized


def _normalize_tcp_port(value: Any, *, field_name: str) -> int:
    """Normalize one TCP port and reject values outside the usable range."""

    port = _normalize_positive_int(value, field_name=field_name)
    if port > 65535:
        raise ValueError(f"{field_name} must be at most 65535, got {port}.")
    return port


def _validate_managed_endpoint_ports(endpoints: Sequence[ResolvedEndpoint]) -> None:
    """Reject overlapping HTTP ports and replica-local vLLM candidate blocks."""

    http_owners: dict[int, str] = {}
    blocks: list[tuple[int, int, str]] = []
    for endpoint in endpoints:
        label = _endpoint_label(endpoint)
        if endpoint.port is None:
            raise ValueError(f"Managed endpoint {label} is missing its HTTP port.")
        previous_http_owner = http_owners.get(endpoint.port)
        if previous_http_owner is not None:
            raise ValueError(
                f"Managed HTTP port {endpoint.port} is used by both "
                f"{previous_http_owner} and {label}."
            )
        http_owners[endpoint.port] = label
        if (
            endpoint.vllm_port_block_start is None
            or endpoint.vllm_port_block_end is None
        ):
            raise ValueError(f"Managed endpoint {label} is missing its vLLM port block.")
        blocks.append(
            (
                endpoint.vllm_port_block_start,
                endpoint.vllm_port_block_end,
                label,
            )
        )

    sorted_blocks = sorted(blocks)
    for previous, current in zip(sorted_blocks, sorted_blocks[1:]):
        if current[0] <= previous[1]:
            raise ValueError(
                "Managed vLLM internal port blocks overlap: "
                f"{previous[2]} uses {previous[0]}-{previous[1]}, "
                f"{current[2]} uses {current[0]}-{current[1]}."
            )
    for http_port, http_owner in http_owners.items():
        for block_start, block_end, block_owner in blocks:
            if block_start <= http_port <= block_end:
                raise ValueError(
                    f"Managed HTTP port {http_port} for {http_owner} overlaps "
                    f"the vLLM internal port block {block_start}-{block_end} "
                    f"for {block_owner}."
                )


def reserve_managed_server_ports(
    endpoints: Sequence[ResolvedEndpoint],
) -> ManagedPortReservations:
    """Reserve fixed HTTP ports and the first free TP1 port in each block.

    The returned sockets keep the successful preflight choices unavailable to
    other local consumers during the Ray all-node barrier.  Each socket is
    released immediately before its corresponding vLLM subprocess starts.
    """

    reservations = ManagedPortReservations()
    for endpoint in endpoints:
        endpoint.selected_vllm_port = None
        endpoint.vllm_port_selection = None
        endpoint.rejected_vllm_ports.clear()
        endpoint.env.pop("VLLM_PORT", None)

    try:
        # HTTP ports define client-visible URLs and therefore cannot fall back.
        for endpoint in endpoints:
            if endpoint.port is None:
                raise ValueError(
                    f"Managed endpoint {_endpoint_label(endpoint)} has no HTTP port."
                )
            probe, conflict = _try_reserve_tcp_port(endpoint.port)
            if probe is None:
                raise ManagedServerPortUnavailableError(
                    endpoint=endpoint,
                    port_kind="http",
                    attempted_ports=[endpoint.port],
                    conflicts=[conflict],
                )
            reservations.add(endpoint, probe)

        # TP1 needs one TCPStore rendezvous port.  The replica-local block is a
        # deterministic fallback pool, not a requirement that every candidate
        # be simultaneously free.
        for endpoint in endpoints:
            block_start = endpoint.vllm_port_block_start
            block_end = endpoint.vllm_port_block_end
            if block_start is None or block_end is None:
                raise ValueError(
                    f"Managed endpoint {_endpoint_label(endpoint)} has no vLLM block."
                )
            attempted_ports: list[int] = []
            for candidate in range(block_start, block_end + 1):
                attempted_ports.append(candidate)
                probe, conflict = _try_reserve_tcp_port(candidate)
                if probe is None:
                    endpoint.rejected_vllm_ports.append(conflict)
                    continue
                endpoint.selected_vllm_port = candidate
                endpoint.vllm_port_selection = (
                    "preferred" if candidate == block_start else "fallback"
                )
                endpoint.env["VLLM_PORT"] = str(candidate)
                reservations.add(endpoint, probe)
                break
            if endpoint.selected_vllm_port is None:
                raise ManagedServerPortUnavailableError(
                    endpoint=endpoint,
                    port_kind="vllm_block",
                    attempted_ports=attempted_ports,
                    conflicts=endpoint.rejected_vllm_ports,
                )
        return reservations
    except Exception:
        reservations.close()
        for endpoint in endpoints:
            endpoint.selected_vllm_port = None
            endpoint.vllm_port_selection = None
            endpoint.env.pop("VLLM_PORT", None)
        raise


def managed_port_system_context() -> dict[str, Any]:
    """Capture network-namespace settings available without shell access."""

    return {
        "ephemeral_port_range": _read_proc_int_pair(
            Path("/proc/sys/net/ipv4/ip_local_port_range")
        ),
        "reserved_port_ranges": _read_proc_text(
            Path("/proc/sys/net/ipv4/ip_local_reserved_ports")
        ),
    }


def _try_reserve_tcp_port(port: int) -> tuple[Any | None, dict[str, Any]]:
    """Bind one reusable probe socket and retain it on success."""

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # TIME_WAIT should not by itself reject a reusable listener candidate.
        # SO_REUSEPORT remains intentionally disabled because a live listener
        # must still make the port unavailable.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", port))
        # listen() turns the bound socket into an exclusive reservation. A
        # merely bound SO_REUSEADDR socket could coexist with another probe
        # until one side eventually attempted to listen.
        probe.listen(1)
    except OSError as exc:
        probe.close()
        return None, _port_conflict_payload(port, exc)
    return probe, {}


def _validate_selected_vllm_ports(endpoints: Sequence[ResolvedEndpoint]) -> None:
    """Require every selected runtime port to remain inside its owned block."""

    for endpoint in endpoints:
        selected = endpoint.selected_vllm_port
        block_start = endpoint.vllm_port_block_start
        block_end = endpoint.vllm_port_block_end
        if selected is None or block_start is None or block_end is None:
            raise ValueError(
                f"Managed endpoint {_endpoint_label(endpoint)} has no preflight "
                "vLLM port selection."
            )
        if not block_start <= selected <= block_end:
            raise ValueError(
                f"Managed endpoint {_endpoint_label(endpoint)} selected vLLM "
                f"port {selected} outside block {block_start}-{block_end}."
            )


def _port_conflict_payload(port: int, exc: OSError) -> dict[str, Any]:
    """Build a self-contained collision record for remote-only diagnosis."""

    tcp_sockets = _proc_tcp_rows_for_port(port)
    inodes = {
        str(row["inode"])
        for row in tcp_sockets
        if row.get("inode") not in (None, "", "0")
    }
    owners = _socket_process_owners(inodes)
    for row in tcp_sockets:
        row["processes"] = owners.get(str(row.get("inode")), [])
    return {
        "port": int(port),
        "errno": exc.errno,
        "reason": str(exc),
        "tcp_sockets": tcp_sockets,
    }


def _read_proc_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_proc_int_pair(path: Path) -> list[int] | None:
    raw = _read_proc_text(path)
    if raw is None:
        return None
    parts = raw.split()
    if len(parts) != 2:
        return None
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError:
        return None


def _proc_tcp_rows_for_port(port: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_read_proc_tcp_table(Path("/proc/net/tcp"), port=port, ipv6=False))
    rows.extend(_read_proc_tcp_table(Path("/proc/net/tcp6"), port=port, ipv6=True))
    return rows


def _read_proc_tcp_table(
    path: Path,
    *,
    port: int,
    ipv6: bool,
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            local_address, local_port = _decode_proc_tcp_endpoint(
                parts[1], ipv6=ipv6
            )
            remote_address, remote_port = _decode_proc_tcp_endpoint(
                parts[2], ipv6=ipv6
            )
        except (ValueError, OSError):
            continue
        if local_port != port:
            continue
        rows.append(
            {
                "family": "ipv6" if ipv6 else "ipv4",
                "state": _TCP_STATE_NAMES.get(parts[3], parts[3]),
                "local_address": local_address,
                "local_port": local_port,
                "remote_address": remote_address,
                "remote_port": remote_port,
                "uid": int(parts[7]),
                "inode": parts[9],
            }
        )
    return rows


def _decode_proc_tcp_endpoint(raw: str, *, ipv6: bool) -> tuple[str, int]:
    address_hex, port_hex = raw.split(":", 1)
    packed = bytes.fromhex(address_hex)
    if ipv6:
        packed = b"".join(
            packed[offset : offset + 4][::-1]
            for offset in range(0, len(packed), 4)
        )
    else:
        packed = packed[::-1]
    return str(ipaddress.ip_address(packed)), int(port_hex, 16)


def _socket_process_owners(inodes: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Best-effort map visible socket inodes to processes in this container."""

    if not inodes:
        return {}
    owners: dict[str, list[dict[str, Any]]] = {}
    proc_root = Path("/proc")
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError:
        return owners
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            fd_paths = list((process_dir / "fd").iterdir())
        except OSError:
            continue
        matched: set[str] = set()
        for fd_path in fd_paths:
            try:
                target = os.readlink(fd_path)
            except OSError:
                continue
            if not target.startswith("socket:[") or not target.endswith("]"):
                continue
            inode = target[8:-1]
            if inode in inodes:
                matched.add(inode)
        if not matched:
            continue
        cmdline = _read_proc_text(process_dir / "cmdline")
        process = {
            "pid": int(process_dir.name),
            "cmdline": None if cmdline is None else cmdline.replace("\x00", " ").strip(),
        }
        for inode in matched:
            owners.setdefault(inode, []).append(process)
    return owners


def _endpoint_label(endpoint: ResolvedEndpoint) -> str:
    """Return a compact endpoint label suitable for logs and exceptions."""

    internal = (
        "none"
        if (
            endpoint.vllm_port_block_start is None
            or endpoint.vllm_port_block_end is None
        )
        else (
            f"block={endpoint.vllm_port_block_start}-"
            f"{endpoint.vllm_port_block_end},selected="
            f"{endpoint.selected_vllm_port}"
        )
    )
    return (
        f"{endpoint.group_name}:{endpoint.replica_index}"
        f"(gpu={endpoint.gpu_id},http={endpoint.port},vllm={internal})"
    )


def _endpoint_log_name(endpoint: ResolvedEndpoint) -> str:
    """Build a traversal-safe stable filename for one managed server log."""

    safe_group = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in endpoint.group_name
    )
    return f"{safe_group}_replica_{endpoint.replica_index:02d}.log"


def _managed_process_exit_message(
    endpoint: ResolvedEndpoint,
    *,
    return_code: int,
) -> str:
    """Build a failure message with direct routing to the replica log."""

    log_suffix = f", log={endpoint.log_path}" if endpoint.log_path else ""
    return (
        f"Managed endpoint {_endpoint_label(endpoint)} exited before readiness "
        f"with return code {return_code}{log_suffix}."
    )


def _signal_process_group(record: _ManagedProcess, sig: signal.Signals) -> None:
    """Signal the isolated process group, with a direct-process fallback."""

    try:
        os.killpg(record.process.pid, sig)
        return
    except ProcessLookupError:
        return
    except (AttributeError, PermissionError):
        pass
    if record.process.poll() is not None:
        return
    if sig == signal.SIGTERM:
        record.process.terminate()
    else:
        record.process.kill()


def _endpoint_ready(base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    if not text.strip():
        return True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict)
