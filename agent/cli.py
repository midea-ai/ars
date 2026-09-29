"""Command-line entrypoint for the reward-agent harness."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch_runner import RewardAgentBatchRunner
from .config import RewardAgentConfig
from .runner import RewardAgentRunner
from .serving import (
    ResolvedEndpoint,
    ServingManager,
    derive_client_endpoints_from_server,
    load_server_config,
    resolve_client_endpoint_groups,
    resolve_server_groups,
    resolve_server_startup_config,
    ServerStartupConfig,
    validate_client_server_alignment,
)
from .session import RewardAgentSession


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "run-batch":
        return _run_batch_command(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reward-agent harness against a local vLLM server.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one reward-agent case.")
    run_parser.add_argument("--case-dir", type=Path, required=True, help="Path to the sanitized case directory.")
    run_parser.add_argument("--run-dir", type=Path, default=None, help="Optional output run directory.")
    run_parser.add_argument("--config", type=Path, default=None, help="Optional reward-agent YAML config.")
    run_parser.add_argument("--base-url", default=None, help="OpenAI-compatible vLLM base URL.")
    run_parser.add_argument("--model", default=None, help="Served vLLM model name.")
    run_parser.add_argument("--context-window", type=int, default=None, help="Context window used for budget display.")
    run_parser.add_argument("--timeout-seconds", type=float, default=None, help="HTTP timeout per vLLM request.")
    run_parser.add_argument("--media-mode", choices=["video", "tile"], default=None, help="Default visual evidence mode.")
    run_parser.add_argument(
        "--max-attached-frames-per-call",
        type=int,
        default=None,
        help="Hard limit on attached frames per visual evidence call.",
    )
    run_parser.add_argument("--resume", action="store_true", help="Resume session state from run_dir if present.")
    run_parser.add_argument(
        "--skip-readiness-check",
        action="store_true",
        help="Do not call /v1/models before the first chat request.",
    )
    run_parser.add_argument(
        "--quiet-context",
        action="store_true",
        help="Do not print context token budget lines to the terminal.",
    )

    batch_parser = subparsers.add_parser("run-batch", help="Run reward-agent over a LeRobot dataset.")
    batch_parser.add_argument(
        "--lerobot-root",
        type=Path,
        required=True,
        help="Concrete LeRobot dataset leaf root; collection roots with multiple leaves are rejected.",
    )
    batch_parser.add_argument("--output-root", type=Path, required=True, help="Batch output root.")
    batch_parser.add_argument(
        "--start-server",
        type=Path,
        default=None,
        metavar="SERVER_CONFIG",
        help="Start vLLM servers from this server.yaml before scheduling the batch.",
    )
    batch_parser.add_argument(
        "--endpoints",
        default=None,
        help="Comma-separated external endpoint base URLs, overriding vllm.endpoint_groups.",
    )
    batch_parser.add_argument(
        "--max-concurrent-per-endpoint",
        type=int,
        default=1,
        help="Lease capacity for each endpoint from --endpoints.",
    )
    batch_parser.add_argument("--batch-id", default=None, help="Optional deterministic batch id.")
    batch_parser.add_argument(
        "--num-nodes",
        type=_parse_positive_int_arg,
        default=None,
        help=(
            "Optional total Ray GPU-node count. Omit it or set 1 for the "
            "existing single-node path; values greater than 1 connect to "
            "the platform-started Ray cluster."
        ),
    )
    batch_parser.add_argument(
        "--ray-cluster-ready-timeout-seconds",
        type=_parse_positive_float_arg,
        default=1800.0,
        help=(
            "Seconds to wait for eligible Ray GPU nodes and every node actor's "
            "GPU reservation."
        ),
    )
    batch_parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Attempts per episode, overriding batch.max_retries from --config.",
    )
    batch_parser.add_argument(
        "--episode-indices",
        type=_parse_episode_indices_arg,
        default=None,
        help=(
            "Comma-separated LeRobot episode_index values to schedule, "
            "overriding batch.episode_indices from --config."
        ),
    )
    batch_parser.add_argument(
        "--proposal-batch-results",
        type=Path,
        default=None,
        help=(
            "Optional external proposal batch_results.jsonl. Rows for selected "
            "episodes are validated against --lerobot-root and used as the "
            "candidate progress curve; other episodes are ignored."
        ),
    )
    batch_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Reward-agent YAML config with vllm.endpoint_groups.",
    )
    batch_parser.add_argument("--model", default=None, help="Served vLLM model name.")
    batch_parser.add_argument("--context-window", type=int, default=None, help="Context window used for budget display.")
    batch_parser.add_argument("--timeout-seconds", type=float, default=None, help="HTTP timeout per vLLM request.")
    batch_parser.add_argument("--media-mode", choices=["video", "tile"], default=None, help="Default visual evidence mode.")
    batch_parser.add_argument(
        "--max-attached-frames-per-call",
        type=int,
        default=None,
        help="Hard limit on attached frames per visual evidence call.",
    )
    batch_parser.add_argument(
        "--skip-readiness-check",
        action="store_true",
        help="Do not wait for endpoint /v1/models before scheduling episodes.",
    )
    batch_parser.add_argument(
        "--quiet-context",
        action="store_true",
        help="Do not print context token budget lines to the terminal.",
    )
    return parser


def _run_command(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    session = RewardAgentSession(
        args.case_dir,
        run_dir=args.run_dir,
        config=config,
        resume=args.resume,
    )
    runner = RewardAgentRunner(
        session=session,
        config=config,
        print_context=not args.quiet_context,
        check_server_ready=not args.skip_readiness_check,
    )
    status = runner.run()
    print(
        "Reward-agent run finished: "
        f"run_dir={status['run_dir']} "
        f"completed={status['completed']} "
        f"status={status['status']} "
        f"finalized={status['finalized']}"
    )
    return 0


def _run_batch_command(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    num_nodes = int(getattr(args, "num_nodes", None) or 1)
    if num_nodes > 1:
        return _run_ray_batch_command(args, config=config, num_nodes=num_nodes)
    return _run_local_batch_command(args, config=config)


def _run_local_batch_command(
    args: argparse.Namespace,
    *,
    config: RewardAgentConfig,
) -> int:
    """Run an episode batch against local or externally managed endpoints."""

    client_endpoints, server_endpoints, startup_config = (
        _resolve_batch_endpoints_from_args(args, config)
    )
    if server_endpoints and args.skip_readiness_check:
        raise ValueError(
            "run-batch --start-server requires managed readiness checks and "
            "does not accept --skip-readiness-check."
        )
    resolved_batch_id = (
        args.batch_id
        if args.batch_id is not None
        else datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    manager = ServingManager(
        client_endpoints,
        server_endpoints=server_endpoints,
        startup_config=startup_config,
    )
    runner = RewardAgentBatchRunner(
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
        config=config,
        serving_manager=manager,
        batch_id=resolved_batch_id,
        proposal_batch_results=getattr(args, "proposal_batch_results", None),
        check_server_ready=False,
        print_context=not args.quiet_context,
    )
    # All source/config validation and the atomic batch-directory reservation
    # must finish before readiness checks, log creation, or managed Popen calls.
    prepared = runner.prepare()
    if server_endpoints:
        manager.set_log_dir(prepared.workspace.servers_dir)
    try:
        if server_endpoints:
            manager.start_managed()
        elif not args.skip_readiness_check:
            manager.wait_ready()
        batch = runner.run(prepared)
    finally:
        manager.shutdown()

    print(
        "Reward-agent batch finished: "
        f"batch_dir={batch.batch_dir} "
        f"batch_results={batch.batch_results} "
        f"total_episodes={batch.total_episodes}"
    )
    return 0


def _run_ray_batch_command(
    args: argparse.Namespace,
    *,
    config: RewardAgentConfig,
    num_nodes: int,
) -> int:
    """Dispatch a contiguous static shard to each platform-owned Ray node."""

    if args.start_server is None:
        raise ValueError(
            "run-batch --num-nodes > 1 requires --start-server so every "
            "Ray node can launch the same server.yaml layout."
        )
    external_endpoints = [
        endpoint.strip()
        for endpoint in str(args.endpoints or "").split(",")
        if endpoint.strip()
    ]
    if external_endpoints:
        raise ValueError(
            "run-batch --num-nodes > 1 does not accept --endpoints; each "
            "Ray worker uses node-local endpoints from --start-server."
        )
    if args.skip_readiness_check:
        raise ValueError(
            "run-batch --num-nodes > 1 requires the cross-node readiness "
            "barrier and does not accept --skip-readiness-check."
        )

    # Keep Ray optional for local development: importing this module does not
    # import ray itself, and this branch is unreachable for normal run-batch.
    from .ray_batch_runner import run_ray_batch

    batch = run_ray_batch(
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
        config=config,
        server_config=args.start_server,
        num_nodes=num_nodes,
        batch_id=args.batch_id,
        proposal_batch_results=getattr(args, "proposal_batch_results", None),
        max_concurrent_per_endpoint=int(args.max_concurrent_per_endpoint),
        cluster_ready_timeout_seconds=float(
            args.ray_cluster_ready_timeout_seconds
        ),
        print_context=not args.quiet_context,
    )
    print(
        "Reward-agent Ray batch finished: "
        f"batch_dir={batch.batch_dir} "
        f"distributed_summary={batch.distributed_summary} "
        f"num_nodes={batch.num_nodes} "
        f"total_episodes={batch.total_episodes} "
        f"status={batch.status}"
    )
    return 0 if batch.status == "completed" else 1


def _resolve_batch_endpoints_from_args(
    args: argparse.Namespace,
    config: RewardAgentConfig,
) -> tuple[
    list[ResolvedEndpoint],
    list[ResolvedEndpoint],
    ServerStartupConfig | None,
]:
    server_endpoints: list[ResolvedEndpoint] = []
    startup_config: ServerStartupConfig | None = None
    if args.start_server is not None:
        server_payload = load_server_config(args.start_server)
        startup_config = resolve_server_startup_config(server_payload.get("startup"))
        server_endpoints = resolve_server_groups(server_payload.get("server_groups", []))

    endpoints = [
        endpoint.strip()
        for endpoint in str(args.endpoints or "").split(",")
        if endpoint.strip()
    ]
    if endpoints:
        client_endpoints = resolve_client_endpoint_groups(
            [
                {
                    "name": "cli_external",
                    "max_concurrent": int(args.max_concurrent_per_endpoint),
                    "endpoints": endpoints,
                }
            ]
        )
    elif config.vllm.endpoint_groups:
        client_endpoints = resolve_client_endpoint_groups(list(config.vllm.endpoint_groups))
    elif server_endpoints:
        client_endpoints = derive_client_endpoints_from_server(
            server_endpoints,
            max_concurrent=int(args.max_concurrent_per_endpoint),
        )
    else:
        raise ValueError(
            "run-batch requires vllm.endpoint_groups in --config, --endpoints, "
            "or --start-server so client endpoints can be derived."
        )

    if server_endpoints:
        validate_client_server_alignment(
            client_endpoints=client_endpoints,
            server_endpoints=server_endpoints,
        )
    return client_endpoints, server_endpoints, startup_config


def _load_config_from_args(args: argparse.Namespace) -> RewardAgentConfig:
    overrides: dict[str, Any] = {}
    if getattr(args, "base_url", None) is not None:
        overrides.setdefault("vllm", {})["base_url"] = args.base_url
    if args.model is not None:
        overrides.setdefault("vllm", {})["model"] = args.model
    if args.context_window is not None:
        overrides.setdefault("vllm", {})["context_window"] = args.context_window
    if args.timeout_seconds is not None:
        overrides.setdefault("vllm", {})["timeout_seconds"] = args.timeout_seconds
    if args.media_mode is not None:
        overrides.setdefault("visual", {})["media_mode"] = args.media_mode
    if args.max_attached_frames_per_call is not None:
        overrides.setdefault("visual", {})["max_attached_frames_per_call"] = args.max_attached_frames_per_call
    if getattr(args, "max_retries", None) is not None:
        overrides.setdefault("batch", {})["max_retries"] = args.max_retries
    if getattr(args, "episode_indices", None) is not None:
        overrides.setdefault("batch", {})["episode_indices"] = args.episode_indices

    if args.config is not None:
        return RewardAgentConfig.from_yaml(args.config, overrides=overrides)
    if overrides:
        return RewardAgentConfig.from_dict(overrides)
    return RewardAgentConfig.default()


def _parse_episode_indices_arg(raw_value: str) -> list[int]:
    """Parse a shell-friendly comma-separated episode_index list."""

    text = str(raw_value).strip()
    if not text:
        raise argparse.ArgumentTypeError("--episode-indices must be a non-empty comma-separated list.")

    indices: list[int] = []
    seen: set[int] = set()
    for position, token in enumerate(text.split(",")):
        stripped = token.strip()
        if not stripped:
            raise argparse.ArgumentTypeError(
                f"--episode-indices contains an empty item at position {position}."
            )
        try:
            value = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--episode-indices item {stripped!r} is not an integer."
            ) from exc
        if value < 0:
            raise argparse.ArgumentTypeError(
                f"--episode-indices item {stripped!r} must be non-negative."
            )
        if value in seen:
            raise argparse.ArgumentTypeError(
                f"--episode-indices contains duplicate episode_index {value}."
            )
        seen.add(value)
        indices.append(value)
    return indices


def _parse_positive_int_arg(raw_value: str) -> int:
    """Parse a strictly positive CLI integer without accepting floats."""

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {raw_value!r}."
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {raw_value!r}."
        )
    return value


def _parse_positive_float_arg(raw_value: str) -> float:
    """Parse a finite positive CLI duration."""

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive number, got {raw_value!r}."
        ) from exc
    if not (value > 0.0) or value == float("inf"):
        raise argparse.ArgumentTypeError(
            f"expected a finite positive number, got {raw_value!r}."
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
