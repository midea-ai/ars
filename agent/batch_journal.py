"""Durable JSONL journal shared by reward and progress batch runners."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class BatchJournal:
    """Append durable batch events/results and atomically refresh a summary.

    A journal represents one fresh process run.  It deliberately does not load
    old terminal identities for skipping or merging; callers must apply their
    overwrite/new-directory policy before constructing it.
    """

    def __init__(self, batch_dir: str | Path) -> None:
        self.batch_dir = Path(batch_dir).expanduser().resolve()
        self.batch_results = self.batch_dir / "batch_results.jsonl"
        self.batch_events = self.batch_dir / "batch_events.jsonl"
        self.batch_summary = self.batch_dir / "batch_summary.json"
        self._lock = threading.RLock()
        self._terminal_identities: set[tuple[str, str]] = set()
        self._results_existed_at_start = (
            self.batch_results.is_file() and self.batch_results.stat().st_size > 0
        )

    def append_event(self, payload: Mapping[str, Any]) -> None:
        """Durably append one event as exactly one strict UTF-8 JSON line."""

        self._append_jsonl(self.batch_events, payload)

    def append_terminal_result(self, row: Mapping[str, Any]) -> None:
        """Append one episode terminal row, rejecting same-run duplicates."""

        identity = _terminal_identity(row)
        with self._lock:
            if self._results_existed_at_start:
                raise FileExistsError(
                    f"{self.batch_results} already contained terminal rows when "
                    "this BatchJournal was created; resume/merge is not supported."
                )
            if identity in self._terminal_identities:
                raise ValueError(
                    "Duplicate terminal batch-result row in the current run for "
                    f"{identity[0]}={identity[1]!r}."
                )
            # Record the identity only after the durable append succeeds.  A
            # failed disk write may therefore be retried by the caller.
            self._append_jsonl_unlocked(self.batch_results, row)
            self._terminal_identities.add(identity)

    def refresh_summary(self, base_payload: Mapping[str, Any]) -> None:
        """Atomically rewrite summary fields derived from durable terminal rows."""

        with self._lock:
            rows = self._read_terminal_results_unlocked()
            status_counts: dict[str, int] = {}
            for row in rows:
                status = str(row.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1

            payload = dict(base_payload)
            payload["terminal_result_count"] = len(rows)
            payload["status_counts"] = dict(sorted(status_counts.items()))
            payload["completed_episodes"] = sum(
                1 for row in rows if row.get("status") == "completed"
            )
            payload["failed_episodes"] = sum(
                1 for row in rows if row.get("status") != "completed"
            )
            self._atomic_write_json_unlocked(self.batch_summary, payload)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._append_jsonl_unlocked(path, payload)

    def _append_jsonl_unlocked(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError(f"JSONL payload must be a mapping, got {type(payload)!r}.")
        encoded = json.dumps(
            _json_ready(payload, batch_dir=self.batch_dir),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if "\n" in encoded or "\r" in encoded:
            # json.dumps escapes control newlines inside strings, so a literal
            # newline here would signal an unsafe/custom encoder regression.
            raise ValueError("Serialized JSONL payload unexpectedly contains a newline.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_terminal_results_unlocked(self) -> list[dict[str, Any]]:
        if not self.batch_results.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.batch_results.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.batch_results}: invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"{self.batch_results}: line {line_number} must be a JSON object."
                    )
                rows.append(payload)
        return rows

    def _atomic_write_json_unlocked(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            _json_ready(payload, batch_dir=self.batch_dir),
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
            # Persist the directory entry where the platform supports opening
            # a directory descriptor.  The file itself was already fsynced.
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def _terminal_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    """Choose a stable episode identity without relying on result ordering."""

    source_traj_id = row.get("source_traj_id")
    if source_traj_id not in (None, ""):
        return ("source_traj_id", str(source_traj_id))
    episode_index = row.get("episode_index")
    if episode_index is not None:
        if isinstance(episode_index, bool):
            raise TypeError("terminal result episode_index must be an integer, not bool.")
        return ("episode_index", str(episode_index))
    raise ValueError(
        "A terminal batch-result row must contain source_traj_id or episode_index."
    )


def _json_ready(value: Any, *, batch_dir: Path) -> Any:
    """Convert supported scientific/Python values without permitting sets."""

    if isinstance(value, Path):
        resolved = value.expanduser().resolve()
        try:
            return resolved.relative_to(batch_dir).as_posix()
        except ValueError:
            return str(resolved)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value), batch_dir=batch_dir)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist(), batch_dir=batch_dir)
    if isinstance(value, np.generic):
        return _json_ready(value.item(), batch_dir=batch_dir)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, batch_dir=batch_dir)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, batch_dir=batch_dir) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("Sets are not supported in batch JSON artifacts.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported batch JSON value type: {type(value)!r}.")


__all__ = ["BatchJournal"]
