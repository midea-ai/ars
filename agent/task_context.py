"""Reward-agent-only text and reference examples resolved by robot task."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import TaskContextConfig
from .reference_examples import ReferenceExampleConfig, parse_reference_examples


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_task_text(text: str) -> str:
    """Normalize task keys only enough to ignore accidental whitespace changes."""

    return _WHITESPACE_RE.sub(" ", str(text).strip())


@dataclass(frozen=True, slots=True)
class TaskContextItem:
    """One normalized task key with optional text and multimodal references."""

    task: str
    normalized_task: str
    instruction_append: str | None
    reference_examples: tuple[ReferenceExampleConfig, ...]


@dataclass(frozen=True, slots=True)
class ResolvedTaskContext:
    """The exact task-bound context selected for one reward-agent run."""

    base_instruction: str
    final_instruction: str
    instruction_append: str | None
    reference_examples: tuple[ReferenceExampleConfig, ...]
    matched: bool
    matched_task: str | None
    context_path: Path | None

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return task matching metadata without serializing private media paths."""

        return {
            "enabled": self.context_path is not None,
            "matched": bool(self.matched),
            "matched_task": self.matched_task,
            "path": str(self.context_path) if self.context_path is not None else None,
            "instruction_append": self.instruction_append,
            "reference_example_count": len(self.reference_examples),
        }


@dataclass(frozen=True, slots=True)
class TaskContextCoverage:
    """Batch-level task context coverage and selected component counts."""

    enabled: bool
    matched_episodes: int
    unmatched_episodes: int
    matched_tasks: int
    unmatched_tasks: int
    episodes_with_instruction_append: int
    episodes_with_reference_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "matched_episodes": int(self.matched_episodes),
            "unmatched_episodes": int(self.unmatched_episodes),
            "matched_tasks": int(self.matched_tasks),
            "unmatched_tasks": int(self.unmatched_tasks),
            "episodes_with_instruction_append": int(self.episodes_with_instruction_append),
            "episodes_with_reference_examples": int(self.episodes_with_reference_examples),
        }


class TaskContextResolver:
    """Resolve task-bound text and references by exact normalized task text."""

    def __init__(
        self,
        *,
        context_by_task: dict[str, TaskContextItem],
        path: Path | None,
        require_all_matched: bool,
    ) -> None:
        self._context_by_task = dict(context_by_task)
        self.path = path
        self.require_all_matched = require_all_matched

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @classmethod
    def disabled(cls) -> "TaskContextResolver":
        return cls(context_by_task={}, path=None, require_all_matched=False)

    @classmethod
    def from_config(cls, config: TaskContextConfig) -> "TaskContextResolver":
        """Load and validate a task context sidecar once before scheduling."""

        # Normalize programmatic Path values and YAML strings before resolving
        # the task-context file. YAML validation remains the caller's responsibility.
        raw_path = str(config.path or "").strip()
        if not raw_path:
            return cls.disabled()
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"task_context.path does not exist: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"{path}: expected a mapping with an items list.")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise TypeError(f"{path}: items must be a list of task context mappings.")

        context_by_task: dict[str, TaskContextItem] = {}
        raw_task_by_normalized: dict[str, str] = {}
        for index, raw_item in enumerate(items):
            field_name = f"items[{index}]"
            if not isinstance(raw_item, dict):
                raise TypeError(f"{path}: {field_name} must be a mapping.")
            raw_task_value = raw_item.get("task")
            if not isinstance(raw_task_value, str) or not raw_task_value.strip():
                raise ValueError(f"{path}: {field_name}.task must be a non-empty string.")
            raw_task = raw_task_value.strip()
            normalized_task = normalize_task_text(raw_task)
            if normalized_task in raw_task_by_normalized:
                previous = raw_task_by_normalized[normalized_task]
                raise ValueError(
                    f"{path}: duplicate task after whitespace normalization: "
                    f"{previous!r} and {raw_task!r}."
                )

            raw_append = raw_item.get("instruction_append")
            if raw_append is not None and not isinstance(raw_append, str):
                raise TypeError(f"{path}: {field_name}.instruction_append must be a string.")
            instruction_append = str(raw_append).strip() if raw_append is not None else None
            if instruction_append == "":
                raise ValueError(f"{path}: {field_name}.instruction_append must not be empty.")

            # Keep normalized keys unique for both text and reference-only entries.
            raw_task_by_normalized[normalized_task] = raw_task

            reference_examples = parse_reference_examples(
                raw_item.get("reference_examples"),
                source_path=path,
                field_name=f"{field_name}.reference_examples",
            )
            if instruction_append is None and not reference_examples:
                raise ValueError(
                    f"{path}: {field_name} must define instruction_append, "
                    "reference_examples, or both."
                )

            context_by_task[normalized_task] = TaskContextItem(
                task=raw_task,
                normalized_task=normalized_task,
                instruction_append=instruction_append,
                reference_examples=reference_examples,
            )

        return cls(
            context_by_task=context_by_task,
            path=path,
            require_all_matched=config.require_all_matched,
        )

    def resolve(self, base_instruction: str) -> ResolvedTaskContext:
        """Resolve one task before any instruction append is applied."""

        base = str(base_instruction).strip()
        item = self._context_by_task.get(normalize_task_text(base))
        if item is None:
            return ResolvedTaskContext(
                base_instruction=base,
                final_instruction=base,
                instruction_append=None,
                reference_examples=(),
                matched=False,
                matched_task=None,
                context_path=self.path,
            )
        final_instruction = (
            f"{base}\n{item.instruction_append}"
            if item.instruction_append is not None
            else base
        )
        return ResolvedTaskContext(
            base_instruction=base,
            final_instruction=final_instruction,
            instruction_append=item.instruction_append,
            reference_examples=item.reference_examples,
            matched=True,
            matched_task=item.task,
            context_path=self.path,
        )

    def require_coverage(self, resolved: Iterable[ResolvedTaskContext]) -> None:
        """Fail before scheduling when strict task context coverage is enabled."""

        resolved_list = list(resolved)
        if not self.enabled or not self.require_all_matched:
            return
        missing = sorted(
            {item.base_instruction for item in resolved_list if not item.matched},
            key=normalize_task_text,
        )
        if not missing:
            return
        formatted = "\n".join(f"  - {task}" for task in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            "task_context.require_all_matched=true, but some LeRobot tasks "
            f"have no context entry:\n{formatted}{suffix}"
        )


def summarize_task_context_coverage(
    resolved: Iterable[ResolvedTaskContext],
) -> TaskContextCoverage:
    """Summarize exact task matching and selected component coverage."""

    resolved_list = list(resolved)
    enabled = any(item.context_path is not None for item in resolved_list)
    if not enabled:
        return TaskContextCoverage(False, 0, 0, 0, 0, 0, 0)

    matched_tasks: set[str] = set()
    unmatched_tasks: set[str] = set()
    for item in resolved_list:
        target = matched_tasks if item.matched else unmatched_tasks
        target.add(normalize_task_text(item.base_instruction))
    matched_episodes = sum(1 for item in resolved_list if item.matched)
    return TaskContextCoverage(
        enabled=True,
        matched_episodes=matched_episodes,
        unmatched_episodes=len(resolved_list) - matched_episodes,
        matched_tasks=len(matched_tasks),
        unmatched_tasks=len(unmatched_tasks),
        episodes_with_instruction_append=sum(
            1 for item in resolved_list if item.instruction_append is not None
        ),
        episodes_with_reference_examples=sum(
            1 for item in resolved_list if item.reference_examples
        ),
    )
