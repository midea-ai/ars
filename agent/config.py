"""Configuration loading for the reward-agent harness and workflow runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .schemas import MediaMode, ResetServerCacheMode


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "ARS-vlm"
DEFAULT_CONTEXT_WINDOW = 65536
DEFAULT_WORKFLOW_RESOURCE = "workflows/default.yaml"


def _require_boolean(value: Any, *, field_name: str) -> bool:
    """Require a native YAML/JSON boolean without truthiness coercion."""

    if type(value) is not bool:
        raise TypeError(
            f"{field_name} must be a boolean, got {type(value).__name__}."
        )
    return value


def _validate_boolean_mapping_keys(
    payload: dict[str, Any],
    *,
    field_prefix: str,
    keys: tuple[str, ...],
) -> None:
    """Validate known boolean keys while leaving unrelated extension keys untouched."""

    for key in keys:
        if key in payload:
            _require_boolean(payload[key], field_name=f"{field_prefix}.{key}")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Model sampling settings used for every chat call unless a stage overrides them."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 0
    max_tokens_per_step: int = 8192
    max_tokens_final: int = 8192
    reserved_output_tokens: int = 2048
    thinking_token_budget: int | None = None
    thinking_budget_forced_end_phrase: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": True})

    def to_request_payload(
        self,
        *,
        final_step: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }
        stage_overrides = dict(overrides or {})
        for key, value in stage_overrides.items():
            if key == "chat_template_kwargs" and isinstance(value, dict):
                merged_kwargs = dict(values["chat_template_kwargs"])
                merged_kwargs.update(value)
                values[key] = merged_kwargs
            elif key in values:
                values[key] = value

        payload = {
            "temperature": float(values["temperature"]),
            "top_p": float(values["top_p"]),
            "top_k": int(values["top_k"]),
            "min_p": float(values["min_p"]),
            "presence_penalty": float(values["presence_penalty"]),
            "frequency_penalty": float(values["frequency_penalty"]),
            "repetition_penalty": float(values["repetition_penalty"]),
            "seed": int(values["seed"]),
            "max_tokens": int(values["max_tokens_final"] if final_step else values["max_tokens_per_step"]),
            "chat_template_kwargs": dict(values["chat_template_kwargs"]),
        }
        # vLLM treats thinking_token_budget as a request-level sampling
        # parameter. Keep it absent unless explicitly configured so servers
        # without --reasoning-config continue to work.
        if values["thinking_token_budget"] is not None:
            payload["thinking_token_budget"] = int(values["thinking_token_budget"])
        return payload


@dataclass(frozen=True, slots=True)
class LabelBarModeConfig:
    """Per-media label fields rendered outside agent-facing visual evidence."""

    frame: bool = True
    progress: bool = True

    def __post_init__(self) -> None:
        """Keep external label visibility controls strictly boolean."""

        _require_boolean(self.frame, field_name="visual.external_label_bar.<mode>.frame")
        _require_boolean(self.progress, field_name="visual.external_label_bar.<mode>.progress")


@dataclass(frozen=True, slots=True)
class ExternalLabelBarConfig:
    """External label-bar controls for agent-facing inspect_visual artifacts."""

    enabled: bool = False
    video: LabelBarModeConfig = field(default_factory=LabelBarModeConfig)
    tile: LabelBarModeConfig = field(default_factory=LabelBarModeConfig)

    def __post_init__(self) -> None:
        """Reject truthy strings for the top-level label-bar switch."""

        _require_boolean(self.enabled, field_name="visual.external_label_bar.enabled")


@dataclass(frozen=True, slots=True)
class VisualConfig:
    """Bounds for model-requested visual evidence."""

    media_mode: MediaMode = "video"
    max_attached_frames_per_call: int = 64
    agent_video_fps: int = 1
    review_video_fps: float | None = None
    external_label_bar: ExternalLabelBarConfig = field(default_factory=ExternalLabelBarConfig)
    tile_columns: int = 4
    tile_max_width: int = 4096
    video_codec: str = "libx264"

    def __post_init__(self) -> None:
        """Reject ambiguous non-integer indexing rates for agent-facing videos."""

        if isinstance(self.agent_video_fps, bool) or not isinstance(self.agent_video_fps, int):
            raise TypeError("visual.agent_video_fps must be a positive integer.")
        if self.agent_video_fps <= 0:
            raise ValueError("visual.agent_video_fps must be a positive integer.")


@dataclass(frozen=True, slots=True)
class ProgressConfig:
    """Numeric value-space settings for progress curves."""

    value_max_points: int = 1000

    def __post_init__(self) -> None:
        """Reject ambiguous value-space bounds before tools expose them."""

        if isinstance(self.value_max_points, bool) or not isinstance(self.value_max_points, int):
            raise TypeError("progress.value_max_points must be a positive integer.")
        if self.value_max_points <= 0:
            raise ValueError("progress.value_max_points must be a positive integer.")

    @property
    def default_jump_threshold_points(self) -> int:
        """Use roughly a 5% jump threshold at any configured progress scale."""

        return max(1, int(round(float(self.value_max_points) * 0.05)))


@dataclass(frozen=True, slots=True)
class IsSuccessGroundTruthConfig:
    """Controls use of the source episode's terminal success label."""

    enabled: bool = False

    def __post_init__(self) -> None:
        """Require an explicit boolean so strings cannot silently enable the feature."""

        _require_boolean(self.enabled, field_name="is_success_ground_truth.enabled")


@dataclass(frozen=True, slots=True)
class VllmConfig:
    """Connection settings for the externally managed vLLM server."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    context_window: int = DEFAULT_CONTEXT_WINDOW
    timeout_seconds: float = 900.0
    reset_server_cache: ResetServerCacheMode = "never"
    endpoint_groups: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def root_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[: -len("/v1")]
        return self.base_url.rstrip("/")

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"


@dataclass(frozen=True, slots=True)
class TaskContextConfig:
    """Optional reward-agent-only context resolved by exact normalized task text."""

    path: str | None = None
    require_all_matched: bool = False

    def __post_init__(self) -> None:
        """Require an explicit matching-policy boolean at the config boundary."""

        _require_boolean(
            self.require_all_matched,
            field_name="task_context.require_all_matched",
        )


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Batch scheduler settings that are not used by single-case runs."""

    max_retries: int = 1
    episode_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Keep batch scheduling controls explicit before launching workers."""

        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("batch.max_retries must be a positive integer.")
        if self.max_retries <= 0:
            raise ValueError("batch.max_retries must be a positive integer.")
        object.__setattr__(
            self,
            "episode_indices",
            _normalize_episode_indices(
                self.episode_indices,
                field_name="batch.episode_indices",
            ),
        )


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One prompt-controlled stage in a workflow."""

    stage_id: str
    prompt: str
    max_steps: int = 24
    fresh_session: bool = False
    allowed_tools: tuple[str, ...] = ()
    initial_media: dict[str, Any] = field(default_factory=dict)
    initial_context: dict[str, Any] = field(default_factory=dict)
    tool_config: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    promote_observations_to_confirmed_on_complete: bool = False

    def __post_init__(self) -> None:
        """Validate stage control flags before they can affect runner behavior."""

        field_prefix = f"workflow stage {self.stage_id!r}"
        _require_boolean(self.fresh_session, field_name=f"{field_prefix}.fresh_session")
        _require_boolean(
            self.promote_observations_to_confirmed_on_complete,
            field_name=(
                f"{field_prefix}.promote_observations_to_confirmed_on_complete"
            ),
        )
        _validate_boolean_mapping_keys(
            self.initial_media,
            field_prefix=f"{field_prefix}.initial_media",
            keys=("full_video",),
        )
        _validate_boolean_mapping_keys(
            self.initial_context,
            field_prefix=f"{field_prefix}.initial_context",
            keys=("progress_snapshot", "progress_summary"),
        )
        if (
            self.promote_observations_to_confirmed_on_complete
            and self.stage_id != "bootstrap_observations"
        ):
            raise ValueError(
                "promote_observations_to_confirmed_on_complete=true is only supported "
                "for the bootstrap_observations stage."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageSpec":
        stage_id = str(data.get("id") or data.get("stage_id") or "").strip()
        if not stage_id:
            raise ValueError(f"Workflow stage is missing a non-empty id: {data!r}")
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Workflow stage {stage_id!r} is missing a prompt path.")
        tool_config = dict(data.get("tool_config") or {})
        _validate_stage_tool_config(stage_id=stage_id, tool_config=tool_config)
        return cls(
            stage_id=stage_id,
            prompt=prompt,
            max_steps=int(data.get("max_steps", 24)),
            fresh_session=data.get("fresh_session", False),
            allowed_tools=tuple(str(tool) for tool in data.get("allowed_tools", ())),
            initial_media=dict(data.get("initial_media") or {}),
            initial_context=dict(data.get("initial_context") or {}),
            tool_config=tool_config,
            generation=dict(data.get("generation") or {}),
            promote_observations_to_confirmed_on_complete=data.get(
                "promote_observations_to_confirmed_on_complete",
                False,
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Prompt workflow metadata loaded from a YAML resource or local file."""

    name: str
    common_prompt: str
    stages: tuple[StageSpec, ...]
    source_path: str
    prompt_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str) -> "WorkflowSpec":
        name = str(data.get("name") or "default").strip()
        common_prompt = str(data.get("common_prompt") or "").strip()
        if not common_prompt:
            raise ValueError("Workflow is missing common_prompt.")
        stages = tuple(StageSpec.from_dict(stage) for stage in data.get("stages", ()))
        if not stages:
            raise ValueError("Workflow must contain at least one stage.")
        return cls(
            name=name,
            common_prompt=common_prompt,
            stages=stages,
            source_path=source_path,
        )


@dataclass(frozen=True, slots=True)
class RewardAgentConfig:
    """Top-level harness configuration."""

    vllm: VllmConfig = field(default_factory=VllmConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    task_context: TaskContextConfig = field(default_factory=TaskContextConfig)
    is_success_ground_truth: IsSuccessGroundTruthConfig = field(
        default_factory=IsSuccessGroundTruthConfig
    )
    batch: BatchConfig = field(default_factory=BatchConfig)
    workflow: WorkflowSpec | None = None

    @classmethod
    def default(cls) -> "RewardAgentConfig":
        workflow = load_workflow(DEFAULT_WORKFLOW_RESOURCE)
        return cls(workflow=workflow)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "RewardAgentConfig":
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config YAML must contain a mapping, got {type(payload).__name__}.")
        if overrides:
            payload = _deep_merge_dicts(payload, overrides)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardAgentConfig":
        workflow_value = payload.get("workflow")
        if isinstance(workflow_value, WorkflowSpec):
            workflow = workflow_value
        elif isinstance(workflow_value, str):
            workflow = load_workflow(workflow_value)
        elif isinstance(workflow_value, dict):
            workflow_path = workflow_value.get("path")
            workflow = load_workflow(str(workflow_path)) if workflow_path else WorkflowSpec.from_dict(
                workflow_value,
                source_path="<inline>",
            )
        else:
            workflow = load_workflow(DEFAULT_WORKFLOW_RESOURCE)

        vllm_payload = dict(payload.get("vllm", {}) or {})
        if "endpoint_groups" in vllm_payload:
            vllm_payload["endpoint_groups"] = tuple(
                dict(group) for group in (vllm_payload.get("endpoint_groups") or ())
            )
        generation_payload = payload.get("generation", {}) or {}
        progress_payload = payload.get("progress", {}) or {}
        visual_payload = payload.get("visual", {}) or {}
        task_context_payload = payload.get("task_context", {}) or {}
        is_success_ground_truth_payload = (
            payload["is_success_ground_truth"]
            if "is_success_ground_truth" in payload
            else {}
        )
        batch_payload = payload["batch"] if "batch" in payload else {}
        if not isinstance(task_context_payload, dict):
            raise TypeError("task_context must be a mapping with path/require_all_matched fields.")
        if not isinstance(is_success_ground_truth_payload, dict):
            raise TypeError("is_success_ground_truth must be a mapping with an enabled field.")
        if not isinstance(batch_payload, dict):
            raise TypeError("batch must be a mapping with max_retries/episode_indices fields.")
        return cls(
            vllm=VllmConfig(**_filter_dataclass_kwargs(VllmConfig, vllm_payload)),
            generation=GenerationConfig(**_filter_dataclass_kwargs(GenerationConfig, generation_payload)),
            progress=ProgressConfig(**_filter_dataclass_kwargs(ProgressConfig, progress_payload)),
            visual=_load_visual_config(visual_payload),
            task_context=TaskContextConfig(
                **_filter_dataclass_kwargs(TaskContextConfig, task_context_payload)
            ),
            is_success_ground_truth=IsSuccessGroundTruthConfig(
                **_filter_dataclass_kwargs(
                    IsSuccessGroundTruthConfig,
                    is_success_ground_truth_payload,
                )
            ),
            batch=BatchConfig(**_filter_dataclass_kwargs(BatchConfig, batch_payload)),
            workflow=workflow,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.workflow is not None:
            data["workflow"] = asdict(self.workflow)
        return data


def load_workflow(path: str | Path) -> WorkflowSpec:
    """Load a workflow from a package resource or a filesystem path."""

    text, source_path = _read_resource_or_file(path)
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow YAML must contain a mapping, got {type(payload).__name__}.")
    workflow = WorkflowSpec.from_dict(payload, source_path=source_path)
    prompt_paths = [workflow.common_prompt] + [stage.prompt for stage in workflow.stages]
    prompt_hashes = {
        prompt_path: _short_hash(load_prompt_text(prompt_path))
        for prompt_path in prompt_paths
    }
    return WorkflowSpec(
        name=workflow.name,
        common_prompt=workflow.common_prompt,
        stages=workflow.stages,
        source_path=workflow.source_path,
        prompt_hashes=prompt_hashes,
    )


def load_prompt_text(path: str | Path) -> str:
    """Load prompt text from the reward-agent package or a direct file path."""

    text, _source_path = _read_resource_or_file(path)
    return text.rstrip() + "\n"


def _read_resource_or_file(path: str | Path) -> tuple[str, str]:
    raw_path = Path(path).expanduser()
    if raw_path.is_file():
        return raw_path.read_text(encoding="utf-8"), str(raw_path.resolve())

    resource_path = str(path).strip().replace("\\", "/")
    if resource_path.startswith("agent/"):
        resource_path = resource_path.removeprefix("agent/")
    package_root = files("agent")
    resource = package_root.joinpath(resource_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Reward-agent resource not found: {path}")
    return resource.read_text(encoding="utf-8"), f"agent/{resource_path}"


def _filter_dataclass_kwargs(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    return {key: value for key, value in dict(payload).items() if key in allowed}


def _normalize_episode_indices(raw_value: Any, *, field_name: str) -> tuple[int, ...] | None:
    """Validate an optional explicit LeRobot episode_index schedule."""

    if raw_value is None:
        return None
    if isinstance(raw_value, str) or not isinstance(raw_value, (list, tuple)):
        raise TypeError(f"{field_name} must be null or a non-empty list of non-negative integers.")
    if not raw_value:
        raise ValueError(f"{field_name} must be null or a non-empty list of non-negative integers.")

    indices: list[int] = []
    seen: set[int] = set()
    for position, value in enumerate(raw_value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name}[{position}] must be a non-negative integer.")
        if value < 0:
            raise ValueError(f"{field_name}[{position}] must be a non-negative integer.")
        if value in seen:
            raise ValueError(f"{field_name} contains duplicate episode_index {value}.")
        seen.add(value)
        indices.append(value)
    return tuple(indices)


def _load_visual_config(payload: dict[str, Any]) -> VisualConfig:
    """Build VisualConfig while preserving nested external_label_bar structure."""

    visual_kwargs = _filter_dataclass_kwargs(VisualConfig, payload)
    label_payload = visual_kwargs.pop("external_label_bar", {}) or {}
    if not isinstance(label_payload, dict):
        raise TypeError("visual.external_label_bar must be a mapping with enabled/video/tile fields.")
    visual_kwargs["external_label_bar"] = _load_external_label_bar_config(label_payload)
    return VisualConfig(**visual_kwargs)


def _load_external_label_bar_config(payload: dict[str, Any]) -> ExternalLabelBarConfig:
    """Build nested label-bar config from YAML dictionaries."""

    video_payload = payload.get("video", {}) or {}
    tile_payload = payload.get("tile", {}) or {}
    if not isinstance(video_payload, dict):
        raise TypeError("visual.external_label_bar.video must be a mapping.")
    if not isinstance(tile_payload, dict):
        raise TypeError("visual.external_label_bar.tile must be a mapping.")
    _validate_boolean_mapping_keys(
        video_payload,
        field_prefix="visual.external_label_bar.video",
        keys=("frame", "progress"),
    )
    _validate_boolean_mapping_keys(
        tile_payload,
        field_prefix="visual.external_label_bar.tile",
        keys=("frame", "progress"),
    )
    return ExternalLabelBarConfig(
        enabled=payload.get("enabled", False),
        video=LabelBarModeConfig(**_filter_dataclass_kwargs(LabelBarModeConfig, video_payload)),
        tile=LabelBarModeConfig(**_filter_dataclass_kwargs(LabelBarModeConfig, tile_payload)),
    )


def _validate_stage_tool_config(*, stage_id: str, tool_config: dict[str, Any]) -> None:
    """Validate stage-local tool policy values that affect tool schemas."""

    if not isinstance(tool_config, dict):
        raise TypeError(f"Workflow stage {stage_id!r} tool_config must be a mapping.")
    inspect_config = tool_config.get("inspect_visual")
    if inspect_config is None:
        return
    if not isinstance(inspect_config, dict):
        raise TypeError(f"Workflow stage {stage_id!r} tool_config.inspect_visual must be a mapping.")

    mode = str(inspect_config.get("mode", "model")).strip()
    if mode not in {"model", "video", "tile"}:
        raise ValueError(
            f"Workflow stage {stage_id!r} tool_config.inspect_visual.mode must be "
            f"'model', 'video', or 'tile', got {mode!r}."
        )

    sample_every_n = inspect_config.get("sample_every_n", "model")
    if str(sample_every_n).strip() != "model":
        if isinstance(sample_every_n, bool):
            raise TypeError(
                f"Workflow stage {stage_id!r} tool_config.inspect_visual.sample_every_n "
                "must be 'model' or a positive integer."
            )
        try:
            parsed_sample_every_n = int(sample_every_n)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Workflow stage {stage_id!r} tool_config.inspect_visual.sample_every_n "
                "must be 'model' or a positive integer."
            ) from exc
        if parsed_sample_every_n <= 0:
            raise ValueError(
                f"Workflow stage {stage_id!r} tool_config.inspect_visual.sample_every_n "
                "must be a positive integer when fixed."
            )

    crop = str(inspect_config.get("crop", "enabled")).strip()
    if crop not in {"enabled", "disabled"}:
        raise ValueError(
            f"Workflow stage {stage_id!r} tool_config.inspect_visual.crop must be "
            f"'enabled' or 'disabled', got {crop!r}."
        )


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
