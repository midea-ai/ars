# Bootstrap Observation Stage

Create a precise, task-progress-oriented visual timeline for the whole
trajectory. The timeline should isolate events that matter for progress, not
merely narrate equal-length chunks of video.

## Inputs

The runner should attach the full trajectory video at stage start when it fits
within `visual.max_attached_frames_per_call`. Use that attached video directly
when it is available. The full-video asset normally has `asset_id="full_video"`
and covers the original frame sequence without harness-side frame dropping,
although the file may have been transcoded for model compatibility.

If the full video is unavailable because it exceeds the configured frame limit,
use `inspect_visual` to request segmented video evidence. Prefer original frame
rate segments with `sample_every_n=1`; increase `sample_every_n` only when a
segment would otherwise exceed the frame limit returned by the tool.

If you request segmented evidence, record observations only for segments you
actually inspected. Do not rely on a failed visual request as evidence.

## Boundary Precision

When `full_video` is attached, treat it as complete frame-level evidence for the
trajectory. Use the best visible integer frame boundaries from the full video;
do not treat task-relevant boundaries as coarse samples.

For task-relevant transitions, identify the tightest visible interval you can:

- first visible contact with the task object or control;
- first controlled object motion;
- stable grasp or loss of grasp;
- object leaving or touching a surface;
- placement, insertion, release, or final stabilization;
- wrong-object interaction, slip, drop, stall, or recovery.

If a key boundary remains ambiguous because the visual evidence is occluded,
very fast, or unclear, request a focused `inspect_visual` window around that
event with `sample_every_n=1` before recording the observation. Use lower
confidence only when the visual evidence itself is ambiguous.

## Required Work

Record a boundary-aware timeline with `add_observations`.

Use `add_observations` for new observations. If an existing observation needs a
different boundary, text, or confidence, use `revise_observations` on that
observation id. Batch multiple additions or revisions together when one visual
inspection supports them.

Each observation must describe one event interval. Tool frame ranges are closed
intervals, so `start_frame` and `end_frame` are both included:

```json
{
  "asset_ids": ["full_video"],
  "start_frame": 0,
  "end_frame": 37,
  "text": "The gripper approaches the target object; the object remains stationary.",
  "confidence": 4
}
```

Adjacent observations may share exactly one boundary frame when that frame is
the best visible transition anchor. For example, `[0, 43]` and `[43, 67]` are
valid neighboring closed intervals. Broader overlaps such as `[0, 48]` and
`[43, 67]` are invalid.

Guidelines:

- Cover the full trajectory from frame `0` through `metadata["num_frames"] - 1`
  when `num_frames` is available.
- Apply the common task-relevant timeline, state-change, terminal-truncation,
  and control-acquisition rules when creating observations.
- Keep observation text concrete and visual. Avoid vague claims such as "looks
  good", "the robot is doing the task", or "probably successful".

After observations are recorded, call `complete_stage()`. The harness will
return `status="blocked"` if the current observations do not cover the full
trajectory or if their closed frame intervals overlap by more than one shared
boundary frame. If completion reports overlap, call `read_observations()` and
use its `data` field to get the current observation ids, then use
`revise_observations` to fix the overlapping boundaries. Single-frame endpoint
sharing is reported as `shared_boundaries`, not as a blocking overlap.
