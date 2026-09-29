# Progress Stage

Audit or generate the progress curve using progress information provided at
stage start and the observation timeline when available.

## Stage Start

Use observations supplied in the stage context or returned by
`read_observations` when they are available.

The runner should also provide an initial `progress_snapshot` and
`progress_summary` for this stage. Use them as the starting point for progress
audit or generation. `read_progress` and `summarize_progress` remain available
for local rechecks, especially after edits.

In this stage's Case context, `progress_state` is only a compact status summary.
The runner-provided `progress_snapshot` contains the frame-level progress rows.

`progress_snapshot` tells you whether candidate progress rows are present:

- If `has_progress_proposal=true`, your mode is `audit_and_correct_candidate_progress`. Verify the proposal and edit unsupported values.
- If `has_progress_proposal=false`, your mode is `generate_progress_from_visual_evidence`. You must assign all frames before export.

Treat proposal values as candidates to verify, not as ground truth.

## Internal Plan

Before editing, form a short internal plan:

- What robot instruction was sent?
- Which available observation intervals mark task-relevant state changes?
- Which candidate progress jumps or suspicious intervals need checking?
- Which progress rows or segments need correction or generation?
- Whether any progress-critical boundary still needs a final visual recheck.

Do not output a long plan unless it helps the final result. Use the plan to
guide tool calls.

## Optional Focused Recheck

When observations are available, use them as the main structure for progress.
You may still call `inspect_visual` for a focused recheck when a candidate jump,
subtle state change, or final completion judgment would materially change the
curve.

When using observations, respect their boundaries. Do not invent a new internal
progress boundary that contradicts the observation timeline unless you have just
inspected focused evidence and the final summary explicitly reports that the
observations were insufficient for that finer split. When in doubt, keep
progress edits aligned with observation intervals.

For subtle target-entity/control changes, an uncropped full-frame clip alone is
not sufficient evidence for a high-confidence major progress increase. If
available observations do not already include focused local evidence, request a
focused `inspect_visual` asset using only the arguments exposed in this stage's
tool schema before assigning that major increase.

Follow the common Visual Inspection Tool schema rules. In this stage, when the
schema exposes `mode`, prefer `mode="video"` for candidate jumps whose timing
matters and `mode="tile"` for before/during/after state comparison. Use `crop`
only if the current schema exposes it.

When the observation supports a progress increase, explicitly describe the
task-relevant entity/control state before and after the interval. Do not infer
success from gripper or arm motion alone. If the target entity/control state
change is not directly visible after focused inspection, lower confidence and
avoid claiming high progress for that state change.

Avoid vague observations such as:

- "Looks good."
- "The robot is doing the task."
- "Probably successful."
- "The reward seems right."

Do not call observation-writing tools in this stage; they are not available.
If the observation timeline is unavailable or insufficient, use focused visual
evidence where needed rather than inventing unsupported visual facts.

## Candidate Evaluation

When candidate progress is present, compare it against visual evidence:

- If candidate values increase during task-relevant preparation, the increase
  should be modest. Generic, ineffective, or wrong-target preparation should
  stay flat; preparation that damages an already achieved task relation may
  justify a small regression.
- Large increases should align with visible task-relevant state changes, not
  merely with robot motion or with a state that was already achieved before the
  interval.
- For subtle entity/control changes, require direct evidence that the
  task-relevant entity/control itself changed state. Hand, gripper, or arm
  motion near the target is not enough by itself.
- A major increase for a subtle entity/control change requires supporting
  focused local visual evidence. Without that asset, reduce confidence
  and keep the progress increase modest even if the full-frame video appears
  plausible.
- A high final score should require visible completion, with stability judged
  from available evidence. For simulator-truncated terminal videos, a visibly
  achieved final task condition at the last frames can justify high final
  progress even without extra post-success dwell frames.
- Regressing progress should be assigned when the robot manipulates the wrong object, loses control, stalls, or fails to change the required task-relevant state. Use a plateau only when the interval preserves the same useful task-relevant state and does not move the robot farther from the required next subgoal.
- Do not use tiny positive deltas to acknowledge effort on a wrong object; wrong-object manipulation is flat at best and usually regressing.
- Repeated same attempts, unusually long approach, unusually long holding, or
  visible stalling without a new task-relevant state improvement should be
  treated as local regression, especially when the robot is no closer to a
  useful target relation than before the interval.
- Sparse candidate annotations should be judged by their stated frame ranges or nearest visible intervals.

Do not collapse all progress to zero merely because the final task fails.
Modest progress may be appropriate for task-relevant preparation.
Keep preparation-phase values low and conservative because the central task
state has not changed yet.

The configured progress value range is
`[{{progress_value_min_points}}, {{progress_value_max_points}}]`. Avoid
over-interpreting tiny numeric differences. Focus on relative shape, temporal
alignment, and support from visible events.

Use `summarize_progress` to recheck candidate jumps after edits or when the
initial summary is insufficient. Use `read_progress(..., include_edit_spans=true)`
after edits to verify that the current curve matches your intended corrections.

## Subgoal Progress Scale

For sequential robot instructions, identify the visible, verifiable subgoals
implied by the instruction before assigning high progress. If the instruction
contains $N$ such subgoals, then the progress value at the moment the $n$-th
subgoal becomes visibly completed should not exceed $n/N$ of the configured
progress range.

After the $n$-th subgoal is completed but before the $(n+1)$-th subgoal is
completed, progress may increase for visible preparation, approach, or attempts
toward the next subgoal, but it should remain below the $(n+1)/N$ completion
level. Treat the interval $[n/N, (n+1)/N)$ as "the first $n$ subgoals are
complete, and the trajectory is attempting the next subgoal."

## Reward/Progress Judgment Guidance

Base the judgment on task-relevant state transitions:

- task-relevant preparation without object/control state change: modest progress
  or weak positive reward at most;
- generic, ineffective, wrong-target, wrong-direction, repeated, or unusually
  slow preparation: local regression unless it is a brief reasonable hold that
  visibly preserves an already useful task relation;
- stable contact but no control: still conservative;
- transition from no reliable control to stable control, followed by a first
  task-relevant state change such as lift-off, separation, actuation, or initial
  displacement: a substantial local progress increase can be justified;
- sustained carrying, transport, adjustment, or continued controlled motion
  after control is already established: progress should usually increase
  gradually according to visible advancement toward the target relation or final
  state;
- correct placement, insertion, or final configuration: high progress can be
  justified if stable in the available evidence or visibly terminally satisfied
  at the video endpoint;
- wrong object, wrong direction, drop, or unstable final state: assign a local progress decrease for the affected interval; final progress should reflect only the strongest required task state still preserved after the failure.

Do not reward time passing, camera motion, or visually impressive robot movement
unless it advances the task.

For deformable-object tasks, apply the common deformable-object-state rule
before assigning high progress or final completion. A bunched, twisted,
wrongly folded, partially spread, or otherwise task-mismatched object should
stay below completion-level progress even if it has been manipulated.

Apply the common control-acquisition rules when assigning progress for brief
control, constraint, attachment, locking, or support events. Treat verified
acquisition as a temporally localized event, not as generic preparation or later
transport. Assign progress with these rules:

- Setup, approach, pose adjustment, and alignment should receive gradual,
  modest progress only when they visibly improve readiness for the target
  entity or control.
- The brief acquisition action is a task-critical attempt. Do not give the main
  acquisition jump merely because the gripper closes, contact occurs, a tool
  touches, or the target becomes occluded.
- Use later outcome frames to judge whether the attempt actually established
  control or constraint, then assign the main progress jump retrospectively to
  the earliest supportable acquisition/control-complete boundary. Do not move
  the main acquisition credit to a later lift, pull, carry, transport, or first
  obvious target-motion frame merely because that frame verifies success more
  clearly.
- The 2-frame acquisition interval should be its own progress edit with
  `end_frame = start_frame + 1`. Choose the best visible boundary near the
  completion of closure, locking, attachment, or equivalent acquisition. Choose
  `[k, k+1]` at the first obvious target-motion frame only when no earlier
  control-complete boundary can be supported after focused inspection, and state
  that uncertainty in the edit reason.
- Treat this as a hard editing constraint for major acquisition credit. If an
  acquisition-related edit changes progress by at least
  `{{progress_default_jump_threshold_points}}` points, that edit must be the
  tight 2-frame acquisition edit. A longer edit whose reason is "grasp",
  "acquisition", "control established", "locked", "attached", or equivalent
  should be only setup/preparation or post-acquisition gradual motion, not the
  main acquisition jump.
- If the observation timeline merges setup and acquisition, do not use the whole
  observation as the acquisition edit. Use the available evidence and, if
  needed, one focused `inspect_visual` request to locate the best 2-frame
  acquisition boundary. If no boundary can be supported, keep the
  acquisition credit below the major-jump threshold and state the uncertainty
  instead of smearing a large jump over the merged interval.
- After verified acquisition, immediate lift-off, separation, pull-away, height,
  transport, carrying, or other sustained controlled motion should receive only
  additional gradual progress for the visible advancement that happens after
  control has already been acquired.
- If later outcome evidence refutes the attempt, such as an empty closed
  gripper, slip, drop, object left behind, unlocked relation, failed attachment,
  or visible need to reopen, retry, or recover, do not treat the acquisition
  attempt as successful progress. Keep it flat or assign a local regression
  according to the remaining readiness and target-entity state.
- If the later outcome is absent or still ambiguous, keep acquisition credit
  conservative and describe the uncertainty rather than forcing a high jump.

For ineffective task-critical attempts, compare both the target entity/control
state and the robot-target readiness relation before and after the attempt. If
the robot attempts a task-critical action such as grasping, pushing, pulling,
inserting, placing, pressing, switching, or actuating, but fails to produce the
intended task-relevant state change, do not treat the attempt itself as
progress. If the failed attempt leaves the robot less ready than before, such as
an empty closed gripper, missed contact, lost alignment, overshoot, unstable
contact, or a visible need to reopen, reposition, or recover, assign a mild
local regression with `end_progress_points` lower than
`start_progress_points`. If the failed attempt also worsens the target
entity/control state, such as moving it away from the goal, disturbing
alignment, causing slip or drop, reversing actuation, destabilizing placement,
or breaking a completed subgoal, assign a stronger local regression. Keep the
curve flat only when the failed attempt visibly preserves the same or better
useful readiness relation as before. Post-failure progress should reflect the
strongest task-relevant state still visibly preserved after the attempt, not the
robot's effort.

For next-subgoal preparation in sequential tasks, keep the increase smaller
than the progress assigned to actually contacting, controlling, transporting,
placing, inserting, actuating, releasing, or stabilizing the next task-relevant
entity/control. Do not give next-subgoal preparation progress when the available
evidence says the previous subgoal is not yet stable or the motion breaks a
previously achieved task relation.

For small or subtle state changes, base the progress judgment on the target
entity/control itself: compare its visible pose, configuration, contact/support
relation, or actuation state before and after the interval. If the target state
change cannot be seen clearly, do not assign a high-confidence major increase
even if the gripper motion looks plausible.

For those small or subtle state changes, "seen clearly" means the relevant
local region has been inspected with `crop`; a full-frame
clip by itself is not enough to justify a high-confidence major increase.

Do not assign the main progress increase for a prior subgoal to a later
withdrawal, release, lift-away, or transition-to-next-target interval. Judge
only the new task-relevant relation that improves inside the edited interval.
Such post-action transition intervals should usually be plateaus or modest
preparation increases unless the withdrawal or release itself visibly creates
or stabilizes the required task state.

For failure cases, do not stretch the interpretation toward success. State the
strongest visible achievement and where the trajectory stops making
task-relevant progress. A failed trajectory can still have a nonzero curve if it
visibly reaches an approach, alignment, contact, partial manipulation, or other
task-relevant intermediate state.

## Progress Editing And Generation

Use `edit_progress_segments` for local event-timed progress edits and
`calibrate_progress_curve` for trajectory-level affine calibration from two
explicit frame/value anchors.

`edit_progress_segments` applies one or more local segment edits in a single
call. Each segment linearly interpolates between two typed endpoint values:
`start_progress_points` at `start_frame` and `end_progress_points` at
`end_frame`. Use this to capture gradual, fine-grained increases or decreases
across visually meaningful intervals, including small approach improvements,
alignment changes, controlled motion, brief regressions, slips, or recovery.
Inside each segment, do not pass JSON arrays or strings for progress values.

When building adjacent piecewise segments, share the boundary frame instead of
leaving a one-frame gap between segment endpoints. For example, use `[0, 43]`
and `[43, 67]`, not `[0, 42]` and `[43, 67]`, when frame `43` is the semantic
anchor between the two progress trends. The shared boundary frame should have
the same progress value in both adjacent segment entries. Segments in one batch
must already be ordered and may not overlap beyond that single shared frame.

For a plateau, set `start_progress_points` equal to `end_progress_points`. For a
small regression, set `end_progress_points` lower than `start_progress_points`.
The returned edit record automatically reports the segment `trend` as
`increase`, `flat`, or `decrease` from those endpoint values; do not pass
`trend` as an input.

For candidate proposals:

- Keep proposal values where visible evidence supports them.
- Edit local unsupported intervals instead of rewriting the whole curve by default.
- `edit_progress_segments` only supports local `propagation="none"` edits. If a later baseline is wrong, explicitly rewrite the affected later intervals.
- Prefer evidence-grounded segment edits over per-frame micromanagement.

For no-proposal cases:

- Generate a complete current progress curve from visual evidence.
- Use broad piecewise segments: low/flat before task-relevant behavior, modest approach or preparation progress, stronger increases around key task-relevant achievements, and high values only for visibly complete terminal task states.
- All frames must be assigned before export.

## Affine Calibration Guidance

Use `calibrate_progress_curve` when the current curve's relative trend is
credible, but two key frames should correspond to clearer semantic progress
values. Typical anchors include the start frame mapping to 0, a frame where the
$n$-th subgoal becomes completed mapping to at most $n/N$ of the configured
range, or a failure endpoint mapping back to the strongest subgoal state still
visibly preserved.

The tool applies one affine transform to assigned current-progress rows only.
Unassigned rows remain unassigned. Both anchor frames must already be assigned,
and the tool blocks without changing the curve if the anchors would require a
negative scale or if any transformed assigned row would leave the configured
progress range.

Do not use `calibrate_progress_curve` when a local peak, jump, plateau, or
regression is placed at the wrong frame; when wrong-object manipulation,
failed attempts, empty grasps, drops, or slips are incorrectly rewarded; or
when the relative trend between the anchors is not credible. Fix those issues
first with `edit_progress_segments`.

Every `edit_progress_segments` segment must include:

- `start_progress_points` and `end_progress_points`, integers in
  `[{{progress_value_min_points}}, {{progress_value_max_points}}]`;
- a concrete evidence-grounded reason;
- the relevant closed frame range;
- a confidence value in `[1, 5]`.

Every `calibrate_progress_curve` call must include two assigned frame/value
anchors, a concrete evidence-grounded reason, and a confidence value in
`[1, 5]`. Reasons may refer to observation intervals, focused visual
assets, or visible events. They do not need to mechanically cite an observation
id.

Do not set final progress near `{{progress_value_max_points}}` unless the
final task condition is visibly complete and either stable in the available
frames or terminally satisfied at the video endpoint. Lack of additional frames
after a visible terminal success should not be treated as partial success by
itself. If the trajectory only reaches an intermediate state and then fails,
final progress should reflect that strongest visible intermediate achievement
rather than dropping automatically to zero or stretching toward completion.
Validation treats a final progress value below
`{{progress_value_max_points}}` as an incomplete terminal curve.

Once available evidence establishes the main task outcome, make progress edits
promptly. Do not continue requesting visual evidence if existing evidence
already supports the needed correction or generated segment.

## When To Stop Inspecting

You usually have enough evidence when:

- the robot instruction is clear;
- available observations, if any, identify the main trajectory structure;
- focused clips or tiles, when needed, cover any remaining key success/failure transitions;
- progress values have been corrected or generated with `edit_progress_segments` and any needed `calibrate_progress_curve`;
- all progress rows are assigned;
- `validate_progress()` returns `ok=true`;
- a final summary for the validated progress curve has been written with `write_summary`;
- remaining uncertainty would not change the main conclusion.

If uncertainty would change the conclusion, request one more focused visual
artifact before finalizing.

## Finalization

Required final tool order:

```text
edit_progress_segments(...)  # one call with one or more segments, unless the proposal is fully supported
calibrate_progress_curve(...)  # optional, only when two semantic anchors justify a global affine calibration
validate_progress()
write_summary(...)
complete_stage()
```

If the proposal is fully supported and no edit is needed, still call
`validate_progress()`, `write_summary()`, and `complete_stage()`.

Only call `write_summary()` after `validate_progress()` returns `ok=true`.
If `validate_progress()` returns `ok=false` with `status="blocked"`, follow the
returned `blocked_items` and `next_required_actions`, fix the curve with
`edit_progress_segments` or `calibrate_progress_curve` as appropriate, and call
`validate_progress()` again before writing the summary.
A blocked `write_summary()` means the summary was not accepted; resolve the
reported issue, and retry.

The `summary` passed to `write_summary` should be Markdown and include:

- robot instruction;
- progress mode: `audit_and_correct_candidate_progress` or `generate_progress_from_visual_evidence`;
- whether candidate progress was present;
- observations, if any, and any focused visual evidence inspected, with `asset_id` and closed frame ranges;
- progress edits or generated segments, with closed frame ranges;
- main reward/progress judgment;
- evidence supporting that judgment;
- candidate mismatches, if any.

You may have made mistakes even if your analysis seems convincing. Use the
`external_review_need` passed to `write_summary` to rate how much this result
needs independent external review, based on the limits of your own ability and
how difficult the video was for you to understand. Do not base the rating on
whether the trajectory succeeded or failed. `1` means no external review is
needed; `5` means external review is strongly needed.

If review is needed, use `external_review_focus` to state briefly what the
reviewer should check. Otherwise omit `external_review_focus`.

Use `candidate_supported: null` when no candidate progress proposal is present
or when the proposal cannot be evaluated from the available evidence.

Use `candidate_progress_present: false` and
`progress_mode: "generate_progress_from_visual_evidence"` when no candidate
progress proposal is present.

Call `complete_stage()` after `write_summary()`. If it returns `ok=false` with
`status="blocked"`, follow the returned `blocked_items` and
`next_required_actions`, fix the curve or rewrite the summary as requested, and
call the relevant validation/finalization tool again.
