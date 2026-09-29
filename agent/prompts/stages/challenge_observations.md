# Observation Audit and Challenge Stage

Audit and challenge the candidate observation timeline.

Bootstrap/proposal observations are candidate visual claims, not conclusions.
Assume they may contain serious errors. In particular, adjacent time boundaries
may be far from the real visual transition, intervals may be too coarse, and
some boundaries may be numerically tidy guesses rather than actual task state
changes. Do not start by looking for reasons to agree with the proposal. Your
first job is to challenge each current observation against new visual evidence,
strongly prefer revision when the evidence does not precisely support the
proposal, and confirm only the observations that survive that audit.
Treat bootstrap observations as especially suspect when they resemble an
expected sequence implied by the robot instruction rather than directly verified
visual events.

## Required Audit Decisions

Use `read_observations()` to see the current observations, draft ids, and
coverage diagnostics. Every current observation must become `confirmed` before
this stage can complete.

For every current observation, make one of these audit decisions before calling
any confirmation tool:

- `as-is`: the current frame range, text, and confidence are supported;
- `needs boundary revision`: the start or end frame is too early or too late;
- `needs split`: the observation mixes multiple task-progress phases that
  should be separated;
- `needs merge`: two or more adjacent observations are actually one continuous
  task-progress phase with no meaningful visible boundary between them;
- `wrong text`: the frame range is usable, but the text or confidence is wrong.

Only `as-is` observations may be confirmed. For the other decisions, use
`revise_observations`, use `add_observations` as needed when a split creates
additional intervals, and use `merge_observations` when adjacent observations
should become one interval.

## Required Visual Challenge

Before confirming an observation, request new `inspect_visual` evidence in this
stage. Bootstrap/source assets from earlier stages are not valid confirmation
evidence in this stage, even when the current observation was originally drafted
from them. Each confirmation must cite a new `asset_id` returned by
`inspect_visual` during `challenge_observations`.

For observation `[s, e]`, use `start_frame < s` when possible and
`end_frame > e` when possible; clamp only at the trajectory boundaries. The
audit evidence should let you challenge:

- the visual claim inside the observation interval;
- the start boundary, including evidence just before the interval when earlier
  frames exist;
- the end boundary, including evidence just after the interval when later frames
  exist.

Challenge each observation with new visual evidence, then either confirm it
as-is, revise it, split it, or merge it with adjacent observations.

Do not preserve a proposed boundary because it is numerically tidy. Boundaries
at multiples of 5 or 10 are especially suspicious unless focused visual
evidence shows that the actual state change happens there. Prefer concrete
visual changes such as contact, control acquisition, control completion, target
motion, target state change, or a clear task-progress phase transition.

For a long observation, do not only inspect the two outer boundaries. Also check
whether the interval hides an internal task-progress-relevant transition that
should become its own observation.

For approach, alignment, repositioning, or transition observations, apply the
common task-relevant timeline rules and challenge whether the motion qualifies as
preparation. Generic, ineffective, wrong-target, or wrong-direction motion should
not be confirmed as task-relevant preparation.

Apply the common control-acquisition rules before confirming. Challenge whether
the timeline exposes a separate setup interval, a narrow
acquisition/control-complete boundary suitable for downstream 2-frame progress
credit, later outcome evidence, and any sustained motion or retry phases. Treat
merged setup/acquisition, acquisition/outcome, or failed-retry/successful-retry
intervals as `needs split` unless focused visual evidence shows the phases are
not visually distinguishable. Request focused temporal evidence across suspected
first attempts, reopen/reposition phases, and later successful attempts when that
distinction could change the audit. When the current tool schema allows sampling
control, prefer dense sampling for these short windows.

Clip the request to the valid trajectory range. If the returned evidence uses
sparse sampling, make sure it still visibly checks the observation interior and
both available outside-boundary sides. Use `crop` when the current tool schema
exposes it and the observation depends on a small, subtle, thin, partially
occluded, or visually ambiguous task-relevant entity/control changing state.

Use `confirm_observations` only after the new visual evidence supports the
`as-is` decision for the current observation boundaries, text, and confidence.
Batch multiple `as-is` confirmations together when the same challenge-stage
inspection supports them.

## When To Revise Or Merge

Use `revise_observations` if the current observation has the wrong frame range,
wrong text, wrong confidence, or merges phases that should not be merged.
Use `merge_observations` if adjacent observations split one continuous phase
only because the proposal used a coarse or arbitrary boundary.

Common reasons to revise:

- the proposal boundary is coarse, arbitrary, or numerically tidy with no
  matching visual transition;
- adjacent time boundaries are far from the real visual transition;
- an observation is too coarse and should be narrowed around the actual
  task-progress change, including when it hides an internal key boundary;
- approach/preparation and task-critical action are in the same observation;
- task-critical action and withdrawal/transition are in the same observation;
- setup, acquisition/control-complete boundary, outcome evidence, or sustained
  motion are merged even though the common control-acquisition rules require
  separate intervals;
- a failed first attempt, recovery/repositioning, and later successful attempt
  are merged into one approach or acquisition interval;
- approach, alignment, or repositioning motion is mislabeled as task-relevant
  preparation or generic no-progress motion;
- the text infers task success from arm or gripper motion without visible
  target entity/control state evidence;
- the text treats a simulator-truncated terminal video as partial success only
  because no post-success dwell frames are visible;
- the confidence is too high for the available visual evidence.

Common reasons to merge:

- adjacent observations describe the same preparation, approach, alignment,
  acquisition, motion, outcome, or no-progress phase;
- the only apparent boundary between adjacent observations is an arbitrary
  numerical cut rather than a visible task-progress transition;
- the text changes but the visible task-progress state does not;
- the proposed split would create fake downstream progress credit at a frame
  where no meaningful visual transition occurs.

If one observation should be split, revise the original observation into the
first resulting interval, then use `add_observations` for the remaining intervals.
All revised or newly added observations are draft observations and must be
confirmed with challenge-stage visual evidence before this stage can complete.
When revising or adding split intervals, apply the common observation tool
evidence-coverage rule to every submitted interval.

All merged observations are also draft observations. After calling
`merge_observations`, inspect the merged interval and available outside-boundary
context, then confirm the merged observation only if the new evidence supports
its full frame range, text, and confidence.

## Coverage

The current observations must cover the full trajectory with closed intervals.
No frame gaps are allowed. Adjacent observations may share exactly one boundary
frame, such as `[0, 43]` and `[43, 67]`, when that frame is the best visible
transition anchor. Broader overlaps are not allowed.

Use `read_observations()` after revisions when you need to check the remaining
draft ids, gaps, shared boundaries, or overlaps.

When all observations are confirmed and coverage is complete, call
`complete_stage()`. If completion returns `status="blocked"`, follow the
returned `blocked_items` and `next_required_actions`.
