## Progress Audit Summary

**Robot instruction:** "put kitchen spoon on plate"

**Progress mode:** `audit_and_correct_candidate_progress`

**Candidate progress present:** Yes (proposal loaded with 25 assigned frames)

### Observations Used
- **obs_0001** [0, 10]: Gripper approaches carrot (not spoon). Carrot, plate, and spoon stationary. Confirmed via `video_clip_7678a4dcae27` (frames 0-12).
- **obs_0002** [11, 14]: Gripper closes around the carrot. Spoon untouched. Confirmed via `video_clip_21447555c2f4` (frames 9-16).
- **obs_0003** [15, 20]: Gripper lifts and transports the carrot toward the plate. Spoon stationary. Confirmed via `video_clip_28c203593fa2` (frames 13-22).
- **obs_0004** [21, 24]: Gripper places carrot on plate. Spoon never interacted with. Confirmed via `video_clip_c1597dacbebc` (frames 19-24).

### Candidate Progress Mismatches
The candidate proposal incorrectly rewarded wrong-object manipulation:
- **Frames 0-10:** Proposal increased from 117→207 (approach phase). **Correction:** Flat at 0. Robot approached carrot (wrong target), not spoon.
- **Frames 11-14:** Proposal increased from 210→218 (grasp). **Correction:** Flat at 0. Robot grasped carrot (wrong object), not spoon.
- **Frames 15-20:** Proposal decreased from 218→211 (lift/transport). **Correction:** Flat at 0. Robot lifted and transported carrot (wrong object).
- **Frames 21-24:** Proposal jumped from 209→391 (placement). **Correction:** Flat at 0. Robot placed carrot on plate (wrong object). Spoon never touched.

### Progress Edits Applied
Four segments edited to flat at 0:
1. **[0, 10]:** 0→0 — Wrong-target approach (carrot instead of spoon)
2. **[10, 14]:** 0→0 — Wrong-object grasp (carrot)
3. **[14, 20]:** 0→0 — Wrong-object lift and transport (carrot)
4. **[20, 24]:** 0→0 — Wrong-object placement (carrot on plate, spoon untouched)

### Main Reward/Progress Judgment
**Final progress: 0/1000.** The robot completely failed the task by manipulating the wrong object (carrot) throughout the entire trajectory. The spoon was never touched. The entire curve is flat at 0 because no task-relevant state was achieved at any point.
