# Agent A: tracking under delayed SAM results

Code commit: `24c6abb`, branch `s4b-agent-a`. The clip-review changes, including
`tests/test_clip_review_report.py`, were committed to main as `6177865` before
both worktrees were created. Nothing has been pushed.

The worktree uses the shared Python 3.12 environment with `PYTHONPATH=src`.
`.env` and `data/` are symlinks and remain excluded from git.

## Outcome

The implementation remains experimental: it does not recover an identity-verified hidden target in either report clip, and false hazards increase. The spatial-recall gain in clip 1 is seven wrong-person matches and one ambiguous match. Do not treat it as a validated target detection.

Before/after below means paired SAM without/with tracking on the same cached inputs.

| Clip | Spatial recall before → after | Audited target recall after | False hazards before → after | Duplicates before → after |
|---|---:|---:|---:|---:|
| blos_pedestrian_001 | 0.000 → 0.727 | 0.000 | 922 → 1656 | 117 → 137 |
| blos_pedestrian_002 | 0.000 → 0.000 | 0.000 | 638 → 947 | 58 → 93 |

## Implementation

| File | Change |
|---|---|
| `src/coopx/observers/tracker.py` | Per-camera IDs, weighted motion fits, velocity uncertainty, Hungarian assignment with unmatched detections, box and HSV appearance gates. |
| `src/coopx/observers/camera.py` | Publish signed motion estimates; accept asynchronous model completions. An older positive remains evidence even after a newer empty result. |
| `src/coopx/envelope.py` | Optional signed `ObservationMessage.velocity_std_mps`, required because CX1 carried velocity and position uncertainty but no velocity uncertainty. CX2 flag 0x08 appends three float64 values (24 bytes). Absent-extension packets remain byte-identical; older decoders must be upgraded to accept the extension. |
| `src/coopx/eval/fusion.py` | Predict to decision time, grow uncertainty, expire tracks at four seconds, check free space at capture position/time, retain refutation across updates and sensor renaming. Witness local detections remain protected. |
| `src/coopx/eval/runner.py` | Process completions in availability order; keep late measurements from overwriting newer positions. Run the same matrix and generate the fixed latency sweep. Capture-image provenance is used only for evaluation audits. |
| `src/coopx/eval/metrics.py` | Opt-in temporal matching for all precision/false-hazard/duplicate calculations; consecutive-frame interpolation; indexed GT history; identity audits. |
| `src/coopx/eval/report.py` | Generated strict comparisons, 30 sweep rows, identity checks, source diagnostics, attack exposure and ledger summaries. |
| `tests/test_tracker.py`, `tests/test_latency_tolerance.py` | Motion, signatures, out-of-sequence completion, no future velocity inputs, expiry, capture-time contradictions, persistent refutation, temporal scoring and identity regressions. |

`meta/reasoner.py`, `pipeline.py` and `ui/` were not changed. The envelope change
is the small shared-code extension needed to authenticate velocity uncertainty.

`track_id` in a belief is the existing source-local `Observation.object_id`.
Optional belief fields include `velocity_mps`, `velocity_std_mps`,
`predicted_position` (XYZ), `position_std_m`, `capture_time_us`,
`captured_position`, `source_id` and `observation_ref`; `xy` is the predicted XY.
Camera boxes in evaluation traces are unsigned audit provenance and never enter
admission, fusion or the motion model. IDs are not reused by a tracker instance;
a restarted publisher must use a fresh session.

## Verification and reproduction

330 tests passed in 79.265 seconds. Targeted Ruff checks and `git diff --check`
passed. The run's source hashes are in `data/s4b-agent-a/release-source.sha256`.

Run from the Agent A worktree:

```sh
env -u MODEL_API_KEY PYTHONPATH=src /home/deep/meta_hackathon/.venv/bin/python -m unittest discover -s tests -q
env -u MODEL_API_KEY PYTHONPATH=src /home/deep/meta_hackathon/.venv/bin/python -m coopx.eval.runner --out-dir data/s4b-agent-a/reproduce --before-path data/s4b-agent-a/before-runs.json --summary data/s4b-agent-a/reproduced-summary.md
```

The CLI defaults to offline cache replay. Re-scoring existing traces makes no
model calls:

```sh
env -u MODEL_API_KEY PYTHONPATH=src /home/deep/meta_hackathon/.venv/bin/python -m coopx.eval.runner --out-dir data/s4b-agent-a/release --rescore-only
```

## Parameters

The 1.5 m / 3 m match radii and default attack scoring remain unchanged. The
requested sweep uses the same τ = 0, 1, 2, 3, 4 s values for both report clips.
τ > 0 scores a recent position rather than the current position.

The motion fit and association gap are both two seconds: four intervals at the
existing 2 Hz SAM cadence. Measurement standard deviation has a 0.1 m floor
(the existing LiDAR lift already floors XY at 0.2 m); velocity standard deviation
has a 0.1 m/s floor to avoid certainty from a noiseless fit. The four-second
prediction cap leaves a one-second guard inside the existing five-second
freshness limit. The 1.5 m/s² process-noise value is an explicit constant-velocity
assumption, not a calibrated confidence guarantee.

Speed operating limits are 8 m/s for pedestrians, 60 for vehicles and 20 for
cyclists/unknown objects. They are conservative operating assumptions, not
universal physical limits. They were checked on non-target tuning pedestrians.
The visual association gates were rounded above the p95 values from 75 tuning
pairs, excluding pedestrian 53: 2 box diagonals/s (p95 1.767), Hellinger distance
0.7 (p95 0.633), and combined cost 5.5 (p95 5.242). They are fixed across both
report clips. Position innovation is normalized by measurement uncertainty;
box-overlap and appearance terms each have weight 2, and normalized centre
motion weight 1. These gates were not chosen to optimize report-clip recall.

The identity audit has fixed IoU >= 0.25 and a 0.10 separation from its runner-up.
Those are explicit audit acceptance criteria, not changes to scoring or motion.
Ambiguous and unknown identities remain visible and never count as verified
matches to the labelled target.

## Remaining limitations

Velocity is a constant-motion estimate from imperfect lifts. Tracking IDs are
not proof of a person's physical identity: crowded scenes can switch matches.
The capture-time image audit exposes these cases instead of treating a nearby
pedestrian as the target. Cross-camera appearance re-identification is not
implemented. Local processing/queueing performance has not been established as
a live real-time guarantee; replay retains measured SAM service times and the
existing fixed transport/LiDAR delays.

The current clips do not establish the requested hidden-target early-warning
claim. Clip 1's first unambiguous target SAM box is available at 5.042 s, after
the ego first sees it at 4.3 s; its lift is already about 3.062 m wrong at capture.
Earlier candidate boxes are ambiguous with another pedestrian. Clip 2's first
accurate target result is available at 2.811 s (about 2.911 s after the link),
after its annotated hidden interval ends at 1.8 s. Its labelled centre moves only
0.143 m in that interval. Prediction cannot manufacture a semantic seed before
it arrives. Later visibility is censored; no lead time is invented.

The source-data checks and all spatial, identity and τ results are generated in
[summary.md](summary.md). Both report clips remain in the evaluation. The next
blockers are source instance/3D association and initial semantic availability;
they cannot be resolved by increasing the matching radius or relabelling the
scored window.

## Completed evaluation and spend

The full matrix has 168 runs and the fixed sweep has 30 rows. All matrix seeds and scenarios match the captured baseline configuration.

Witness safety flips: 0. False distrust: 0. Attack outcomes, false hazards and duplicates are in the generated summary. Final-tick attack catch is reported alongside in-run exposure so expiry is not confused with verification.

Agent A made no new HTTP inference calls: all study clients were offline and used cached SAM results. New Agent A API spend is $0. The ledger is shared with concurrent work; its total change is not attributable to this agent.

| Shared ledger snapshot | Settled USD | Outstanding USD |
|---|---:|---:|
| Turn start | 0.48827400 | 0.22499625 |
| Completion | 0.62950035 | 0.22499625 |

No reservations were released and no ledger rows were deleted.
