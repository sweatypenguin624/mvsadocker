# Stage 1 — unique-vehicle detection, tracking and 15-minute counting

Stage 1 answers exactly one question: **how many distinct physical vehicles
crossed the counting line, and in which 15-minute window.** It assigns *no*
vehicle type. Every detection is generic `vehicle`. Stage 2 reads the crops
Stage 1 writes and decides 2-wheeler / 3-wheeler / 4-wheeler.

```
video ──► detect (class-agnostic) ──► fuse boxes ──► track ──┐
                                                             │  pass 1 (streaming)
                                                             ▼
                       15-min counts ◄── count line ◄── resolve identities
                       + crops/vehicle_*/                     pass 2 (post)
```

## Why two passes

Counting is **not** done online. Two of the three ways a vehicle gets
double-counted can only be detected using frames *after* the tracker's
mistake, so an online counter structurally cannot fix them:

| Failure | Where it's fixed | Evidence needed |
|---|---|---|
| Two boxes on one vehicle, one inside the other | `stage1_fusion.py`, per frame | none — visible immediately |
| One vehicle tiled into several boxes (front/door/rear of a bus) | `stage1_identity.py` lockstep | many frames of rigid co-motion |
| One vehicle split into track A then track B by an ID switch | `stage1_identity.py` temporal stitch | the later track must exist first |

The previous pipeline had **none** of these: it keyed vehicles on the raw
tracker id, so every tracker mistake became a counting mistake. A bus
filling the frame carried three simultaneous track ids and was counted
three times.

## The two error directions are handled separately

- **Missing a vehicle is unrecoverable** — nothing downstream can invent it.
  So detection runs at a low confidence floor (0.15), the tracker keeps lost
  tracks alive for 90 frames, and the identity stage only ever *merges*
  identities. It can never delete a vehicle that was detected.
- **Over-detecting is recoverable** — fusion, identity merging and the
  crossing guards clean it up with evidence, rather than by raising
  thresholds and silently losing real vehicles.

## Why fusion alone can't fix the tiled-bus case

Tiled panel boxes barely overlap *each other* — that is what makes them
tiled. So neither IoU nor intersection-over-minimum can group them from a
single frame without also merging genuinely adjacent vehicles in
neighbouring lanes. The honest discriminator is temporal: panels bolted to
one body hold a near-constant centre-to-centre offset for the whole track,
while two vehicles drift apart. See `_lockstep_pairs` and the pair of tests
`test_tiled_panels_are_beyond_per_frame_fusion` /
`test_lockstep_merges_tiled_panels_of_one_bus`.

## Usage

```bash
# 1. Calibrate the counting line once per camera (strongly recommended --
#    without it Stage 1 falls back to a horizontal line at 60% frame height
#    and flags every run as uncalibrated).
env/bin/python scripts/traffic_stage1/calibrate_line.py \
    --extract-frame --video path/to/video.mp4
# read the two endpoint pixel coords off the frame, then:
env/bin/python scripts/traffic_stage1/calibrate_line.py \
    --set-line --camera-key cam_1_eb --line 120,700,1800,700 --reference-size 1920,1080

# 2. Run Stage 1.
env/bin/python scripts/traffic_stage1/run.py \
    --video videos/actual_test/08.00.00-09.00.00.mp4 \
    --start-time 2026-07-20T08:00:00 \
    --camera-key cam_1_eb \
    --output results/stage1_run
```

`--start-time` sets the wall clock of frame 0 and therefore every 15-minute
window label; it is inferred from Dahua-style paths
(`<YYYY-MM-DD>/08.00.00-09.00.00.mp4`) when not given.

## Output contract for Stage 2

```
results/stage1_run/
├── stage1_summary.json     counts per 15-min window, diagnostics
├── vehicles.jsonl          one line per counted vehicle
└── crops/
    └── vehicle_000001/     ONE folder per physical vehicle
        ├── best.jpg        highest-quality view
        ├── crop_02..05.jpg
        └── metadata.json
```

Stage 2 should classify the crops in each `vehicle_*/` folder and vote
across them, then join back to `vehicles.jsonl` on `vehicle_id` to turn the
per-window totals into per-window 2W/3W/4W counts. Only vehicles that were
actually counted get a crop folder, so every folder is exactly one vehicle
in exactly one 15-minute window.

## Tuning

All knobs live in `config/stage1_config.yaml`, with per-camera overrides
under `cameras:` selected by `--camera-key`. The two worth revisiting first
on new footage:

- `detector.conf` — lower it if vehicles are being missed entirely.
- `identity.lockstep_max_offset_drift_frac` — raise it if large vehicles
  still split into multiple counts; lower it if distinct vehicles travelling
  together are being merged.

Every merge is recorded with its reason in `vehicles.jsonl`
(`merge_reasons`), so an over- or under-merge can be traced to the rule that
caused it rather than guessed at.

## Tests

`tests/test_stage1.py` — 26 tests covering fusion, identity merging (both
the merges and the merges that must *not* happen), line crossing, jitter
rejection and interval bucketing.
