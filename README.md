# MVSA -- Pedestrian Survey + Demographic Estimation Pipeline

MVSA runs on the University of Oregon research server to count pedestrians
entering a fixed ROI in CCTV footage (YOLO11x + ByteTrack), and optionally
estimate age/gender for each counted pedestrian via two independent
demographic stages: a face stage (InsightFace) and a whole-body/clothing
stage (a PA-100K-trained pedestrian-attribute model). See Section 19 for
why there are two.

## 1. Architecture

```
DAV file
  |  ffmpeg -c:v copy (remux, no re-encode)     [video_utils.py]
  v
MP4 (original HEVC stream)
  |  cv2.VideoCapture, one frame at a time       [video_utils.py]
  v
YOLO11x + ByteTrack (device=0, imgsz=1280, conf=0.35, class 0 = person)
                                                  [detector.py]
  v
per-frame Detection list (bbox, conf, class, track_id)
  |
  +--> PedestrianTracker: bottom-center of bbox vs ROI polygon
  |    track_id counted exactly once, on first ROI entry  [tracker.py, roi.py]
  |
  +--> (optional, disabled by default) RiderFilter: person bbox vs
  |    vehicle bboxes -> pedestrian / rider / uncertain    [rider_filter.py]
  |
  +--> once a track is counted, every Nth visible frame, BOTH of:
       |
       +--> head/upper-body crop -> InsightFace face detection + genderage
       |      -> FaceQualityAssessor (size/blur/brightness/det-confidence
       |         gate) -> DemographicAnalyzer.analyze()
       |      -> FaceSample(source="face")        [demographics.py,
       |                                            face_quality.py]
       |
       +--> full-body crop -> PP-LCNet person-attribute model (ONNX)
              -> BodyAttributeAnalyzer.analyze()
              -> FaceSample(source="body")         [body_attributes.py]
  v
TrackDemographicAggregator: one record per track_id, combining both
  sources' samples in a single weighted vote (gender AND age_group;
  numeric age_estimate comes from face samples only)  [aggregation.py]
  v
OutputWriter: tracked.mp4, hourly_pedestrian_counts.csv, track_demographics.csv,
  hourly_demographics.csv, tracks.csv, track_positions.json, run_summary.json
                                                              [output.py]
  ^
  |  every checkpoint.interval_frames, atomic JSON snapshot  [checkpoint.py]
```

Key properties:

- **The ROI polygon is defined exactly once**, in `config/config.yaml`, and
  read everywhere via `scripts/roi.py`. No file hard-codes ROI coordinates.
- **Demographics is a second stage, independent of YOLO**, and now has two
  independent estimators within it. `demographics.py` and
  `body_attributes.py` each only need a BGR person crop; swapping either
  model later means editing one file.
- **No face images or embeddings are saved by default.** Only per-track
  aggregate numbers/text reach disk. Debug crops are strictly opt-in and
  capped (`output.save_debug_face_crops`, off by default).
- **Confidence concepts are kept separate and never conflated**: YOLO
  detection confidence, ByteTrack's (non-numeric) track continuity,
  InsightFace's face-detection `det_score`, the heuristic face-image
  quality score, InsightFace's documented *confidence proxy* (its genderage
  model has no calibrated confidence -- see
  [Section 16](#16-why-agegender-predictions-are-estimates)), and the body
  model's own sigmoid probabilities (which ARE genuine per-class
  probabilities, but calibrated on PA-100K, not this camera -- see
  [Section 19](#19-body-attribute-clothing-based-estimation)). Every
  `FaceSample` records which estimator (`source`) produced it, and the
  demographic-test report breaks predictions down by source, specifically
  so face-based and body-based trust signals are never silently merged
  into one number without that provenance being inspectable.
- **New track IDs are always allowed to count again** if a person leaves and
  re-enters frame -- MVSA does not attempt cross-session re-identification.

## 2. Directory structure

```
mvsa/
├── config/
│   └── config.yaml               # single source of truth for all thresholds/paths
├── scripts/
│   ├── main.py                   # CLI entry point (detect/full/demographics/demographics-test)
│   ├── pipeline.py                # orchestration: wires every module into the frame loop
│   ├── config_loader.py           # YAML -> typed, validated Config dataclasses
│   ├── models.py                  # shared dataclasses + age-bucket / gender constants
│   ├── roi.py                     # ROI polygon, point-in-polygon, bottom-center
│   ├── video_utils.py             # DAV->MP4 remux, ffprobe, streaming frame reader
│   ├── detector.py                # YOLO11x + ByteTrack wrapper
│   ├── tracker.py                 # ROI-entry counting, per-track state
│   ├── rider_filter.py            # optional pedestrian/rider/uncertain heuristic
│   ├── face_quality.py            # face-crop quality gating (size/blur/brightness/conf)
│   ├── demographics.py            # InsightFace-backed age/gender estimation
│   ├── body_attributes.py         # whole-body/clothing age/gender estimation (ONNX)
│   ├── aggregation.py             # per-track demographic aggregation (multi-source vote)
│   ├── checkpoint.py              # atomic JSON checkpoint save/load
│   ├── output.py                  # CSV/video/JSON writers, video annotation drawing
│   ├── utils.py                   # logging setup, weighted stats, env sanity check
│   └── vehicle_counting/          # independent pilot: hourly vehicle counts by class (Section 20)
│       ├── vehicle_classes.py     # requested-category -> stock-YOLO-class mapping (single source of truth)
│       ├── vehicle_config_loader.py
│       ├── vehicle_tracker.py     # per-class ROI-entry counting
│       ├── vehicle_output.py      # CSV/video writers for this pilot
│       ├── vehicle_pipeline.py    # orchestration
│       └── vehicle_counter.py     # CLI entry point
├── tests/
│   ├── conftest.py
│   ├── test_roi.py
│   ├── test_age_buckets.py
│   ├── test_aggregation.py
│   ├── test_config.py
│   ├── test_vehicle_classes.py
│   └── test_vehicle_tracker.py
├── requirements-demographics.txt
├── pytest.ini
└── README.md

# Already present on the server, used as-is:
# env/, models/yolo11x.pt, tools/ffmpeg, tools/ffprobe, videos/, results/
# models/insightface/ is downloaded automatically on first run (Section 5).
# models/person_attribute/person_attribute.onnx is a one-time offline
# conversion -- see Section 5b -- not downloaded automatically.
```

## 3. Installation

The server already has a working `~/mvsa/env` with `torch 2.1.0+cu118`,
`torchvision 0.16.0+cu118`, `numpy 1.26.1`, and a working `ultralytics`.
**Do not `pip install torch`/`torchvision`/`numpy` again.**

```bash
cd ~/mvsa
source env/bin/activate
export PYTHONNOUSERSITE=1   # already set in this project; do not unset it
python -c "import torch, torchvision, numpy, ultralytics, cv2, yaml; \
  print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

That should print `2.1.0+cu118 0.16.0+cu118 1.26.1`. If it doesn't, stop and
fix the base environment before installing anything from this repo.

## 4. Existing PyTorch environment

`scripts/utils.py::verify_core_dependencies()` runs automatically at the
start of every `main.py` invocation and **logs a warning** (never aborts)
if `torch`/`torchvision`/`numpy` don't match the versions above. Treat any
such warning as a signal to check what changed -- most likely something was
installed without `--no-deps`.

## 5. Installing demographic dependencies safely

Do this in order, verifying versions after each step. Read
`requirements-demographics.txt` first -- it explains why each package is
there.

```bash
cd ~/mvsa
source env/bin/activate

# 1. Snapshot current versions so you can detect any drift.
pip show torch torchvision numpy | grep -E "Name|Version"

# 2. Install InsightFace's actual runtime deps WITHOUT letting it also
#    pull in its own transitive requirements (which are unpinned and could
#    drag in a numpy/scipy upgrade).
pip install --no-deps -r requirements-demographics.txt

# 3. Install InsightFace itself, also with --no-deps, so it cannot pull in
#    anything not already covered by step 2.
pip install --no-deps insightface==0.7.3

# 4. Verify nothing moved.
pip show torch torchvision numpy | grep -E "Name|Version"
python -c "import torch, torchvision, numpy; \
  print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

Expected output after step 4 is unchanged from step 1
(`2.1.0+cu118`, `0.16.0+cu118`, `1.26.1`). If `import insightface` raises
`ModuleNotFoundError` for some other pure-Python package (this can happen --
insightface's exact import graph varies slightly by version), install just
that one missing package with `pip install --no-deps <package>` and re-check
versions again. Do not run a bare `pip install insightface` -- that pulls
its full, unpinned dependency tree.

**GPU note:** `demographics.device` defaults to `"cpu"` in config.yaml. Face
crops are small and sampled sparsely (every Nth frame per track), so CPU
inference is normally fast enough and carries zero risk of interacting with
torch's CUDA runtime. Only set `device: "cuda"` and additionally install
`onnxruntime-gpu` (also with `--no-deps`, matching your CUDA setup) after
validating it in isolation -- two separately-installed CUDA-using libraries
in one process is a plausible source of the exact kind of breakage this
section is trying to avoid.

### Downloading the InsightFace model

The first time `demographics`/`full`/`demographics-test` mode runs,
InsightFace downloads the `buffalo_l` model pack (detection + genderage
ONNX models, ~300MB) to `config.demographics.insightface_root`
(default: `~/mvsa/models/insightface`). This requires outbound internet
access from the compute node. If the compute node has no internet access:

1. On a machine with internet access, run any short Python snippet that
   constructs `insightface.app.FaceAnalysis(name="buffalo_l")` once, or
   download directly:
   ```bash
   curl -L -o buffalo_l.zip \
     https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   ```
2. Copy/unzip it to `~/mvsa/models/insightface/models/buffalo_l/` on the
   server, so the `.onnx` files land directly in that directory.
3. Re-run the pipeline; no download will be attempted if the files are
   already present.

**The second demographic model (body/clothing attributes) is a separate,
one-time setup -- see Section 19.3** -- do that before your first `full` or
`demographics-test` run too, or set `body_attributes.enabled: false` in
config.yaml to run without it.

## 6. Configuring the ROI

Edit `config/config.yaml`'s `roi:` section. `polygon` is a list of `[x, y]`
points in the coordinate space of `reference_width` x `reference_height`
(native camera resolution, 1920x1080 by default). If you ever process video
at a different resolution, the polygon is automatically scaled -- you do not
need multiple polygons.

ROI membership is always tested against the **bottom-center ("feet") point**
of the YOLO person bounding box, never the box center -- see
`scripts/roi.py::bottom_center`.

## 7. Running detection

```bash
cd ~/mvsa
source env/bin/activate
python scripts/main.py detect --config config/config.yaml --run-name actual_08-09
```

Outputs land in `results/actual_08-09/`:
`tracked.mp4`, `hourly_pedestrian_counts.csv`, `tracks.csv`,
`track_positions.json`, `run_summary.json`, `checkpoint.json`, `logs/run.log`.

## 8. Running the demographic test (do this first)

Before processing full videos, validate whether this specific camera's
footage actually supports age/gender estimation:

```bash
python scripts/main.py demographics-test --config config/config.yaml \
  --run-name demo_test_08-09
```

This runs the full pipeline but stops shortly after
`demographics_test.max_tracks` (default 20) pedestrians have been counted
(plus `tail_frames` extra frames so those tracks accumulate more than one
sample). It additionally writes `demographic_test_report.csv` with one row
per test track: frame count, face-detection count, usable-sample count,
every gender/age prediction made, the final aggregated result, and quality
statistics. Use this to tune `face_quality` thresholds in config.yaml before
trusting results at scale -- e.g. if `num_usable_face_samples` is 0 for
nearly every track, the camera angle/resolution is likely too poor for this
demographic stage and thresholds (or camera placement) need revisiting.

## 9. Running the full demographic pipeline

Once the demographic test looks reasonable:

```bash
python scripts/main.py full --config config/config.yaml --run-name actual_08-09
```

This does detection, tracking, ROI counting, and inline demographics in one
pass over the video, writing every output file listed in Section 2.

Alternatively, run `detect` first and `demographics` afterward as a second
pass (no re-running of YOLO -- it re-derives person crops on the fly from
the saved bbox coordinates in `track_positions.json`, never from saved
images):

```bash
python scripts/main.py detect --run-name actual_08-09
python scripts/main.py demographics --run-name actual_08-09
```

## 10. Output CSV formats

**hourly_pedestrian_counts.csv**
| hour | pedestrian_entries |

**tracks.csv** (per-track metadata from `detect`/`full`)
| track_id | first_seen_frame | first_seen_time | last_seen_frame | last_seen_time | entered_roi | entered_roi_frame | rider_status |

**track_demographics.csv**
| track_id | first_seen | last_seen | gender | gender_confidence | age_estimate | age_group | age_confidence | valid_face_samples | total_face_attempts | average_face_quality | status |

Despite the column names (kept stable rather than renamed, since this file
predates the body estimator), `valid_face_samples`/`total_face_attempts`/
`average_face_quality` now count samples from **both** estimators combined
(source="face" and source="body" -- Section 19); `gender` and `age_group`
are each a single confidence-weighted vote across whichever samples a
track has. `age_estimate` is the exception: it is numeric and can only ever
come from face samples, so it is `None` for any track whose classification
came only from the body estimator. `status` is one of: `valid`,
`insufficient_face_quality`, `no_face_detected`, `not_sampled` -- computed
across both estimators (see Section 17's closing note).
`gender_confidence`/`age_confidence` are the documented proxy score
described in Section 16 (face) and Section 19.4 (body), not a model-native
probability of correctness. To see which estimator actually produced a
given track's result, use `demographic_test_report.csv` below (from
`demographics-test` mode) rather than this file.

**hourly_demographics.csv**
| hour | pedestrian_entries | male | female | unknown_gender | age_0_18 | age_18_60 | age_60_plus | unknown_age |

Per-category counts can legitimately sum to less than `pedestrian_entries`
for that hour -- unusable samples are never silently dropped, they show up
as `unknown_gender`/`unknown_age`.

**demographic_test_report.csv** (demographics-test mode only)
| track_id | num_frames | num_face_attempts | num_face_usable | face_gender_predictions | face_age_predictions | num_body_attempts | num_body_usable | body_gender_predictions | body_age_group_predictions | final_gender | final_age | final_age_group | status | quality_min | quality_max | quality_mean |

`final_gender`/`final_age`/`final_age_group`/`status` are computed via the
same `TrackDemographicAggregator` used for `track_demographics.csv` (not a
separate hand-rolled vote), so this report can never disagree with that
file -- it only adds the raw per-source breakdown of what fed the vote.
Use the `num_face_usable` vs `num_body_usable` columns to see, per track,
whether a `valid` result actually came from a resolved face or only from
the body/clothing estimator.

## 11. Inspecting annotated videos

`results/<run-name>/tracked.mp4` shows: the ROI polygon, each person's
bounding box + `ID <track_id>`, gender/age_group text (only once the
*latest usable estimate for that track* -- face if usable that frame, else
body -- clears `output.show_demographics_min_quality`; unknown results are
never drawn), a timestamp, and the running pedestrian count. When
`rider_filter.enabled` and `exclude_riders_from_count` are both true, this
counter is labeled "Pedestrian count (riders excluded, live estimate)" and
excludes a track the instant it's classified as a rider at its ROI-entry
frame -- a live approximation of the final count, since the authoritative
exclude-riders decision (a majority vote over each track's whole visible
history) can't be known until a track is finished being seen. It will
occasionally disagree in edge cases with the exact numbers in
`hourly_pedestrian_counts.csv`/`tracks.csv`, which remain the source of
truth. The demographic label doesn't indicate which estimator produced it --
cross-reference
`demographic_test_report.csv` (Section 10) if that matters for a specific
track. Play it with any standard player (`ffplay results/<run>/tracked.mp4`,
or copy it off the server).

## 12. Resuming interrupted jobs

```bash
python scripts/main.py full --run-name actual_08-09 --resume
```

`detect`, `full`, and `demographics-test` all support `--resume`, reading
`results/<run-name>/checkpoint.json` (saved every
`checkpoint.interval_frames` frames, written atomically so a killed process
can never leave a corrupt checkpoint). Resume restores per-track state,
counted-track IDs, and accumulated face samples, then continues from the
next frame.

**Known limitations of resume:**
- New ByteTrack IDs assigned after a resume start at 1 again internally;
  the pipeline offsets them past the highest pre-resume track ID so they
  cannot collide with (and silently merge into) an existing track. Track
  IDs are therefore not perfectly stable across a resume, but counts and
  demographics remain correct.
- OpenCV's `VideoWriter` cannot append to an existing MP4. **Annotated
  video writing is automatically disabled for resumed runs** (a warning is
  logged); all CSV/JSON outputs are unaffected. If you need a complete
  annotated video for an interrupted run, re-run without `--resume` (this
  reprocesses from frame 0).

## 13. GPU usage

The pipeline never hard-codes a GPU UUID. Select the GPU externally:

```bash
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python scripts/main.py full --run-name actual_08-09
```

`config.model.device: "0"` then refers to that GPU as PyTorch's `cuda:0`.
The demographic stage defaults to CPU (`config.demographics.device: "cpu"`)
regardless of `CUDA_VISIBLE_DEVICES` -- see Section 5's GPU note before
changing this.

## 14. Performance considerations

- Frames are streamed one at a time (`cv2.VideoCapture`); the full video is
  never loaded into memory.
- Demographic sampling only happens for **counted** tracks, and only every
  `demographics.sample_interval`-th visible frame -- not every frame of
  every detection.
- `demographics` mode's second pass still decodes every video frame
  sequentially (required for correctness on HEVC streams without frequent
  keyframes), but skips YOLO inference and InsightFace inference entirely
  except on the specific frames a sample is due, so it is materially
  cheaper than a `full` run when you only need to re-tune demographic
  thresholds without re-detecting pedestrians.
- An A100 80GB comfortably runs YOLO11x at `imgsz=1280` well above 20 FPS,
  so the pipeline is not GPU-bound at this frame rate; the streaming
  frame-read and (CPU) InsightFace calls are more likely bottlenecks --
  profile before assuming GPU is the constraint.

## 15. Known limitations

- **Rider filtering is a geometric heuristic**, not a trained classifier.
  It measures bbox overlap between a person and nearby detected vehicles
  and reports `pedestrian` / `rider` / `uncertain` -- it does not, and does
  not claim to, perfectly separate riders from pedestrians. It is disabled
  by default (`rider_filter.enabled: false`); validate it against this
  camera's footage (spot-check `tracks.csv`'s `rider_status` column against
  `tracked.mp4`) before enabling `exclude_riders_from_count`.
- **No cross-session re-identification.** If a person leaves the camera's
  view and re-enters later, ByteTrack normally assigns a new track ID, and
  MVSA counts it again by design. This means `pedestrian_entries` measures
  ROI-entry events, not unique individuals, if someone loiters in and out
  of frame.
- **Seeking on HEVC streams remuxed with `-c:v copy`** is only
  keyframe-accurate at the container level; OpenCV compensates by decoding
  forward from the nearest keyframe, which is correct but can make
  large `--resume` seeks slower than sequential playback.

## 16. Why age/gender predictions are estimates

InsightFace's `genderage` attribute model (part of the `buffalo_l` pack)
outputs a hard gender class and a scalar age value from a single forward
pass -- it does not expose a calibrated per-prediction probability the way
a detector's objectness score works. Reporting YOLO's detection confidence,
or the fact that a face was found at all, as if it were age/gender
confidence would be actively misleading. Instead, `demographics.py`
computes and clearly labels a **confidence proxy**:

```
confidence_proxy = 0.5 * face_detection_score + 0.5 * face_image_quality_score
```

This is an engineering-defined trust signal used to weight votes in
`aggregation.py` (so a blurry, low-confidence sample influences a track's
final gender/age less than a sharp, high-confidence one) -- it is **not** a
statement that "there is an X% chance this prediction is correct." Treat
`track_demographics.csv`'s `gender_confidence`/`age_confidence` columns
accordingly, and validate final numbers against manual spot-checks of
`tracked.mp4` before using them for any downstream decision.

Gender label mapping (`0 -> female`, `1 -> male`) follows the `buffalo_l`
genderage model's documented output convention; this is a fixed lookup
table, not re-derived from a probability distribution the model doesn't
expose.

## 17. Face-quality limitations on CCTV

CCTV faces are frequently too small, blurry, backlit/underlit, angled away
from the camera, or occluded (helmets, masks, hoods) to support any
reliable age/gender estimate. `face_quality.py` gates on face width/height,
InsightFace's own detection confidence, Laplacian-variance sharpness, and
optionally brightness -- all thresholds are in `config.yaml`'s
`face_quality:` section and are explicitly **not** claimed to be universally
correct; tune them using `demographics-test`'s report and debug crops
(temporarily set `output.save_debug_face_crops: true` while tuning, then
turn it back off). A track's *face* samples specifically get
`status = insufficient_face_quality` (if a face was seen but rejected) or
`no_face_detected` (if InsightFace never found a face at all) if judged in
isolation -- both are reported explicitly rather than silently defaulting
to a guessed value. Since Section 19, a track's *final* status in
`track_demographics.csv` reflects both estimators combined: on this camera
the face path fails for nearly every track (see Section 19), so it is the
body/clothing estimator, not this one, that determines whether most tracks
end up `valid` or `insufficient_face_quality`/`no_face_detected` -- check
`demographic_test_report.csv`'s `num_face_usable` vs `num_body_usable`
columns before assuming a `valid` result came from a resolved face.

## 18. Rider filtering limitations

See Section 15. In addition: the overlap-ratio heuristic in
`rider_filter.py` can be confused by a pedestrian walking closely beside a
parked vehicle, or a cyclist dismounted and walking their bike, in either
direction (false rider or false pedestrian). It is intentionally
architected as a standalone, swappable module (`RiderFilter.classify()`
takes a bbox and a list of same-frame vehicle detections and returns a
result -- nothing else in the pipeline depends on its internals) so it can
be replaced with a trained classifier later without touching
detection/tracking/demographics code.

## 19. Body-attribute (clothing-based) estimation

### 19.1 Why this exists

On this camera (steep overhead angle over a wide intersection, native
1920x1080 spread across the whole ROI) InsightFace's face pipeline resolves
a usable face for only a small fraction of counted tracks -- run
`demographics-test` and look at `demographic_test_report.csv`'s
`num_face_usable` column, which is 0 for most tracks even when dozens of
face *attempts* were made. The face is a handful of pixels; the person's
whole-body silhouette, posture, and clothing are still legible at that same
resolution. `body_attributes.py` runs a pedestrian-attribute model over the
**whole-body** crop (the full YOLO bbox, not just the head region) as a
second, independent estimator that does not depend on a resolved face at
all, and feeds its samples into the same per-track vote in
`aggregation.py` (see Section 1) alongside any face samples that do
resolve.

### 19.2 Model and provenance

The model is PaddleClas's PULC `person_attribute` model
(PPLCNet_x1_0, Apache-2.0), trained on the
[PA-100K](https://www.v7labs.com/open-datasets/pa-100k) pedestrian-attribute
dataset -- real CCTV/surveillance pedestrian imagery, the same domain as
this camera, unlike face datasets. It outputs 26 sigmoid attribute
probabilities per PA-100K's schema (gender, 3-way age bucket, clothing
style/color, bags, hat, glasses, pose, boots); MVSA only uses the gender
(index 22) and age-bucket (indices 19-21, argmax) values, decoded with the
exact rule from PaddleClas's own postprocessing code
(`ppcls/data/postprocess/attr_rec.py::PersonAttribute`) -- see
`body_attributes.py`'s module docstring for the verified index layout.
Before being trusted, the converted model's predictions were checked
against PaddleClas's own published demo outputs for two reference images
(`090004.jpg` -> Male/Age18-60, `090007.jpg` -> Female/Age18-60) and matched
exactly.

**MVSA never runs PaddlePaddle at inference time** -- only an ONNX export
of the model, via `onnxruntime` (the same runtime already used for
InsightFace). PaddlePaddle is only needed once, offline, to convert the
model, and was deliberately kept out of `env/` entirely for that
conversion (see Section 19.3) so it can never interact with `env/`'s
pinned `torch`/`torchvision`/`numpy` versions (Section 3).

### 19.3 Installing/regenerating the ONNX model

`models/person_attribute/person_attribute.onnx` is checked into place as a
build artifact, not regenerated on every run. To reproduce or update it:

```bash
cd ~/mvsa
# A throwaway venv OUTSIDE env/ -- paddlepaddle is only ever needed here,
# for this one-time conversion, never at pipeline runtime.
python3 -m venv .tmp_convert_env
source .tmp_convert_env/bin/activate
pip install paddlepaddle==2.6.2 packaging
# paddle2onnx 2.x hard-requires a paddlepaddle-3.x nightly; 1.3.1 is the
# last release compatible with a stable paddlepaddle 2.x install.
pip install paddle2onnx==1.3.1

mkdir -p models/person_attribute
curl -L -o models/person_attribute/person_attribute_infer.tar \
  https://paddleclas.bj.bcebos.com/models/PULC/person_attribute_infer.tar
tar -xf models/person_attribute/person_attribute_infer.tar -C models/person_attribute/

paddle2onnx \
  --model_dir models/person_attribute/person_attribute_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file models/person_attribute/person_attribute.onnx \
  --opset_version 11   # auto-bumped to 14 for hard_swish; expected

deactivate
rm -rf .tmp_convert_env models/person_attribute/person_attribute_infer*
```

Then re-verify against the two reference images in Section 19.2 before
trusting the new file (resize to 192x256, ImageNet mean/std normalize, RGB,
CHW -- see `body_attributes.py::BodyAttributeAnalyzer._preprocess`; the
ONNX graph's output is already post-sigmoid, do not apply sigmoid again).

### 19.4 Limitations

- **Calibrated on PA-100K, not this camera.** The model's sigmoid outputs
  are genuine per-class probabilities in a way InsightFace's genderage head
  is not (Section 16), but that calibration reflects PA-100K's population
  and camera distribution, not necessarily this intersection's. They are
  used exactly like InsightFace's `confidence_proxy` -- a vote-weighting
  trust signal, not a validated probability of correctness here.
- **Clothing/build are correlated with, not equivalent to, gender/age.**
  This is the same fundamental limitation as a human doing a manual visual
  pedestrian count from CCTV footage at this resolution -- it is
  systematically wrong for anyone whose clothing/build doesn't match the
  training population's typical patterns (e.g. a child in adult-sized
  clothing, gender-nonconforming dress). Spot-check `tracked.mp4` against
  `track_demographics.csv` before treating aggregate numbers as precise.
- **No numeric age.** Unlike InsightFace, this model only ever produces one
  of 3 buckets, never a numeric age -- `track_demographics.csv`'s
  `age_estimate` column is `None` for any track whose bucket came only from
  body samples (see `aggregation.py`); this is intentional, not a bug.
- **Whole-body crop required.** A track counted right at the ROI edge with
  most of its body cut off by the frame boundary will fail
  `body_attributes.min_crop_width`/`min_crop_height` (config.yaml) even if
  YOLO detected it -- these thresholds are deliberately looser than
  `face_quality`'s (the model tolerates blur/low-res far better than a face
  detector does), but a body crop still needs to exist.

## 20. Vehicle counting (pilot)

`scripts/vehicle_counting/` is a separate, independent subsystem for hourly
vehicle counts by class -- not a mode of the pedestrian pipeline above, and
it does not modify or depend on `pipeline.py`, `config.yaml`, or any
demographics code.

### 20.1 Scope: what this pilot can and can't count yet

The eventual requirement is 16 vehicle categories (TAXI, Slow Vehicle,
School/Private/Mini/KSRTC/City/BRTS Bus, Goods Vehicle, FARM VEHICLE,
evCAR, EMERGENCY VEHICLE, E-RICKSAW, CAR, AUTO RICKSAW, 2 WHEELER). Stock
YOLO (COCO's 80-class set) only recognizes 4 of these, so this pilot counts
exactly those 4 buckets and nothing else, until a fine-tuned model exists
for the rest:

| Bucket (CSV column) | COCO class | Covers requested label(s) |
|---|---|---|
| `BICYCLE` | bicycle | (not originally requested; detected as the non-motorized half of "2 WHEELER") |
| `MOTORCYCLE` | motorcycle | 2 WHEELER |
| `CAR` | car | CAR, TAXI, evCAR |
| `BUS` | bus | School Bus, Private Bus, Mini Bus, KSRTC Bus, City Bus, BRTS Bus |

Not detected by this pilot at all (no COCO class is a reasonable
stand-in): **Slow Vehicle, Goods Vehicle, FARM VEHICLE, EMERGENCY VEHICLE,
E-RICKSAW, AUTO RICKSAW**. Every `run_summary.json` includes a
`class_coverage` block spelling this mapping out per-run, generated from
`scripts/vehicle_counting/vehicle_classes.py` -- the single place this
mapping is defined. Once a fine-tuned model distinguishes the remaining
categories (bus subtypes, taxi vs. car, the six pending classes above),
extend that file rather than hard-coding new class ids elsewhere.

### 20.2 Configuring and running

Config lives in `config/vehicle_config.yaml`, separate from
`config/config.yaml`. Its `roi.polygon` defines the counting line/region (a
vehicle is counted once, the first frame its bounding box's bottom-center
point enters this polygon -- same rule as the pedestrian ROI). The shipped
polygon is a full-frame placeholder; edit it to the target camera's actual
road geometry (e.g. with `scripts/roieditor.py`) before trusting counts.

```bash
cd ~/mvsa
source env/bin/activate
python scripts/vehicle_counting/vehicle_counter.py \
    --config config/vehicle_config.yaml \
    --run-name vehicle_pilot_cam1_20260728_0800 \
    --video videos/actual_test/08.00.00-09.00.00.mp4 \
    --start-datetime "2026-07-28 08:00:00"
```

Outputs go to `results/<run-name>/`: `hourly_vehicle_counts.csv` (one row
per hour, one column per bucket above, plus `total`), `vehicle_tracks.csv`
(per-track detail), `tracked.mp4` (annotated video, for spot-checking
before trusting the numbers), and `run_summary.json`.

### 20.3 Known limitations

- No resume/checkpoint support yet (unlike the pedestrian pipeline) --
  built for pilot-scale runs; add it later via `scripts/checkpoint.py`'s
  pattern if hour-long-plus videos make restart-from-scratch impractical.
- The counting ROI placeholder ships un-tuned; counts from an un-edited
  polygon are not meaningful for a real camera.
- `BUS` is a single undifferentiated total across 6 real-world subtypes --
  do not report it as any one of them.
- `CAR` silently includes taxis and EVs -- there is currently no way to
  tell them apart from a generic car in stock YOLO's output.

## Testing

```bash
cd ~/mvsa
pip install pytest pyyaml   # if not already available
pytest
```

All tests in `tests/` run without a GPU and without YOLO/InsightFace
installed -- they exercise `roi.py`, `models.py` (age buckets),
`aggregation.py`, `config_loader.py`, and the vehicle-counting pilot's
`vehicle_classes.py`/`vehicle_tracker.py` directly with synthetic data.
