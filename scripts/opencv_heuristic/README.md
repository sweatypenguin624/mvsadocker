# Bus color classification (OpenCV HSV heuristic)

Local re-implementation of the CPM Hubli-Dharwad Colab notebook: detect and
track buses (YOLO + BoT-SORT), classify each bus's livery color, and
aggregate counts into 15-minute intervals. No Drive/Colab dependency --
takes a local video path directly (a `.dav` input is auto-converted to
`.mp4` via ffmpeg).

## Production pipeline (`run_production.py`)

The config-driven, multi-video entrypoint -- use this over `full_pipeline`
mode below for real runs. Config-driven means an editable YAML file, not
CLI flags: point it at a folder (one camera's worth of footage, processed
video by video) or a single video (for a quick test on e.g. one 15-minute
clip before pointing at the whole folder).

```
env/bin/python scripts/opencv_heuristic/run_production.py \
    --config scripts/opencv_heuristic/production_config.example.yaml
```

Copy `production_config.example.yaml`, edit `mode`/`input_folder` (or
`input_video`)/`output_dir`, and go -- every option is commented in that
file (detector model, camera name, per-file start-time overrides, etc).

What it does differently from `full_pipeline` mode:
- **Detector: YOLO26** (`yolo26s.pt` by default, COCO-pretrained, includes
  'bus'), not YOLO11n. A bigger/newer detector materially cuts down on
  class-confusion false positives -- autos, trucks, and cars the old
  `yolo11n.pt` sometimes called 'bus'. Tiny detections (`MIN_BOX_AREA_FRAC`
  in `config.py`) are dropped before they even get a track ID, filtering
  out another chunk of distant/spurious "bus" detections.
- **One best frame per track, not a running average.** `best_frame.py`
  scores every detection of a track by confidence x box-size x sharpness
  x not-edge-clipped, keeps only the single best crop per track (no frame
  history retained), and classifies that once -- instead of
  `full_pipeline`'s `TrackColorAccumulator`, which reclassifies repeatedly
  across a track's lifetime and fuses the results.
- **Real wall-clock 15-minute windows** ("09:00-09:15"), not
  seconds-since-video-start. `video_discovery.py` infers each video's true
  start time from its filename (`08.00.00-09.00.00.mp4` or
  `20260720_090553_tp00075.mp4` patterns; falls back to file mtime with a
  loud warning -- override per-file via `video_start_times` in the config
  if that's wrong).
- **Per-camera output.** Camera name comes from the config or, by default,
  from each video's containing folder (one folder per camera is the
  expected `mode: folder` layout).

Outputs land in `output_dir`:
- `track_details.csv` -- one row per track: camera, video, first/last
  seen, best-frame timestamp + detection confidence, color + color
  confidence.
- `window_counts.csv` -- unique red/blue/yellow/rest counts per camera per
  15-minute window, summed across every video in the run.
- `window_counts_by_video.csv` -- same, broken out per source video
  (uniqueness only holds within one video -- BoT-SORT state doesn't
  persist across separate files, so a bus present at a file boundary can
  be counted once in each adjacent file).
- `best_frames/<color>/` -- every track's winning crop, for visual audit
  (mirrors `review_grid.py`'s categories).
- `run_summary.txt` -- run-level totals and a per-video breakdown.

Three modes, mirroring the notebook's `RUN_MODE`:

- **collect_crops** -- sample bus crops per tracked ID so they can be sorted
  by hand into `train/red`, `train/blue`, `train/yellow`, `train/rest` for a
  future trained classifier.
- **train_classifier** -- fine-tune a `yolo11n-cls.pt` head on that sorted
  dataset. Not used by `full_pipeline` yet (see below).
- **full_pipeline** -- the actual run: tracks buses, classifies color with
  the OpenCV HSV heuristic (`classify.py`, no trained model needed), and
  writes `track_results.csv`, `15min_counts.csv`, `run_summary.txt`, and two
  PNG charts.

## Usage

```
env/bin/python scripts/opencv_heuristic/run.py full_pipeline \
    --video videos/actual_test/08.00.00-09.00.00.mp4 \
    --output results/bus_color/run_2026_09_02

env/bin/python scripts/opencv_heuristic/run.py collect_crops \
    --video videos/actual_test/08.00.00-09.00.00.mp4 \
    --output results/bus_color/unlabeled_crops

env/bin/python scripts/opencv_heuristic/run.py train_classifier \
    --color-dataset-dir results/bus_color/bus_color_dataset
```

`--device` defaults to `cuda:0` when available, else `cpu`. `--bus-class-id`
overrides auto-detection of the `bus` class in the detector's COCO names.

## Files

- `config.py` -- `PipelineConfig`: HSV color ranges, tracker/detector
  settings, per-mode knobs (including the production-mode best-frame and
  tiny-box-rejection knobs).
- `video_io.py` -- optional `.dav` -> `.mp4` conversion, video metadata.
- `detector.py` -- YOLO detector load + bus-class-id resolution.
- `crops.py` -- `crop_bus`: pad/clip/validate a bounding box crop.
- `classify.py` -- `classify_bus_crop` (per-crop HSV heuristic) and
  `TrackColorAccumulator` (per-track running evidence across many crops,
  used by `full_pipeline` only -- production mode uses `best_frame.py`
  instead).
- `mode_collect_crops.py`, `mode_train_classifier.py`,
  `mode_full_pipeline.py` -- one module per `RUN_MODE`.
- `run.py` -- CLI entrypoint wiring the above together (the three
  `RUN_MODE` modes only -- not production mode, see below).
- `run_config.py` -- `RunConfig`: YAML loader for `run_production.py`.
- `video_discovery.py` -- video discovery, camera-name derivation, and
  filename-based wall-clock start-time inference for production mode.
- `best_frame.py` -- per-track best-frame scoring/selection for production
  mode.
- `mode_production.py` -- single-video production processing (called by
  `run_production.py` once per discovered video).
- `run_production.py` -- production CLI entrypoint (`--config`).
- `production_config.example.yaml` -- documented example config.
- `review_grid.py` -- read-only browser grid for eyeballing heuristic
  predictions by class (red/blue/yellow/rest), with per-image confidence.
- `sort_crops.py` -- browser-based manual labeling UI for building the
  `train_classifier` dataset.
- `auto_classify_crops.py` -- bulk-sorts a crops folder into
  `train/<color>/` using the HSV heuristic itself (fast dataset
  bootstrapping; expect to hand-clean it afterward, it's not ground truth).

## Known limitations (carried over from the notebook)

- Color classification is a pure HSV heuristic; no trained model is wired
  into `full_pipeline`/production mode yet even after running
  `train_classifier` -- `COLOR_CLASSIFIER_MODEL` from the notebook was
  never actually consumed by Cell 3's `full_pipeline` mode, so this local
  version doesn't either. Adding that requires deciding how the trained
  classifier's predictions should combine with (or replace) the HSV
  heuristic.
- The HSV heuristic still occasionally confuses brown/rust delivery trucks
  with red buses (brown and red mainly differ by saturation/value, not
  hue) -- production mode's tighter detector should mean fewer trucks reach
  the classifier at all, but it isn't a complete fix.
- BoT-SORT can bridge short occlusions but a long disappearance produces a
  new track ID on reappearance -- no custom identity-merging is applied,
  in either `full_pipeline` or production mode.
- `save_checkpoint()` in `mode_full_pipeline.py` is an intermediate-results
  backup only, not a true tracker resume: restarting from it re-initializes
  BoT-SORT and will likely assign new IDs to buses already in view.
  Production mode has no checkpointing at all yet -- a crash partway
  through a folder loses progress on the video in flight (already-written
  per-video results for earlier videos in the folder are safe, since
  `track_details.csv`/`window_counts.csv` are only written once at the
  very end of the whole run). Worth adding if folder runs get long enough
  to matter.
- Production mode's per-video track uniqueness doesn't carry across file
  boundaries (see `window_counts_by_video.csv` note above) -- a bus present
  right at the boundary between two consecutive files can be counted once
  in each.
