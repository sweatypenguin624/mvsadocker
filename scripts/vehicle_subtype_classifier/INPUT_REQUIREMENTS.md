# Step 1 (subtype classification) — input requirements

This is the first step of a new pipeline at `scripts/vehicle_subtype_classifier/`.
Its job: given frames that some earlier step has already extracted and
identified as containing one vehicle each, classify each frame into one of
UVH-26's 14 native subtypes (Hatchback, Sedan, SUV, MUV, Bus, Truck,
Three-wheeler, Two-wheeler, LCV, Mini-bus, tempo-traveller, bicycle, Van,
Others — see `models/UVH-26/uvh_classes.txt` and
`config/vehicle_subtype_config.yaml`).

**The frame-extraction step itself is not built yet.** This document is the
contract Step 1 needs that step (or any other frame source) to satisfy.

## What "a frame" must be

- **One vehicle per image, already identified.** Step 1 does not do
  multi-object detection or tracking — it runs UVH-26 in detect mode and
  keeps only the single highest-confidence box per image. If a frame
  contains more than one vehicle, all but the most confident one are
  silently discarded from that frame's label (logged at debug level,
  visible via `num_detections` in the output). Upstream extraction must
  already have cropped or otherwise isolated the vehicle of interest —
  e.g. per-track crops the way `scripts/traffic_layer1/` already saves
  bus crops, or an equivalent per-vehicle crop from a full detector run.
- **Reasonably tight, upright crops.** UVH-26 was trained on vehicle-scale
  boxes; a full uncropped traffic-camera frame will work (it's still a
  valid detection input) but wastes most of Step 1's accuracy budget on
  background and is far more likely to pick up the wrong vehicle if more
  than one is in frame. Prefer crops close to the vehicle's bounding box,
  not the full source frame.
- **No orientation/lighting assumptions beyond what UVH-26 was trained on.**
  No pipeline-side normalization is done (no rotation, exposure
  correction, deblurring). Very low light, extreme motion blur, or heavy
  occlusion will most likely fall through to `Unclassified` (zero
  detections above the confidence threshold) rather than a wrong label,
  since confidence naturally drops — but this isn't guaranteed for
  partial occlusion.

## Directory layout

```
<frames-dir>/
  <anything>/.../frame001.jpg
  <anything>/.../frame002.jpg
  ...
```

- Passed via `--frames-dir` to `scripts/vehicle_subtype_classifier/run.py`.
- Searched **recursively**; any nesting under `--frames-dir` is fine (e.g.
  one subfolder per source track/video, matching how
  `scripts/traffic_layer1/` already lays out `bus_tracks/track_*/`).
- The path of each frame relative to `--frames-dir` is preserved verbatim
  in the output (`subtype_results.jsonl`'s `"frame"` field), so encode
  anything Step 1 needs to know later (source video, track id, timestamp)
  into that relative path/filename now — Step 1 does not parse filenames
  itself, it treats them as opaque identifiers.

## File format

- Extensions read: `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive).
- Must be decodable by OpenCV (`cv2.imread`) — standard BGR raster images.
  Anything `cv2.imread` returns `None` for is skipped and listed under
  `unreadable_frames` in the summary, not silently dropped.
- No minimum/maximum resolution is enforced; images are resized internally
  to `imgsz` (640 by default, `config/vehicle_subtype_config.yaml`) the
  same way Ultralytics resizes any inference input.

## What Step 1 produces (for whatever consumes it next)

Per run, under `--output-dir`:

- `subtype_results.jsonl` — one JSON object per frame:
  `{"frame": "<relative path>", "class": "<one of the 14 classes or Unclassified>",
  "confidence": <float>, "bbox": [x1,y1,x2,y2] | null, "num_detections": <int>}`
- `subtype_summary.json` — run-level counts per class, unreadable frame
  list, elapsed time, model path used.

## Open questions for whoever builds the extraction step

- **Source of frames**: raw video frame sampling, or crops from an
  existing tracker (Layer 1's `bus_tracks/` or a new detector pass)?
  Per-track crops are strongly preferred over raw frame sampling, since
  raw frames risk multiple/no vehicles per image (see above).
  Left this open per current project decision to focus on Step 1 alone
  right now.
- **One classification per frame vs. per track**: Step 1 classifies each
  image independently and does not aggregate across a track the way
  Layer 1's confidence-weighted class voting does
  (`scripts/traffic_layer1/classifier.py::vote_class`). If the eventual
  input is many frames of the same physical vehicle, a later step should
  decide how to reduce those per-frame labels to one track-level label —
  not assumed here.
