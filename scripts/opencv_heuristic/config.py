"""Configuration for the bus color classification pipeline.

Local re-implementation of a 3-cell Colab notebook (CPM Hubli-Dharwad bus
color classification). Colab/Drive-specific plumbing (drive.mount, the fixed
DRIVE_DAV_SOURCE_PATH, /content paths) is replaced with plain CLI args -- see
run.py -- everything else (HSV heuristic, tracker settings, mode logic) is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # Color classes ------------------------------------------------------
    COLOR_CLASSES: list = field(default_factory=lambda: ["red", "blue", "yellow", "rest"])

    # HSV heuristic color classifier (OpenCV hue range is 0-179). Each entry
    # is a list of (hue_low, hue_high) ranges that count as this color. Red
    # wraps around 0/179, so it needs two ranges.
    HSV_COLOR_RANGES: dict = field(default_factory=lambda: {
        "red": [(0, 10), (160, 179)],
        "yellow": [(18, 35)],
        "blue": [(95, 130)],
    })
    HSV_MIN_SATURATION: int = 60   # ignore washed-out/gray/white pixels below this
    HSV_MIN_VALUE: int = 40        # ignore near-black/shadow pixels below this
    # Per-color override of HSV_MIN_SATURATION. Red's hue range borders brown/rust
    # and dull purple/maroon liveries, which sit at lower saturation than genuine
    # red paint -- raising red's floor cuts most of that bleed (see README).
    HSV_COLOR_MIN_SATURATION: dict = field(default_factory=lambda: {"red": 110})
    # If matched pixels are too small a share of all valid pixels, or the winning
    # color doesn't clearly dominate the matched pixels, the crop is too ambiguous
    # to call -- fall back instead of picking whichever color happened to edge out.
    MIN_COLOR_MATCH_COVERAGE: float = 0.30
    MIN_COLOR_WIN_SHARE: float = 0.55
    FALLBACK_COLOR: str = "rest"

    # Detection / tracking -------------------------------------------------
    DETECTOR_MODEL: str = "yolo11n.pt"  # pretrained COCO YOLOv11n; auto-downloads, includes 'bus' class
    BUS_CLASS_ID: int | None = None      # if None, auto-detect the 'bus' class from detector.names
    DETECTION_CONFIDENCE: float = 0.25
    IMGSZ: int | None = None              # None => ultralytics default (640); set to the detector's training size
    TRACKER_CONFIG: str = "botsort.yaml"  # BoT-SORT only; do not use bytetrack/sort/ocsort/deepsort

    # Cropping --------------------------------------------------------------
    MIN_CROP_WIDTH: int = 20
    MIN_CROP_HEIGHT: int = 20
    CROP_PADDING: float = 0.05

    # full_pipeline mode ------------------------------------------------
    CLASSIFY_INTERVAL_SECONDS: float = 0.5
    INTERVAL_MINUTES: int = 15
    SAVE_ANNOTATED_VIDEO: bool = False
    SAVE_DEBUG_CROPS: bool = False
    MAX_DEBUG_CROPS_PER_CLASS: int = 50
    CHECKPOINT_EVERY_N_FRAMES: int = 5000

    # collect_crops mode --------------------------------------------------
    CROP_EVERY_SECONDS: float = 2.0    # save at most one crop per track every N seconds
    MAX_CROPS_PER_TRACK: int = 15      # cap crops collected per track_id
    MAX_TOTAL_CROPS: int = 1500        # hard cap on total crops saved this run

    # train_classifier mode ------------------------------------------------
    CLASSIFIER_BASE_MODEL: str = "yolo11n-cls.pt"
    CLASSIFIER_TRAIN_EPOCHS: int = 30
    CLASSIFIER_IMG_SIZE: int = 128

    # production mode (mode_production.py) ---------------------------------
    # Multi-class vehicle counting groups: UVH-26 class name -> group. A UVH-26
    # fine-tune (see uvh26_train.py) natively distinguishes all of these, so
    # production mode counts every group in one pass, not just buses. Classes
    # not listed here (Bicycle, Others) are tracked but excluded from every
    # group's count -- reported neither as a false inflation of a real
    # category nor silently dropped from track_details.csv, just not one of
    # the requested headline counts. Adjust freely; this is the mapping, not
    # a hardcoded assumption baked into the pipeline.
    VEHICLE_GROUPS: dict = field(default_factory=lambda: {
        "two_wheeler": ["Two-wheeler"],
        "auto_rickshaw": ["Three-wheeler"],
        "four_wheeler": ["Hatchback", "Sedan", "SUV", "MUV", "Van"],
        "bus": ["Bus", "Mini-bus", "Tempo-traveller"],
        "truck": ["Truck", "LCV"],
    })
    # Only this group gets the HSV color-classification second layer (classify.py)
    # -- validated on real bus liveries; not claimed to generalize to the other groups.
    COLOR_CLASSIFIED_GROUP: str = "bus"

    # Detections smaller than this fraction of the frame area are dropped
    # before tracking even sees them, per group. 'bus': on real Gokul Road
    # footage (1920x1080) with the UVH-26 fine-tune, a specific small-van
    # silhouette (~70-190px wide) got misclassified as 'Bus' at confidence up
    # to 0.85 -- overlapping genuine bus confidences, so no confidence
    # threshold could separate them. But absolute box area cleanly does:
    # every false positive found measured under 45,000px^2 while every
    # genuine bus measured over 260,000px^2 (a 5.8x gap, verified across 49
    # tracks). 0.025 (~52,000px^2 at 1920x1080) sits in that gap.
    # 'default' (every other group) is only a generic noise floor, NOT
    # validated with the same rigor as 'bus' -- spot-check other groups via
    # production_review.py before trusting their counts at face value.
    # Re-derive both if run on a camera with a meaningfully different
    # resolution or mounting distance.
    GROUP_MIN_AREA_FRAC: dict = field(default_factory=lambda: {
        "bus": 0.025,
        "default": 0.0008,
    })
    # Cross-class suppression: cargo trucks are the detector's main
    # remaining confusion with 'bus' (empirically: even when the detector's
    # top call for a box is 'bus', it frequently also assigns 'truck' a
    # real, non-negligible score for the exact same box -- just below the
    # normal detection threshold; genuine buses essentially never show this
    # competing signal). Requesting these classes alongside 'bus' at a low
    # threshold surfaces that shadow score so an ambiguous box can be
    # rejected per-frame instead of accepted as a confident 'bus'.
    SUPPRESS_AMBIGUOUS_CLASSES: bool = True
    # "motorcycle" covers auto-rickshaws too -- COCO has no dedicated rickshaw
    # class, and a three-wheeler is visually closer to the detector's notion
    # of 'motorcycle' than 'car' or 'truck'.
    AMBIGUOUS_CLASS_NAMES: list = field(default_factory=lambda: ["truck", "car", "motorcycle"])
    SUPPRESSION_DETECTION_CONF: float = 0.05   # low conf floor used only to surface competing-class shadow scores
    BUS_SUPPRESSION_IOU: float = 0.5           # competing-class box overlap (IoU) with the bus box to trigger rejection
    # Per-track best-frame scoring: score = confidence * area_norm * sharp_norm
    # * edge_penalty. Bigger/sharper/more-confident/less-edge-clipped wins.
    BEST_FRAME_TARGET_AREA_FRAC: float = 0.15   # box area / frame area considered "large enough"; no bonus past this
    BEST_FRAME_SHARPNESS_REF: float = 150.0     # Laplacian-variance value considered "sharp enough"
    BEST_FRAME_EDGE_MARGIN_PX: int = 4          # box within this many px of a frame edge counts as clipped
    BEST_FRAME_EDGE_PENALTY: float = 0.5        # score multiplier applied to edge-clipped detections
    SAVE_BEST_FRAME_CROPS: bool = True          # write each track's winning crop to <output>/best_frames/<color>/

    # Misc ------------------------------------------------------------------
    DEBUG: bool = True
    PROGRESS_EVERY_N_FRAMES: int = 500
