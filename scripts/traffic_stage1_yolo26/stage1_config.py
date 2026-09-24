"""Stage 1 configuration.

Stage 1 is deliberately class-agnostic: it answers only "how many distinct
physical vehicles crossed the counting line, and when". Every knob that
would bias that toward a particular vehicle type lives in Stage 2 instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DetectorConfig:
    # YOLO26 experiment: COCO-pretrained weights (scripts/yolo26x.pt),
    # unlike UVH-26 this is a general-purpose detector, so `include_classes`
    # below restricts it to vehicle categories instead of relying on
    # `exclude_classes` alone.
    weights: str = "scripts/yolo26x.pt"
    device: str = "0"
    imgsz: int = 1280
    # Deliberately low: Stage 1 optimises recall ("no vehicle is missed").
    # Weak detections are still useful to ByteTrack's second association
    # pass, and a track is only *counted* once it clears the far stricter
    # identity gates below, so a low floor here does not create phantom
    # vehicles the way it would in a single-stage counter.
    conf: float = 0.15
    # Class-agnostic NMS: the model's own class labels are discarded, so
    # two boxes on one vehicle under different labels must still compete.
    iou: float = 0.55
    agnostic_nms: bool = True
    # Native class names to drop entirely before collapsing to "vehicle"
    # (e.g. person). Matched case-insensitively against model.names.
    exclude_classes: List[str] = field(default_factory=lambda: ["person", "pedestrian"])
    # COCO class names to keep; empty means "keep everything not excluded"
    # (UVH-26 behaviour). Set for YOLO26's COCO nano weights so traffic
    # lights, stop signs, etc. never collapse into "vehicle".
    include_classes: List[str] = field(
        default_factory=lambda: ["bicycle", "car", "motorcycle", "bus", "truck"]
    )
    half: bool = True
    batch_size: int = 8
    # Sanity bounds on box size, as a fraction of frame area. A single
    # vehicle never fills most of the frame at these camera geometries, so
    # a box above the max is a detector hallucination spanning the scene
    # (observed on real footage: one box covering a whole intersection,
    # which the tracker then happily counted as a vehicle). The min drops
    # specks too small for Stage 2 to classify anyway. Both are recall-safe
    # at these defaults -- widen max_box_area_frac for very close-up
    # cameras where a bus legitimately fills the frame.
    max_box_area_frac: float = 0.35
    min_box_area_frac: float = 0.00015


@dataclass
class FusionConfig:
    """Detection-level box fusion, run before tracking.

    Fixes the failure mode IoU-NMS structurally cannot: several boxes
    tiled front-to-back along one very large/close vehicle each have LOW
    mutual IoU (they barely overlap) yet are all the same object. Overlap
    is therefore measured as intersection-over-minimum-area (IoM), which
    is ~1.0 whenever one box is largely swallowed by another regardless of
    their size difference.
    """

    enabled: bool = True
    iom_thresh: float = 0.65
    # Only fuse when the boxes are plausibly one object: guards against
    # merging a genuinely small vehicle that happens to sit in front of a
    # bus. A motorcycle in front of a bus has a huge area ratio and a
    # much lower containment than a mis-split bus panel.
    max_area_ratio: float = 8.0
    # Require the smaller box to be near-contained, not merely overlapping.
    min_containment: float = 0.80


@dataclass
class TrackerConfig:
    # "botsort" adds camera-motion compensation and optional ReID on top
    # of ByteTrack's two-stage association; "bytetrack" is the plain form.
    tracker_type: str = "botsort"
    # sparseOptFlow GMC estimates and compensates for CAMERA motion between
    # frames -- pure overhead on a fixed CCTV camera (measured: ~29x the
    # cost of the rest of tracking combined, for zero benefit, since there
    # is no camera motion to compensate for). Set to "sparseOptFlow" only
    # if a camera genuinely pans/shakes.
    track_high_thresh: float = 0.45
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.50
    # Generous: a vehicle occluded by a bus for ~2s at 25fps must be
    # re-found as the SAME track rather than counted twice.
    track_buffer: int = 90
    match_thresh: float = 0.85
    fuse_score: bool = True
    gmc_method: str = "none"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.75
    with_reid: bool = False
    reid_model: str = "auto"


@dataclass
class IdentityConfig:
    """Raw tracker IDs -> physical vehicle identities.

    Two independent over-count sources are collapsed here, as a post-pass
    over the whole video (not online), because both need evidence from
    frames later than the one where the mistake is made.
    """

    enabled: bool = True

    # (a) Spatial dedup: two tracks alive at the same time on the same
    # object. Requires sustained co-occurrence, not one unlucky frame.
    spatial_enabled: bool = True
    spatial_iom_thresh: float = 0.70
    spatial_min_overlap_frames: int = 5
    spatial_min_overlap_ratio: float = 0.50

    # (a2) Lockstep dedup: the "one bus split into front/door/rear panels"
    # case. Those panels barely overlap each other, so no per-frame overlap
    # threshold can catch them -- but they are rigidly attached to one
    # body, so their relative offset stays nearly constant for the whole
    # track. Two vehicles in adjacent lanes drift apart instead.
    lockstep_enabled: bool = True
    lockstep_min_frames: int = 12
    # Max total DRIFT of the inter-box centre offset across the shared
    # frames (peak-to-peak range, not std-dev), as a fraction of the body
    # diagonal. Range is the sharper discriminator: two vehicles pulling
    # apart at a steady few px/frame have a small std but a large range,
    # whereas boxes bolted to one body have both near zero.
    lockstep_max_offset_drift_frac: float = 0.15
    # Boxes must also be touching/overlapping: the gap between them, as a
    # fraction of the smaller box's size, stays under this.
    lockstep_max_gap_frac: float = 0.15

    # (b) Temporal stitching: one vehicle split into track A then track B
    # by an ID switch. B must be born close in time and space to where A
    # died, moving the same way.
    temporal_enabled: bool = True
    temporal_max_gap_frames: int = 45
    # Position tolerance is expressed relative to the VEHICLE'S OWN SIZE,
    # not in absolute pixels: the old flat "45px per frame of gap" allowed
    # ~1750px of slack across a 39-frame gap, i.e. most of the frame.
    temporal_tolerance_diag_frac: float = 0.6
    temporal_tolerance_growth_per_frame: float = 0.05
    temporal_tolerance_max_diag_frac: float = 2.0
    # Appearance veto. Two vehicles following the same lane at the same
    # speed are geometrically indistinguishable -- colour is the only
    # cheap signal that separates them. A stitch is rejected when both
    # tracks have a fingerprint and they disagree by more than this.
    # Tracks with no fingerprint (too small to sample) are NOT vetoed.
    #
    # 0.80 is measured, not guessed: on real footage from this camera,
    # crops of the SAME vehicle scored >=0.94 correlation (mean 0.98)
    # while two different scooters in the same lane scored <=0.65
    # (mean 0.61). 0.80 sits in the middle of that gap, so it vetoes the
    # bad stitch without endangering genuine occlusion recovery.
    temporal_min_appearance: float = 0.80
    temporal_min_iou_at_gap: float = 0.15
    temporal_max_direction_deg: float = 45.0
    temporal_max_size_ratio: float = 2.0


@dataclass
class CountingConfig:
    """A vehicle is counted once, at the frame its ground-contact point
    crosses the counting line."""

    # [[x1, y1], [x2, y2]] in reference_width/height pixel space.
    line: Optional[List[List[float]]] = None
    reference_width: int = 1920
    reference_height: int = 1080
    # Fallback when no line is calibrated for this camera: a horizontal
    # line at this fraction of frame height. Counts are usable but should
    # be treated as uncalibrated -- flagged in the summary.
    fallback_line_y_frac: float = 0.60
    # A track must be observed this many frames total to count at all --
    # kills single-frame detector flickers.
    min_track_frames: int = 6
    # ...and must be seen this many frames on EACH side of the line, so a
    # jittering box sitting on the line cannot register a crossing.
    min_frames_each_side: int = 2
    interval_minutes: int = 15


@dataclass
class CropsConfig:
    enabled: bool = True
    max_per_vehicle: int = 5
    jpeg_quality: int = 92
    # Pad the box before cropping so Stage 2 sees wheels/axles and a
    # little context rather than a box shaved tight to the body.
    pad_frac: float = 0.08
    min_box_px: int = 24


@dataclass
class Stage1Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    crops: CropsConfig = field(default_factory=CropsConfig)

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p


def _build(cls, data: Optional[Dict[str, Any]]):
    if not data:
        return cls()
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: Optional[Path] = None, camera_key: Optional[str] = None) -> Stage1Config:
    raw: Dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    cameras = raw.pop("cameras", {}) or {}
    if camera_key:
        if camera_key not in cameras:
            raise KeyError(
                f"camera_key '{camera_key}' not found in {path}. Known: {sorted(cameras)}"
            )
        # Per-camera block overrides the global one, section by section.
        for section, overrides in (cameras[camera_key] or {}).items():
            merged = dict(raw.get(section) or {})
            merged.update(overrides or {})
            raw[section] = merged

    return Stage1Config(
        detector=_build(DetectorConfig, raw.get("detector")),
        fusion=_build(FusionConfig, raw.get("fusion")),
        tracker=_build(TrackerConfig, raw.get("tracker")),
        identity=_build(IdentityConfig, raw.get("identity")),
        counting=_build(CountingConfig, raw.get("counting")),
        crops=_build(CropsConfig, raw.get("crops")),
    )
