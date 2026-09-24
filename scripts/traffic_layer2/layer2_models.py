"""Layer 2 data structures: what one Goods Vehicle track looks like coming
in from Layer 1, and what a finished Layer 2 classification looks like
going out (spec sections 3, 21).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RepresentativeFrame:
    """One saved crop from Layer 1's crops.py, plus the geometry Layer 2
    needs (bbox in source-frame pixel space, detector confidence) that a
    bare JPEG can't carry.
    """

    file: str
    path: Path
    frame_idx: int
    crop_quality_score: float
    bbox: List[float]  # [x1, y1, x2, y2], source-frame pixel space
    detector_conf: float

    @property
    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        h = max(1.0, y2 - y1)
        return (x2 - x1) / h


@dataclass
class GoodsTrack:
    """One Layer 1 track whose final broad class was "Goods Vehicle" --
    Layer 2's unit of work (spec section 4: operate on a track, not
    independent detections).
    """

    track_id: int
    layer1_class: str  # always "Goods Vehicle" for what Layer 2 consumes
    layer1_confidence: float
    direction: str
    first_seen: str
    last_seen: str
    frames_seen: int
    best_frame: Optional[RepresentativeFrame]
    frames: List[RepresentativeFrame] = field(default_factory=list)


@dataclass
class StagePrediction:
    """One classifier stage's output for one representative frame."""

    label: str
    probs: Dict[str, float]
    source: str  # "heuristic" or "torch:<weights_path>"

    @property
    def confidence(self) -> float:
        return self.probs.get(self.label, 0.0)

    @property
    def top2_margin(self) -> float:
        ranked = sorted(self.probs.values(), reverse=True)
        if len(ranked) < 2:
            return 1.0
        return ranked[0] - ranked[1]


@dataclass
class Layer2Result:
    """Final, track-level Layer 2 classification (spec section 21)."""

    track_id: int
    layer1_class: str
    layer2_class: str
    confidence: float
    direction: str
    first_seen: str
    last_seen: str
    best_crop: Optional[str]
    branch_path: List[str] = field(default_factory=list)  # e.g. ["Heavy Truck", "3 Axle Truck"]
    is_uncertain: bool = False
    trolley_track_id: Optional[int] = None
    per_stage: Dict[str, dict] = field(default_factory=dict)  # diagnostics, not spec-required

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "layer_1_class": self.layer1_class,
            "layer_2_class": self.layer2_class,
            "confidence": round(self.confidence, 4),
            "direction": self.direction,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "best_crop": self.best_crop,
            "branch_path": self.branch_path,
            "is_uncertain": self.is_uncertain,
            "trolley_track_id": self.trolley_track_id,
            "per_stage": self.per_stage,
        }
