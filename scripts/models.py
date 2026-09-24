"""Shared data structures used across the MVSA pipeline.

Keeping these in one module avoids duplicate/near-duplicate dataclasses in
detector.py, tracker.py, demographics.py and aggregation.py, and gives
tests a single, GPU-free place to import from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Gender / age-group label constants
# ---------------------------------------------------------------------------

GENDER_MALE = "male"
GENDER_FEMALE = "female"
GENDER_UNKNOWN = "unknown"

AGE_GROUP_0_18 = "0-18"
AGE_GROUP_18_60 = "18-60"
AGE_GROUP_60_PLUS = "60+"
AGE_GROUP_UNKNOWN = "unknown"

STATUS_VALID = "valid"
STATUS_INSUFFICIENT_QUALITY = "insufficient_face_quality"
STATUS_NO_FACE_DETECTED = "no_face_detected"
STATUS_NOT_SAMPLED = "not_sampled"

# A demographic sample comes from one of two independent estimators. Kept as
# an explicit field (rather than inferred) so a sample's provenance is never
# ambiguous when face and body estimates disagree -- see aggregation.py and
# README Section 16/19.
SOURCE_FACE = "face"
SOURCE_BODY = "body"

# InsightFace's genderage attribute model encodes gender as argmax over two
# classes. Per the model's documented convention (and community-verified
# behaviour of the buffalo_l pack): 0 -> female, 1 -> male. This is NOT
# re-derived from probabilities because the model does not expose a
# calibrated softmax; it is a fixed label mapping.
INSIGHTFACE_GENDER_MAP: Dict[int, str] = {0: GENDER_FEMALE, 1: GENDER_MALE}

# PP-LCNet person-attribute model (PA100K label schema; see
# scripts/body_attributes.py for the verified index layout and provenance).
# Age is one of 3 mutually-exclusive PA100K bucket labels -> mapped directly
# onto MVSA's own bucket names, no numeric age is ever synthesized from this.
PA100K_AGE_GROUP_MAP: Dict[str, str] = {
    "AgeLess18": AGE_GROUP_0_18,
    "Age18-60": AGE_GROUP_18_60,
    "AgeOver60": AGE_GROUP_60_PLUS,
}


def age_to_group(age: Optional[float]) -> str:
    """Bucket a numeric age into MVSA's fixed age groups.

    Exact bucket definitions (inclusive):
        0-17  -> "0-18"
        18-60 -> "18-60"
        61+   -> "60+"
        None  -> "unknown"
    """
    if age is None:
        return AGE_GROUP_UNKNOWN
    age = float(age)
    if age < 0:
        return AGE_GROUP_UNKNOWN
    if age <= 17:
        return AGE_GROUP_0_18
    if age <= 60:
        return AGE_GROUP_18_60
    return AGE_GROUP_60_PLUS


# ---------------------------------------------------------------------------
# Detection / tracking
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Detection:
    """A single YOLO detection in one frame."""

    bbox: BBox
    conf: float
    cls_id: int
    track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------


@dataclass
class FaceSample:
    """One demographic estimation attempt for a track, at a given frame.

    ``usable`` reflects whether face_quality.py accepted the sample. When it
    is False, ``gender``/``age`` are always None -- callers must never use a
    model prediction that failed quality gating.
    """

    frame_idx: int
    face_detected: bool
    usable: bool
    gender: Optional[str]
    age: Optional[float]
    quality_score: float
    det_score: float
    confidence_proxy: Optional[float]
    # Which independent estimator produced this sample. Defaults to "face"
    # so every pre-existing checkpoint/call site (InsightFace path, and all
    # of tests/test_aggregation.py) is unaffected. body_attributes.py sets
    # this to SOURCE_BODY explicitly.
    source: str = SOURCE_FACE
    # Pre-bucketed age group, when the estimator produces one directly
    # (body_attributes.py always does; demographics.py leaves this None and
    # aggregation.py falls back to models.age_to_group(age) for numeric-age
    # samples). Kept separate from ``age`` because the body estimator has no
    # numeric age at all -- only a bucket -- and fabricating a fake numeric
    # age from a bucket midpoint would be dishonest precision.
    age_group: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "face_detected": self.face_detected,
            "usable": self.usable,
            "gender": self.gender,
            "age": self.age,
            "quality_score": self.quality_score,
            "det_score": self.det_score,
            "confidence_proxy": self.confidence_proxy,
            "source": self.source,
            "age_group": self.age_group,
        }

    @staticmethod
    def from_dict(d: dict) -> "FaceSample":
        return FaceSample(
            frame_idx=int(d["frame_idx"]),
            face_detected=bool(d["face_detected"]),
            usable=bool(d["usable"]),
            gender=d.get("gender"),
            age=d.get("age"),
            quality_score=float(d["quality_score"]),
            det_score=float(d["det_score"]),
            confidence_proxy=d.get("confidence_proxy"),
            source=d.get("source", SOURCE_FACE),
            age_group=d.get("age_group"),
        )


@dataclass
class TrackDemographicRecord:
    """One row of track_demographics.csv -- the final, aggregated result."""

    track_id: int
    first_seen: str
    last_seen: str
    gender: str
    gender_confidence: Optional[float]
    age_estimate: Optional[float]
    age_group: str
    age_confidence: Optional[float]
    valid_face_samples: int
    total_face_attempts: int
    average_face_quality: float
    status: str

    def to_row(self) -> List:
        return [
            self.track_id,
            self.first_seen,
            self.last_seen,
            self.gender,
            "" if self.gender_confidence is None else round(self.gender_confidence, 4),
            "" if self.age_estimate is None else round(self.age_estimate, 1),
            self.age_group,
            "" if self.age_confidence is None else round(self.age_confidence, 4),
            self.valid_face_samples,
            self.total_face_attempts,
            round(self.average_face_quality, 4),
            self.status,
        ]


TRACK_DEMOGRAPHICS_CSV_HEADER = [
    "track_id",
    "first_seen",
    "last_seen",
    "gender",
    "gender_confidence",
    "age_estimate",
    "age_group",
    "age_confidence",
    "valid_face_samples",
    "total_face_attempts",
    "average_face_quality",
    "status",
]
