"""Layer 1 broad taxonomy + the native-class-name -> bucket mapping loaded
from config/layer1_class_mapping.yaml (see that file for the actual
mapping data -- this module only loads/looks it up, per the spec's
requirement that the mapping be configurable rather than hard-coded).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

if TYPE_CHECKING:
    from config import ConfidenceConfig
    from layer1_models import SourcedDetection

logger = logging.getLogger("mvsa.traffic_layer1")

# The fixed Layer 1 broad-class list (spec section 1). Every counted track's
# final_class is always one of these -- classifier.py never invents a label
# outside this set, and taxonomy.py::map_class falls back to "Others" for
# any native class not explicitly mapped.
BROAD_CLASSES: List[str] = [
    "Car",
    "2-Wheeler",
    "Auto",
    "Bus",
    "Pedestrian",
    "Emergency Vehicle",
    "Cycle",
    "Cycle Rickshaw",
    "Animal/Hand-drawn Cart",
    "Goods Vehicle",
    "Tractor",
    "Others",
]

OTHERS = "Others"


@dataclass
class ClassMapping:
    # detector source ("uvh26" / "coco") -> {native_class_name: bucket}
    by_source: Dict[str, Dict[str, str]]
    pending_model_support: List[str] = field(default_factory=list)
    _warned_unmapped: Set[str] = field(default_factory=set)

    def map_class(self, source: str, native_class_name: str) -> str:
        """Map one detector's native class name to a Layer 1 broad bucket.
        Unrecognized (source, native_class_name) pairs fall back to
        "Others" -- logged once per (source, class) pair, not once per
        detection, so a genuinely unmapped class doesn't flood the log on
        a long video.
        """
        bucket = self.by_source.get(source, {}).get(native_class_name)
        if bucket is not None:
            return bucket
        key = f"{source}:{native_class_name}"
        if key not in self._warned_unmapped:
            logger.warning(
                "No Layer 1 mapping for source=%s native_class=%r -- routing to %r",
                source,
                native_class_name,
                OTHERS,
            )
            self._warned_unmapped.add(key)
        return OTHERS

    def coverage_report(self) -> dict:
        """requested-bucket -> status, written into every run's summary.json
        so it's obvious from the output alone which of the 12 Layer 1
        buckets have real detector support today vs. are still pending a
        model that can see them -- same convention as
        scripts/vehicle_counting/vehicle_classes.py::coverage_report().
        """
        covered_buckets: Set[str] = set()
        for mapping in self.by_source.values():
            covered_buckets.update(mapping.values())
        return {
            "detected_buckets": sorted(covered_buckets),
            "pending_model_support": list(self.pending_model_support),
        }


def tag_and_filter(
    detections: "List[SourcedDetection]",
    class_mapping: ClassMapping,
    confidence: "ConfidenceConfig",
) -> "List[SourcedDetection]":
    """Map every detection's native class to its broad bucket (setting
    ``.bucket`` in place) and drop any detection whose confidence is below
    that bucket's configured threshold (spec section 5: "the detector
    should return only detections that pass the configured threshold",
    applied per-bucket since Ultralytics only supports one global conf per
    model). Original confidence scores are never modified -- only used to
    decide inclusion.

    Centralized here (rather than duplicated in tracker.py and crops.py,
    which both need a detection's bucket) so the mapping+threshold policy
    is applied exactly once per detection.
    """
    kept = []
    for det in detections:
        bucket = class_mapping.map_class(det.source, det.native_class)
        if det.conf < confidence.threshold_for(bucket):
            continue
        det.bucket = bucket
        kept.append(det)
    return kept


def load_class_mapping(mapping_path: Path) -> ClassMapping:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"class mapping file not found: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    by_source: Dict[str, Dict[str, str]] = {}
    for source, entries in raw.items():
        if source == "pending_model_support":
            continue
        if not isinstance(entries, dict):
            continue
        by_source[source] = {str(k): str(v) for k, v in entries.items()}

    pending = list(raw.get("pending_model_support", []) or [])

    unknown_buckets = {
        bucket
        for mapping in by_source.values()
        for bucket in mapping.values()
        if bucket not in BROAD_CLASSES
    }
    if unknown_buckets:
        raise ValueError(
            f"layer1_class_mapping.yaml maps to bucket(s) outside the Layer 1 taxonomy: "
            f"{sorted(unknown_buckets)}"
        )

    return ClassMapping(by_source=by_source, pending_model_support=pending)
