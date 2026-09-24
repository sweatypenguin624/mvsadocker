"""Step 1: classify one already-extracted, single-vehicle frame into one of
UVH-26's 14 native subtype classes.

This is intentionally NOT a detector wrapper in the scripts/detector.py
sense -- there is no tracking, no ROI, no multi-object handling. Each input
image is assumed to already be a single identified vehicle (see
INPUT_REQUIREMENTS.md). The model is still run in its native
detection/predict mode (UVH-26 has no separate classification head), and
this module just reduces "however many boxes came back" to one label per
frame: the highest-confidence box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from subtype_config import SubtypeConfig

logger = logging.getLogger("vehicle_subtype_classifier")


@dataclass
class FrameClassification:
    frame_path: str
    label: str
    confidence: float
    bbox: Optional[List[float]]
    num_detections: int


class SubtypeClassifier:
    def __init__(self, config: SubtypeConfig):
        from ultralytics import YOLO

        if not config.model.path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {config.model.path}")

        self.config = config
        self.model = YOLO(str(config.model.path))
        self._validate_class_names()

    def _validate_class_names(self) -> None:
        """Fail loudly if the loaded weights' own class names don't match
        what this config expects, rather than silently mislabeling frames
        (this bit Layer 1 -- see config/layer1_class_mapping.yaml's notes
        on uvh_classes.txt casing not matching the real weights)."""
        model_names = set(self.model.names.values())
        expected = set(self.config.expected_classes)
        if model_names != expected:
            missing = expected - model_names
            extra = model_names - expected
            raise ValueError(
                "Loaded model's class names don't match expected_classes in "
                f"config/vehicle_subtype_config.yaml. Missing: {sorted(missing)}. "
                f"Unexpected: {sorted(extra)}. Re-verify against the weights' own "
                "model.names before trusting this pipeline's output."
            )

    def classify_frame(self, frame: np.ndarray, frame_path: str) -> FrameClassification:
        results = self.model.predict(
            frame,
            conf=self.config.model.confidence,
            imgsz=self.config.model.imgsz,
            iou=self.config.model.iou,
            agnostic_nms=self.config.model.agnostic_nms,
            device=self.config.model.device,
            verbose=False,
        )
        boxes = results[0].boxes if results else None

        if boxes is None or len(boxes) == 0:
            return FrameClassification(
                frame_path=frame_path,
                label=self.config.no_detection_label,
                confidence=0.0,
                bbox=None,
                num_detections=0,
            )

        confs = boxes.conf.tolist()
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        best_cls_id = int(boxes.cls.tolist()[best_idx])
        best_bbox = boxes.xyxy.tolist()[best_idx]

        if len(boxes) > 1:
            logger.debug(
                "%s: %d boxes returned, keeping highest-confidence only "
                "(%.3f) -- frame is expected to contain one vehicle",
                frame_path,
                len(boxes),
                best_conf,
            )

        return FrameClassification(
            frame_path=frame_path,
            label=self.model.names[best_cls_id],
            confidence=best_conf,
            bbox=[round(v, 2) for v in best_bbox],
            num_detections=len(boxes),
        )
