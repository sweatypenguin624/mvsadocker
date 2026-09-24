"""YOLO11x + ByteTrack wrapper.

Frames are fed to Ultralytics one at a time (rather than handing Ultralytics
the video path directly) so the pipeline controls frame indexing, timestamps,
and can seek to an exact resume point via video_utils.VideoStreamReader.
``persist=True`` keeps ByteTrack's internal track state alive across calls
for the same PersonDetector instance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from models import Detection

logger = logging.getLogger("mvsa")


class PersonDetector:
    """Runs YOLO11x detection + ByteTrack tracking on individual frames."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        confidence: float,
        imgsz: int,
        classes: List[int],
        tracker: str = "bytetrack.yaml",
        iou: float = 0.7,
        agnostic_nms: bool = False,
    ):
        # Imported lazily so importing this module (e.g. from tests via
        # other modules) never requires ultralytics/torch to be installed.
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.classes = classes
        self.tracker = tracker
        # 0.7 and False are Ultralytics' own .track()/.predict() defaults --
        # kept as this class's defaults too so existing callers (the
        # pedestrian pipeline, the vehicle pilot) are unaffected unless they
        # explicitly opt in. agnostic_nms=True runs NMS across ALL classes
        # rather than per-class, which is the standard fix for a detector
        # emitting two overlapping boxes for the same physical object under
        # different class labels (Ultralytics' default per-class NMS never
        # suppresses that case, since the two boxes don't compete with each
        # other within their own class).
        self.iou = iou
        self.agnostic_nms = agnostic_nms

        logger.info(
            "Loaded YOLO model %s (device=%s, imgsz=%d, conf=%.2f, iou=%.2f, agnostic_nms=%s, "
            "classes=%s, tracker=%s)",
            model_path.name,
            device,
            imgsz,
            confidence,
            iou,
            agnostic_nms,
            classes,
            tracker,
        )

    def track_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run detection + tracking on a single BGR frame and return
        parsed Detection objects (all requested classes, each with a
        ByteTrack ``track_id`` when available).
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            iou=self.iou,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        if not results:
            return []
        return self.parse_detections(results[0])

    @staticmethod
    def parse_detections(result) -> List[Detection]:
        detections: List[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        clss = boxes.cls.tolist()
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(xyxy)

        for bbox, conf, cls_id, track_id in zip(xyxy, confs, clss, ids):
            detections.append(
                Detection(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    conf=float(conf),
                    cls_id=int(cls_id),
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return detections

    def reset(self) -> None:
        """Reset ByteTrack's internal state (e.g. before starting a fresh
        video, so track IDs don't carry over from a previous run).
        """
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            predictor.trackers = []
