"""Class-agnostic vehicle detection for Stage 1.

The UVH-26 model is used purely as a "is there a vehicle here, and where"
detector -- its class label is read only to drop non-vehicle classes
(person), then discarded. This is deliberate: UVH-26's known confusions
(e.g. tarpaulin-covered mini trucks classified as three-wheelers) are
exactly the errors Stage 2 exists to correct, and letting a wrong label
influence tracking or counting would bake that error in permanently.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from stage1_config import DetectorConfig, Stage1Config

logger = logging.getLogger("mvsa.traffic_stage1")


class VehicleDetector:
    def __init__(self, config: Stage1Config):
        cfg: DetectorConfig = config.detector
        self.cfg = cfg
        weights = config.resolve(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(f"Stage 1 detector weights not found: {weights}")

        logger.info("Loading Stage 1 detector: %s", weights)
        self.model = YOLO(str(weights))
        self.names = dict(self.model.names)

        excluded = {n.lower() for n in cfg.exclude_classes}
        included = {n.lower() for n in cfg.include_classes}
        self.keep_class_ids = [
            cid
            for cid, name in self.names.items()
            if str(name).lower() not in excluded
            and (not included or str(name).lower() in included)
        ]
        dropped = [self.names[c] for c in self.names if c not in self.keep_class_ids]
        logger.info(
            "Detector classes collapsed to generic 'vehicle': keeping %d of %d (dropped: %s)",
            len(self.keep_class_ids),
            len(self.names),
            dropped or "none",
        )

        self.dropped_by_size = 0
        self.half = bool(cfg.half) and str(cfg.device) != "cpu" and torch.cuda.is_available()

    def detect_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect on a list of frames. Returns one (boxes_xyxy, confs) per frame."""
        if not frames:
            return []
        results = self.model.predict(
            frames,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            classes=self.keep_class_ids,
            agnostic_nms=self.cfg.agnostic_nms,
            half=self.half,
            verbose=False,
        )
        frame_area = float(frames[0].shape[0] * frames[0].shape[1])
        return [self._size_filter(*self._parse(r), frame_area) for r in results]

    def _size_filter(self, boxes: np.ndarray, confs: np.ndarray, frame_area: float):
        if len(boxes) == 0:
            return boxes, confs
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        frac = areas / max(frame_area, 1.0)
        keep = (frac <= self.cfg.max_box_area_frac) & (frac >= self.cfg.min_box_area_frac)
        dropped = int((~keep).sum())
        if dropped:
            self.dropped_by_size += dropped
            logger.debug("Dropped %d box(es) outside the size sanity bounds", dropped)
        return boxes[keep], confs[keep]

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.detect_batch([frame])[0]

    def detect_with_classes(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like detect(), but also returns each box's native class id.

        Only for annotation/QC (annotate_run.py's per-class HUD) -- Stage 1
        proper stays class-agnostic per this module's docstring, so nothing
        in the counting pipeline calls this.
        """
        results = self.model.predict(
            [frame],
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            classes=self.keep_class_ids,
            agnostic_nms=self.cfg.agnostic_nms,
            half=self.half,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int32)
        frame_area = float(frame.shape[0] * frame.shape[1])
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        frac = areas / max(frame_area, 1.0)
        keep = (frac <= self.cfg.max_box_area_frac) & (frac >= self.cfg.min_box_area_frac)
        return xyxy[keep], confs[keep], cls[keep]

    @staticmethod
    def _parse(result) -> Tuple[np.ndarray, np.ndarray]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return (
            boxes.xyxy.cpu().numpy().astype(np.float32),
            boxes.conf.cpu().numpy().astype(np.float32),
        )
