"""Layer 1 detection: two YOLO models run per frame, merged into one tagged
detection stream.

Reuses scripts/detector.py::PersonDetector completely unmodified -- it is
already model-agnostic (any weights path, any requested class id list), so
no new detector wrapper is needed, only an orchestrator that runs two
instances of it and tags each result with its source and native class name
(read from the model's own ``.model.names``, never assumed -- see the
spec's "do not assume class names" requirement).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from detector import PersonDetector  # noqa: E402

from config import DetectorModelConfig, Layer1Config  # noqa: E402
from layer1_models import SourcedDetection  # noqa: E402

logger = logging.getLogger("mvsa.traffic_layer1")

SOURCE_UVH26 = "uvh26"
SOURCE_COCO = "coco"


class MultiSourceDetector:
    """Wraps a UVH-26 PersonDetector (all native vehicle-shaped classes) and
    a stock-COCO PersonDetector (person only), and runs both against the
    same frame each call, returning one merged, source-tagged list.
    """

    def __init__(self, config: Layer1Config):
        self._config = config
        self._uvh26 = self._build(config.models.uvh26, config, floor_conf=self._floor_confidence(config))
        self._coco = self._build(config.models.person, config, floor_conf=self._floor_confidence(config))
        self._uvh26_names: Dict[int, str] = self._uvh26.model.names
        self._coco_names: Dict[int, str] = self._coco.model.names
        logger.info(
            "Layer 1 detectors ready: uvh26 classes=%s, coco classes=%s",
            list(self._uvh26_names.values()),
            [self._coco_names[c] for c in (config.models.person.class_ids or [])],
        )

    @staticmethod
    def _floor_confidence(config: Layer1Config) -> float:
        """The lowest threshold any bucket needs -- passed to YOLO itself
        (which only supports one global conf per .track() call). Real,
        possibly-higher per-bucket thresholds are enforced afterwards, in
        the pipeline, once a detection is mapped to its broad bucket -- see
        config.py::ConfidenceConfig.threshold_for.
        """
        thresholds = [config.confidence.default, *config.confidence.per_bucket.values()]
        return min(thresholds)

    @staticmethod
    def _build(model_cfg: DetectorModelConfig, config: Layer1Config, floor_conf: float) -> PersonDetector:
        return PersonDetector(
            model_path=config.resolve(model_cfg.yolo_weights),
            device=model_cfg.device,
            confidence=floor_conf,
            imgsz=model_cfg.imgsz,
            classes=model_cfg.class_ids,
            tracker=model_cfg.tracker,
            iou=model_cfg.iou,
            agnostic_nms=model_cfg.agnostic_nms,
        )

    def detect(self, frame: np.ndarray) -> List[SourcedDetection]:
        results: List[SourcedDetection] = []

        for det in self._uvh26.track_frame(frame):
            results.append(
                SourcedDetection(
                    source=SOURCE_UVH26,
                    native_class=self._uvh26_names.get(det.cls_id, str(det.cls_id)),
                    bbox=det.bbox,
                    conf=det.conf,
                    raw_track_id=det.track_id,
                )
            )

        for det in self._coco.track_frame(frame):
            results.append(
                SourcedDetection(
                    source=SOURCE_COCO,
                    native_class=self._coco_names.get(det.cls_id, str(det.cls_id)),
                    bbox=det.bbox,
                    conf=det.conf,
                    raw_track_id=det.track_id,
                )
            )

        return results

    def reset(self) -> None:
        self._uvh26.reset()
        self._coco.reset()
