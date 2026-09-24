
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("uvh_test.filters")

class DetectionFilter:
    def __init__(self, min_area: float, min_width: float, min_height: float, class_names: Dict[int, str], valid_classes: List[str] = None):
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.valid_classes = set(valid_classes) if valid_classes else None
        self.class_names = class_names

    def filter_boxes(self, boxes: Any) -> Tuple[Any, dict]:
        stats = {
            "raw": len(boxes) if boxes is not None else 0,
            "low_conf": 0,
            "too_small": 0,
            "invalid_class": 0,
            "passed": 0,
            # Per-class breakdown of every disposition, keyed by class name.
            "per_class": {},
        }

        def _bump(cls_name: str, bucket: str) -> None:
            entry = stats["per_class"].setdefault(
                cls_name, {"raw": 0, "low_conf": 0, "too_small": 0, "invalid_class": 0, "passed": 0}
            )
            entry[bucket] += 1

        if boxes is None or len(boxes) == 0:
            return boxes, stats

        passed_indices = []

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else None

        for i, (box, cls_id) in enumerate(zip(xyxy, clss)):
            class_name = self.class_names.get(cls_id, "Unknown")
            conf = float(confs[i]) if confs is not None else None
            _bump(class_name, "raw")

            if self.valid_classes and class_name not in self.valid_classes:
                stats["invalid_class"] += 1
                _bump(class_name, "invalid_class")
                continue

            w = box[2] - box[0]
            h = box[3] - box[1]
            area = w * h

            if area < self.min_area or w < self.min_width or h < self.min_height:
                stats["too_small"] += 1
                _bump(class_name, "too_small")
                continue

            passed_indices.append(i)
            _bump(class_name, "passed")

        stats["passed"] = len(passed_indices)

        if len(passed_indices) == len(boxes):
            return boxes, stats

        filtered_boxes = boxes[passed_indices]
        return filtered_boxes, stats
