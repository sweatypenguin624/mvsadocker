"""Face-image quality gating.

CCTV faces are frequently too small, blurry, dark, or off-angle to support a
reliable age/gender estimate. This module is deliberately independent of any
specific face-detection/demographic model: it just scores an image region
(given a bbox and a detector confidence) and decides whether the region is
usable at all. All thresholds are read from config -- nothing here is a
"magic number" the user cannot tune.

The composite "quality score" is a heuristic, not a calibrated probability.
It exists because InsightFace's genderage model does not provide anything
resembling a real per-prediction confidence, and the project explicitly
calls for "a clearly defined quality score instead" in that situation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from config_loader import FaceQualityConfig
from models import BBox


@dataclass
class FaceQualityResult:
    width: int
    height: int
    sharpness: float
    brightness: float
    det_score: float
    score: float  # composite heuristic quality score in [0, 1]
    passed: bool
    reasons: List[str] = field(default_factory=list)


class FaceQualityAssessor:
    """Scores a face crop against configurable minimum-quality thresholds."""

    def __init__(self, config: FaceQualityConfig):
        self.config = config

    def assess(
        self, image_bgr: np.ndarray, face_bbox: BBox, det_score: float
    ) -> FaceQualityResult:
        """Assess the quality of a face located at ``face_bbox`` within
        ``image_bgr`` (typically a person crop, so ``face_bbox`` is in that
        crop's local coordinates).
        """
        reasons: List[str] = []
        img_h, img_w = image_bgr.shape[:2]

        x1, y1, x2, y2 = face_bbox
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(img_w, int(round(x2)))
        y2 = min(img_h, int(round(y2)))

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        if width <= 0 or height <= 0:
            return FaceQualityResult(
                width=width,
                height=height,
                sharpness=0.0,
                brightness=0.0,
                det_score=det_score,
                score=0.0,
                passed=False,
                reasons=["invalid_crop"],
            )

        if width < self.config.min_face_width:
            reasons.append(f"face_too_narrow ({width}px < {self.config.min_face_width}px)")
        if height < self.config.min_face_height:
            reasons.append(f"face_too_short ({height}px < {self.config.min_face_height}px)")
        if det_score < self.config.min_face_confidence:
            reasons.append(
                f"low_detection_confidence ({det_score:.2f} < {self.config.min_face_confidence:.2f})"
            )

        crop = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.config.min_sharpness:
            reasons.append(f"too_blurry (sharpness={sharpness:.1f} < {self.config.min_sharpness:.1f})")

        brightness = float(gray.mean())
        if self.config.check_brightness:
            if brightness < self.config.min_brightness:
                reasons.append(f"too_dark (brightness={brightness:.1f})")
            elif brightness > self.config.max_brightness:
                reasons.append(f"too_bright (brightness={brightness:.1f})")

        score = self._composite_score(width, height, sharpness, brightness, det_score)
        passed = len(reasons) == 0

        return FaceQualityResult(
            width=width,
            height=height,
            sharpness=sharpness,
            brightness=brightness,
            det_score=det_score,
            score=score,
            passed=passed,
            reasons=reasons,
        )

    def _composite_score(
        self, width: int, height: int, sharpness: float, brightness: float, det_score: float
    ) -> float:
        """Heuristic 0-1 quality score, NOT a calibrated probability.

        Each component is normalized against a "comfortably above threshold"
        reference (2x the configured minimum) and averaged. This rewards
        margin above the pass/fail line rather than just a binary pass.
        """
        target_area = max(self.config.min_face_width, 1) * max(self.config.min_face_height, 1) * 4
        size_score = min(1.0, (width * height) / target_area) if target_area > 0 else 0.0

        sharpness_ref = max(self.config.min_sharpness, 1.0) * 2
        sharpness_score = min(1.0, sharpness / sharpness_ref)

        brightness_mid = (self.config.min_brightness + self.config.max_brightness) / 2.0
        brightness_range = max((self.config.max_brightness - self.config.min_brightness) / 2.0, 1.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - brightness_mid) / brightness_range)

        det_component = max(0.0, min(1.0, det_score))

        return float(np.mean([size_score, sharpness_score, brightness_score, det_component]))
