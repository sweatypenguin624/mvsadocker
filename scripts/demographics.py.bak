"""Second-stage demographic estimation, independent of YOLO.

DemographicAnalyzer wraps InsightFace's ``buffalo_l`` model pack, using only
its face-detection and genderage sub-models (recognition/landmark models are
explicitly excluded via ``allowed_modules`` to save memory and load time,
since MVSA never does face recognition/re-identification).

CRITICAL CONFIDENCE NOTE (read before trusting gender_confidence/age_confidence):
InsightFace's genderage attribute model outputs a hard gender argmax and a
scalar age value. It does NOT expose a calibrated per-prediction confidence
for either. Treating YOLO's detection confidence, or "the model ran without
error", as a stand-in for age/gender confidence would be misleading. Instead,
this module reports a documented, engineering-defined *confidence_proxy* =
the average of (a) the face detector's det_score and (b) the face-image
quality score from face_quality.py. It is explicitly NOT a probability that
the prediction is correct -- it is a proxy trust signal used for weighting
in aggregation.py, and is labeled as such everywhere it is surfaced (code,
CSV headers, README).

Faces that fail quality gating never produce a gender/age value here, even
if InsightFace technically returned one -- "never force a prediction when
the face is unusable" is enforced at this layer, not left to the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2

import numpy as np

from config_loader import DemographicsConfig
from face_quality import FaceQualityAssessor, FaceQualityResult
from models import GENDER_UNKNOWN, INSIGHTFACE_GENDER_MAP, age_to_group

logger = logging.getLogger("mvsa")


@dataclass
class DemographicEstimate:
    """Return type of DemographicAnalyzer.analyze() -- matches the required
    interface structure, plus a few fields the aggregation stage needs.
    """

    gender: str
    gender_confidence: Optional[float]
    age: Optional[float]
    age_group: str
    age_confidence: Optional[float]
    face_quality: float
    face_detected: bool
    usable: bool
    det_score: float

    def to_dict(self) -> dict:
        return {
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "age": self.age,
            "age_group": self.age_group,
            "age_confidence": self.age_confidence,
            "face_quality": self.face_quality,
        }


class DemographicAnalyzer:
    """Face detection + age/gender estimation for a single person crop.

    Usage:
        analyzer = DemographicAnalyzer(config.demographics, quality_assessor)
        result = analyzer.analyze(person_crop_bgr)
    """

    def __init__(self, config: DemographicsConfig, quality_assessor: FaceQualityAssessor):
        self.config = config
        self.quality_assessor = quality_assessor
        self._app = self._load_model()

    def _load_model(self):
        # Imported lazily: importing demographics.py should not require
        # insightface/onnxruntime to be installed unless demographics are
        # actually enabled and used.
        from insightface.app import FaceAnalysis

        root = Path(self.config.insightface_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        app = FaceAnalysis(
            name=self.config.insightface_model_name,
            root=str(root),
            allowed_modules=["detection", "genderage"],
            providers=self._providers(),
        )
        ctx_id = 0 if self.config.device == "cuda" else -1
        det_size = tuple(self.config.det_size)
        app.prepare(ctx_id=ctx_id, det_size=det_size)

        logger.info(
            "Loaded InsightFace model '%s' from %s (device=%s, det_size=%s)",
            self.config.insightface_model_name,
            root,
            self.config.device,
            det_size,
        )
        return app

    def _providers(self) -> List[str]:
        if self.config.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def analyze(self, person_crop_bgr: np.ndarray) -> DemographicEstimate:
        """Detect a face within a person crop and estimate age/gender.

        Returns a DemographicEstimate. If no face is found, or the face
        fails quality gating, gender/age/age_group are "unknown"/None and
        ``usable`` is False -- callers (aggregation.py) must skip such
        samples when computing track-level statistics.
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return self._unusable_result(face_detected=False)

        scale = 4

        upscaled_crop = cv2.resize(
            person_crop_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        faces = self._app.get(upscaled_crop)

        faces = self._app.get(person_crop_bgr)
        if not faces:
            return self._unusable_result(face_detected=False)

        face = self._select_primary_face(faces, person_crop_bgr.shape)
        bbox = face.bbox / scale
        quality: FaceQualityResult = self.quality_assessor.assess(
            person_crop_bgr, tuple(face.bbox.tolist()), float(face.det_score)
        )

        if not quality.passed:
            logger.debug(
                "Face rejected by quality gate: %s", ", ".join(quality.reasons)
            )
            return DemographicEstimate(
                gender=GENDER_UNKNOWN,
                gender_confidence=None,
                age=None,
                age_group="unknown",
                age_confidence=None,
                face_quality=quality.score,
                face_detected=True,
                usable=False,
                det_score=quality.det_score,
            )

        raw_gender = int(getattr(face, "gender", -1))
        gender = INSIGHTFACE_GENDER_MAP.get(raw_gender, GENDER_UNKNOWN)
        age = float(getattr(face, "age", None)) if getattr(face, "age", None) is not None else None

        confidence_proxy = 0.5 * quality.det_score + 0.5 * quality.score

        return DemographicEstimate(
            gender=gender,
            gender_confidence=confidence_proxy,
            age=age,
            age_group=age_to_group(age),
            age_confidence=confidence_proxy,
            face_quality=quality.score,
            face_detected=True,
            usable=True,
            det_score=quality.det_score,
        )

    @staticmethod
    def _select_primary_face(faces: list, crop_shape) -> "object":
        """When multiple faces appear in a person crop (e.g. a bystander in
        the background), prefer the largest face located in the upper half
        of the crop (where a person's head should be), falling back to the
        overall-largest face if none qualify.
        """
        crop_h = crop_shape[0]

        def area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        upper_half = [f for f in faces if (f.bbox[1] + f.bbox[3]) / 2.0 < crop_h * 0.6]
        candidates = upper_half if upper_half else faces
        return max(candidates, key=area)

    @staticmethod
    def _unusable_result(face_detected: bool) -> DemographicEstimate:
        return DemographicEstimate(
            gender=GENDER_UNKNOWN,
            gender_confidence=None,
            age=None,
            age_group="unknown",
            age_confidence=None,
            face_quality=0.0,
            face_detected=face_detected,
            usable=False,
            det_score=0.0,
        )
