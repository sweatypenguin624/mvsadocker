"""Body/clothing-based age/gender estimation -- a second, independent
fallback demographic estimator for tracks whose face never resolves.

WHY THIS EXISTS: on this camera (steep overhead angle over a wide
intersection, ~1080p), InsightFace's face pipeline (demographics.py)
produces a usable face for only a small fraction of counted tracks -- see
demographic_test_report.csv from any real run. The person's full body
(clothing silhouette, colors, build) is still resolvable even when the face
is a handful of blurred pixels, so this module runs a pedestrian-attribute
model over the *whole-body* crop instead of the face crop.

MODEL PROVENANCE (read before touching thresholds/indices below):
This wraps an ONNX export of PaddleClas's PULC "person_attribute" model --
PPLCNet_x1_0, Apache-2.0, trained on the PA-100K pedestrian-attribute
dataset (real CCTV/surveillance imagery, not face photos, which is exactly
this camera's domain). Source model:
    https://paddleclas.bj.bcebos.com/models/PULC/person_attribute_infer.tar
Converted once, offline, via `paddle2onnx` into
models/person_attribute/person_attribute.onnx (see README installation
section) -- inference here only ever uses onnxruntime, matching the
InsightFace path; PaddlePaddle itself is never a runtime dependency of this
project.

The model outputs 26 sigmoid attribute probabilities per PA-100K's schema.
This module only uses three of them (gender + the 3-way age bucket); the
other 23 (upper/lower clothing style, bags, hat, glasses, pose, boots) are
exactly the clothing/appearance signal MVSA uses for the estimate, but are
not currently surfaced as separate output columns. The exact index layout
and decision rule (gender: index 22, threshold 0.5; age: argmax over
indices 19-21) is reproduced verbatim from PaddleClas's own postprocessing
code (ppcls/data/postprocess/attr_rec.py::PersonAttribute), and was
independently verified against PaddleClas's published demo predictions for
090004.jpg ("Male","Age18-60") and 090007.jpg ("Female","Age18-60") before
being trusted here -- do not change the index mapping without re-verifying
against those two images.

CONFIDENCE NOTE (see demographics.py for the parallel discussion on the
face path): the model's sigmoid outputs ARE genuine per-class probabilities
(unlike InsightFace's genderage head), but they are this model's own
training-time calibration on PA-100K, a different population and camera
distribution than this specific intersection. They are reported and used
exactly like demographics.py's confidence_proxy -- a trust signal for
weighting votes in aggregation.py, not a validated probability of
correctness on this footage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config_loader import BodyAttributesConfig
from models import GENDER_FEMALE, GENDER_MALE, PA100K_AGE_GROUP_MAP

logger = logging.getLogger("mvsa")

# Fixed by the PA-100K/PULC label schema -- see module docstring.
_GENDER_INDEX = 22
_AGE_INDICES = (19, 20, 21)
_AGE_LABELS = ["AgeLess18", "Age18-60", "AgeOver60"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class BodyAttributeEstimate:
    gender: str
    gender_confidence: Optional[float]
    age_group: str
    age_confidence: Optional[float]
    quality_score: float
    usable: bool
    # A single combined trust weight for this sample -- 0.5*gender_confidence
    # + 0.5*age_confidence, mirroring demographics.py's confidence_proxy so
    # aggregation.py can weight votes from both estimators the same way.
    # None when unusable.
    confidence_proxy: Optional[float] = None


class BodyAttributeAnalyzer:
    """Whole-body-crop age/gender estimation for a single person crop.

    Usage:
        analyzer = BodyAttributeAnalyzer(config.body_attributes)
        result = analyzer.analyze(person_crop_bgr)
    """

    def __init__(self, config: BodyAttributesConfig, project_root: Path):
        self.config = config
        self._session = self._load_model(project_root)

    def _load_model(self, project_root: Path):
        # Imported lazily so importing this module never requires
        # onnxruntime unless body_attributes are actually enabled.
        import onnxruntime as ort

        model_path = Path(self.config.model_path)
        if not model_path.is_absolute():
            model_path = (project_root / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"body_attributes.model_path not found: {model_path} -- see README "
                "installation section for how to download/convert the PP-LCNet "
                "person-attribute model."
            )

        session = ort.InferenceSession(str(model_path), providers=self._providers())
        self._input_name = session.get_inputs()[0].name
        logger.info(
            "Loaded body-attribute model from %s (device=%s, input=%dx%d)",
            model_path,
            self.config.device,
            self.config.input_width,
            self.config.input_height,
        )
        return session

    def _providers(self) -> List[str]:
        if self.config.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = normalized.transpose(2, 0, 1)
        return chw[None, ...].astype(np.float32)

    def analyze(self, person_crop_bgr: np.ndarray) -> BodyAttributeEstimate:
        """Estimate gender/age-bucket from a full-body person crop.

        Unlike demographics.py's face path, there is no separate detector
        stage to fail -- the only gate is a minimum crop size (config
        min_crop_width/min_crop_height), below which the model's own
        224px-ish receptive field has nothing meaningful left to work with.
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return self._unusable_result()

        h, w = person_crop_bgr.shape[:2]
        if w < self.config.min_crop_width or h < self.config.min_crop_height:
            return self._unusable_result()

        x = self._preprocess(person_crop_bgr)
        probs = self._session.run(None, {self._input_name: x})[0][0]

        female_prob = float(probs[_GENDER_INDEX])
        gender = GENDER_FEMALE if female_prob > self.config.gender_threshold else GENDER_MALE
        gender_confidence = female_prob if gender == GENDER_FEMALE else 1.0 - female_prob

        age_probs = [float(probs[i]) for i in _AGE_INDICES]
        age_idx = int(np.argmax(age_probs))
        age_group = PA100K_AGE_GROUP_MAP[_AGE_LABELS[age_idx]]
        age_confidence = age_probs[age_idx]

        quality_score = min(
            1.0, (w * h) / float(self.config.min_crop_width * self.config.min_crop_height * 4)
        )

        return BodyAttributeEstimate(
            gender=gender,
            gender_confidence=gender_confidence,
            age_group=age_group,
            age_confidence=age_confidence,
            quality_score=quality_score,
            usable=True,
            confidence_proxy=0.5 * gender_confidence + 0.5 * age_confidence,
        )

    @staticmethod
    def _unusable_result() -> BodyAttributeEstimate:
        return BodyAttributeEstimate(
            gender="unknown",
            gender_confidence=None,
            age_group="unknown",
            age_confidence=None,
            quality_score=0.0,
            usable=False,
        )
