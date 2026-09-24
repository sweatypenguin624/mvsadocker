"""Per-stage Goods Vehicle classifiers (spec sections 6-9): Broad, LCV,
Heavy Truck Configuration. Deliberately three separate objects, not one
8-class monolith (spec section 1).

Each stage tries, in order:
  1. A fine-tuned torchvision checkpoint at stages.<name>.weights_path, if
     configured and the file exists (spec section 18: transfer learning
     from a pretrained backbone).
  2. A deterministic geometry heuristic (bbox aspect ratio + size) as an
     HONEST PLACEHOLDER -- there is no labeled goods_dataset yet (spec
     section 16), so there is nothing to train on. This is not a trained
     model and should not be reported as one; it exists so the pipeline is
     runnable end-to-end today and so every stage has a single swap point
     (drop a checkpoint at weights_path, no pipeline code changes) once
     scripts/traffic_layer2/dataset_export.py's output has been labeled and
     a stage has been trained.

Every predict() call returns a full probability dict over the stage's
labels (never a bare argmax) so temporal aggregation and the
review/uncertain policy have real numbers to work with (spec sections 14,
15, 20).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("mvsa.traffic_layer2")

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class StageClassifierBase:
    """predict_crop() takes an already-cropped BGR image (Layer 1's crops.py
    saves tight crops, not full frames) plus that crop's ORIGINAL bbox in
    source-frame pixel space (used only for aspect-ratio/size features, not
    for re-cropping -- the crop image is already cropped).
    """

    labels: List[str]

    def predict_crop(self, crop_bgr: np.ndarray, source_bbox: List[float]) -> Dict[str, float]:
        raise NotImplementedError


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float64)
    values = values - values.max()
    exp = np.exp(values)
    exp = exp / exp.sum()
    return {k: float(v) for k, v in zip(scores.keys(), exp)}


class HeuristicAspectRatioClassifier(StageClassifierBase):
    """Placeholder classifier: scores each label by how close the crop's
    bbox aspect ratio (width/height) and normalized area are to that
    label's expected range, then softmaxes the (negative) distance into a
    probability distribution. Ranges are rough real-world proportions, not
    fitted to any dataset -- see module docstring.
    """

    # label -> (aspect_ratio_center, aspect_ratio_tolerance, rel_area_center)
    # rel_area is bbox area / frame area, a crude size-vs-camera-distance
    # proxy since Layer 2 has no calibrated real-world scale.
    _PROFILES: Dict[str, Dict[str, tuple]] = {
        "broad": {
            "Goods 3 Wheeler": (1.15, 0.35, 0.03),
            "LCV": (1.7, 0.4, 0.05),
            "Heavy Truck": (2.3, 0.6, 0.09),
            "Tractor": (1.3, 0.4, 0.04),
        },
        "lcv": {
            "Mini LCV": (1.5, 0.35, 0.03),
            "LCV": (1.9, 0.4, 0.06),
        },
        "heavy_truck": {
            "2 Axle Truck": (2.0, 0.35, 0.07),
            "3 Axle Truck": (2.5, 0.4, 0.10),
            "MAV": (3.0, 0.5, 0.13),
        },
    }

    def __init__(self, stage_name: str, labels: List[str]):
        if stage_name not in self._PROFILES:
            raise ValueError(f"no heuristic profile for stage {stage_name!r}")
        self.stage_name = stage_name
        self.labels = labels
        self.profiles = {k: v for k, v in self._PROFILES[stage_name].items() if k in labels}

    def predict_crop(self, crop_bgr: np.ndarray, source_bbox: List[float]) -> Dict[str, float]:
        x1, y1, x2, y2 = source_bbox
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        aspect = w / h
        # 1920x1080 is the Layer 1 reference frame (config/layer1_config.yaml
        # roi.reference_width/height) -- used here only as the denominator
        # for a rough size-vs-camera-distance proxy, not assumed exact.
        rel_area = (w * h) / float(1920 * 1080)

        scores = {}
        for label, (ar_center, ar_tol, area_center) in self.profiles.items():
            ar_dist = abs(aspect - ar_center) / ar_tol
            area_dist = abs(rel_area - area_center) / max(area_center, 1e-3)
            scores[label] = -(ar_dist + 0.5 * area_dist)
        return _softmax(scores)


class HeuristicUniformClassifier(StageClassifierBase):
    """Honest "no signal" placeholder for a question geometry genuinely
    can't answer (e.g. bus livery/branding -- BRTC vs City/Private vs
    School Bus all share the same box shape). Returns a near-uniform
    distribution rather than fabricate a geometry-based guess, so the
    pipeline's min_confidence gate reliably routes these to "Uncertain ..."
    instead of reporting a confident-looking wrong answer.
    """

    def __init__(self, labels: List[str]):
        self.labels = labels

    def predict_crop(self, crop_bgr: np.ndarray, source_bbox: List[float]) -> Dict[str, float]:
        n = len(self.labels)
        base = 1.0 / n
        # Tiny deterministic jitter (not random) so ties break consistently
        # run-to-run, without asserting any real preference.
        return {label: base + (0.001 * (i - n / 2)) for i, label in enumerate(self.labels)}


class TorchStageClassifier(StageClassifierBase):
    """Transfer-learning classifier: ImageNet-pretrained torchvision
    backbone + a linear head loaded from a fine-tuned checkpoint. Only
    instantiated when weights_path is configured and exists -- see
    get_stage_classifier() below.
    """

    def __init__(self, weights_path: Path, labels: List[str], device: str = "cpu"):
        import torch
        import torchvision

        self.labels = labels
        self.device = device
        self.torch = torch

        backbone = torchvision.models.resnet18(weights=None)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(labels))
        state = torch.load(weights_path, map_location=device)
        backbone.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        backbone.eval()
        self.model = backbone.to(device)

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_crop(self, crop_bgr: np.ndarray, source_bbox: List[float]) -> Dict[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return {label: 1.0 / len(self.labels) for label in self.labels}
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)[0]
            probs = self.torch.softmax(logits, dim=0).cpu().numpy()
        return {label: float(p) for label, p in zip(self.labels, probs)}


def get_stage_classifier(stage_name: str, labels: List[str], weights_path: Optional[str], project_root: Path) -> StageClassifierBase:
    if weights_path:
        resolved = Path(weights_path)
        if not resolved.is_absolute():
            resolved = project_root / weights_path
        if resolved.exists():
            try:
                logger.info("Stage %r: loading trained checkpoint %s", stage_name, resolved)
                return TorchStageClassifier(resolved, labels)
            except Exception as e:
                logger.error("Stage %r: failed to load %s (%s) -- falling back to heuristic", stage_name, resolved, e)
        else:
            logger.warning("Stage %r: configured weights_path %s does not exist -- using heuristic placeholder", stage_name, resolved)
    else:
        logger.info("Stage %r: no weights_path configured -- using heuristic placeholder (see module docstring)", stage_name)

    if stage_name in HeuristicAspectRatioClassifier._PROFILES:
        return HeuristicAspectRatioClassifier(stage_name, labels)
    return HeuristicUniformClassifier(labels)


def load_crop_image(path: Path) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    img = cv2.imread(str(path))
    return img
