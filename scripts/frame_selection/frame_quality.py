"""Per-frame image quality scoring.

Generic, content-agnostic heuristics for scoring how "good" a single video
frame looks as a standalone image: how sharp it is, how well exposed, how
much contrast and colour information it carries, and how much visual
information (entropy) it holds. Deliberately independent of any
video-reading logic -- ``FrameQualityScorer.score`` takes a decoded BGR
frame (or a plain photo) and nothing else, so it can be reused outside
this package.

The composite "score" is a heuristic in [0, 100], not a calibrated
probability -- there is no ground truth for "how good does this frame
look" to calibrate against. Each component is normalized against a
reference value at which it saturates to 1.0; the references were picked
against typical 720p-1080p footage and are exposed as constructor
parameters so callers can retune them for very different source material
(e.g. low-light CCTV, where the default sharpness reference is too high).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class QualityWeights:
    """Relative importance of each component in the composite score.

    Values are relative, not required to sum to 1 -- ``normalized()``
    rescales them. Defaults favour sharpness and exposure, since a
    blurry or blown-out frame is unusable regardless of how colourful or
    detailed it otherwise is.
    """

    sharpness: float = 0.35
    exposure: float = 0.25
    contrast: float = 0.15
    colorfulness: float = 0.10
    entropy: float = 0.15

    def normalized(self) -> "QualityWeights":
        total = self.sharpness + self.exposure + self.contrast + self.colorfulness + self.entropy
        if total <= 0:
            raise ValueError("quality weights must sum to a positive number")
        return QualityWeights(
            sharpness=self.sharpness / total,
            exposure=self.exposure / total,
            contrast=self.contrast / total,
            colorfulness=self.colorfulness / total,
            entropy=self.entropy / total,
        )


@dataclass
class FrameQuality:
    """Raw metrics plus the normalized sub-scores and composite score for
    one frame. Raw values are kept so callers can debug/retune, not just
    consume the final number."""

    sharpness: float          # variance of Laplacian; higher = crisper edges
    brightness: float         # mean luminance, 0-255
    contrast: float           # std of luminance, 0-255
    clipped_fraction: float   # fraction of pixels blown out or crushed to black
    colorfulness: float       # Hasler & Susstrunk (2003) metric
    entropy: float            # Shannon entropy of the luminance histogram, bits
    sharpness_score: float
    exposure_score: float
    contrast_score: float
    colorfulness_score: float
    entropy_score: float
    score: float               # composite quality score, 0-100

    def as_dict(self) -> Dict[str, float]:
        return {
            "sharpness": round(self.sharpness, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "clipped_fraction": round(self.clipped_fraction, 4),
            "colorfulness": round(self.colorfulness, 2),
            "entropy": round(self.entropy, 3),
            "sharpness_score": round(self.sharpness_score, 4),
            "exposure_score": round(self.exposure_score, 4),
            "contrast_score": round(self.contrast_score, 4),
            "colorfulness_score": round(self.colorfulness_score, 4),
            "entropy_score": round(self.entropy_score, 4),
            "score": round(self.score, 2),
        }


def _saturating(value: float, ref: float) -> float:
    """Linearly maps ``value`` to [0, 1], saturating at 1.0 once it
    reaches ``ref``. Simple and monotonic -- good enough for a heuristic
    where we only care about relative ranking, not a precise curve."""
    if ref <= 0:
        return 0.0
    return float(max(0.0, min(1.0, value / ref)))


class FrameQualityScorer:
    """Scores a single BGR frame. Stateless (holds only tunable
    parameters) and safe to reuse across many frames/threads."""

    def __init__(
        self,
        weights: Optional[QualityWeights] = None,
        sharpness_ref: float = 350.0,
        contrast_ref: float = 60.0,
        colorfulness_ref: float = 60.0,
    ):
        self.weights = (weights or QualityWeights()).normalized()
        self.sharpness_ref = sharpness_ref
        self.contrast_ref = contrast_ref
        self.colorfulness_ref = colorfulness_ref

    def score(self, frame_bgr: np.ndarray) -> FrameQuality:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr is empty")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # A small pre-blur before the Laplacian keeps sharpness measuring
        # genuine multi-pixel edge structure rather than single-pixel
        # sensor/compression noise -- on heavily-compressed CCTV footage,
        # raw (unblurred) Laplacian variance can be 10-20x inflated by
        # block artifacts alone, which makes a noisy-but-blurry frame
        # outscore a genuinely sharp one.
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpness = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        clipped_fraction = float(
            (np.count_nonzero(gray <= 3) + np.count_nonzero(gray >= 252)) / gray.size
        )
        colorfulness = self._colorfulness(frame_bgr)
        entropy = self._entropy(gray)

        sharpness_score = _saturating(sharpness, self.sharpness_ref)
        exposure_score = self._exposure_score(brightness, clipped_fraction)
        contrast_score = _saturating(contrast, self.contrast_ref)
        colorfulness_score = _saturating(colorfulness, self.colorfulness_ref)
        entropy_score = min(1.0, entropy / 8.0)  # 8 bits = max possible entropy

        w = self.weights
        composite = (
            w.sharpness * sharpness_score
            + w.exposure * exposure_score
            + w.contrast * contrast_score
            + w.colorfulness * colorfulness_score
            + w.entropy * entropy_score
        )

        return FrameQuality(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            clipped_fraction=clipped_fraction,
            colorfulness=colorfulness,
            entropy=entropy,
            sharpness_score=sharpness_score,
            exposure_score=exposure_score,
            contrast_score=contrast_score,
            colorfulness_score=colorfulness_score,
            entropy_score=entropy_score,
            score=composite * 100.0,
        )

    @staticmethod
    def _colorfulness(frame_bgr: np.ndarray) -> float:
        """Hasler & Susstrunk (2003) colorfulness metric: combines the
        spread and mean of the rg/yb opponent-colour channels. Cheap and
        correlates well with human colourfulness judgements."""
        b, g, r = cv2.split(frame_bgr.astype("float32"))
        rg = r - g
        yb = 0.5 * (r + g) - b
        rg_std, rg_mean = float(rg.std()), float(rg.mean())
        yb_std, yb_mean = float(yb.std()), float(yb.mean())
        std_root = float(np.sqrt(rg_std ** 2 + yb_std ** 2))
        mean_root = float(np.sqrt(rg_mean ** 2 + yb_mean ** 2))
        return std_root + 0.3 * mean_root

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum()
        if total <= 0:
            return 0.0
        p = hist / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _exposure_score(brightness: float, clipped_fraction: float) -> float:
        """Rewards mid-range brightness with little blown-out/crushed
        detail, independent of scene content. A frame can be perfectly
        sharp and still useless if it's a silhouette or a white-out."""
        ideal = 128.0
        brightness_score = max(0.0, 1.0 - abs(brightness - ideal) / ideal)
        clipping_score = max(0.0, 1.0 - clipped_fraction * 4.0)  # >=25% clipped -> 0
        return 0.6 * brightness_score + 0.4 * clipping_score
