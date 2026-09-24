"""Cheap appearance fingerprints, used to stop bad temporal stitches.

Motivation (a real over-merge caught on GPU footage): two scooters -- one
silver carrying two riders, one red carrying one -- travelled the same lane
on the same line about 1.5s apart. The second was stitched onto the first
as a "resumed" track, because geometrically it was a near-perfect
continuation: same path, same speed, landing within 60px of the predicted
position. No motion model can separate those two. Their COLOUR separates
them instantly.

An HSV histogram is deliberately crude. It is not re-identification and
will not tell two silver scooters apart -- it exists only to veto stitches
between obviously different-looking vehicles, which is where the motion
model is blind. Cheap enough to run on every kept observation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Coarse bins: robust to compression noise, lighting and slight pose
# change, while still separating red from silver from yellow.
H_BINS, S_BINS = 12, 4


def compute(frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    """Normalised hue-saturation histogram of the vehicle box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    # Trim the border: box edges include road surface, which is the same
    # grey for every vehicle and washes out the real colour signal.
    dx, dy = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    crop = frame[y1 + dy: y2 - dy, x1 + dx: x2 - dx]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
    total = hist.sum()
    if total <= 0:
        return None
    return (hist / total).flatten().astype(np.float32)


def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Histogram correlation in [-1, 1]; None when either side is missing.

    None means "no opinion" -- callers must not treat a missing fingerprint
    as a mismatch, or vehicles too small to fingerprint would never stitch.
    """
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


class RunningAppearance:
    """Mean fingerprint over a track's observations.

    Averaging rather than keeping the latest: a single frame can be caught
    mid-occlusion or badly lit, and one bad frame should not decide whether
    two tracks merge.
    """

    def __init__(self, max_samples: int = 30):
        self.max_samples = max_samples
        self._sum: Optional[np.ndarray] = None
        self._n = 0

    def add(self, hist: Optional[np.ndarray]) -> None:
        if hist is None or self._n >= self.max_samples:
            return
        self._sum = hist.copy() if self._sum is None else self._sum + hist
        self._n += 1

    @property
    def mean(self) -> Optional[np.ndarray]:
        if self._sum is None or self._n == 0:
            return None
        return (self._sum / self._n).astype(np.float32)
