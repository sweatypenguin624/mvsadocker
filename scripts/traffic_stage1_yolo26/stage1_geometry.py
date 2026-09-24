"""Box geometry shared by fusion, identity dedup and counting.

All boxes are [x1, y1, x2, y2] in source-frame pixels.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

BBox = Sequence[float]


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def iom(a: BBox, b: BBox) -> float:
    """Intersection over the area of the SMALLER box.

    Unlike IoU this stays ~1.0 when a small box is swallowed by a much
    larger one, which is exactly the "one bus split into several panel
    boxes" case that IoU-thresholded NMS leaves untouched.
    """
    inter = intersection(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(area(a), area(b))
    return inter / smaller if smaller > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def union_box(boxes: Iterable[BBox]) -> List[float]:
    boxes = list(boxes)
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def ground_point(box: BBox) -> Tuple[float, float]:
    """Bottom-centre ("where the vehicle touches the road").

    Matches scripts/roi.py::bottom_center so Stage 1 counts on the same
    reference point every other pipeline in this repo does.
    """
    return ((box[0] + box[2]) / 2.0, box[3])


def side_of_line(point: Tuple[float, float], p1: Sequence[float], p2: Sequence[float]) -> float:
    """Signed cross product: >0 one side, <0 the other, 0 exactly on it."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def segment_intersects(
    a1: Sequence[float], a2: Sequence[float], b1: Sequence[float], b2: Sequence[float]
) -> bool:
    """True when segment a1-a2 properly crosses segment b1-b2.

    Used to require that the vehicle's motion segment actually crosses the
    *finite* counting line, not the infinite line through it -- otherwise
    traffic in an adjacent lane well past the line's endpoints would count.
    """
    d1 = side_of_line(b1, a1, a2)
    d2 = side_of_line(b2, a1, a2)
    d3 = side_of_line(a1, b1, b2)
    d4 = side_of_line(a2, b1, b2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Collinear-touch cases: treat a point exactly on the other segment as
    # a crossing, so a vehicle whose centre lands precisely on the line is
    # not silently dropped.
    for d, p, s1, s2 in ((d1, b1, a1, a2), (d2, b2, a1, a2), (d3, a1, b1, b2), (d4, a2, b1, b2)):
        if d == 0 and _on_segment(p, s1, s2):
            return True
    return False


def _on_segment(p: Sequence[float], s1: Sequence[float], s2: Sequence[float]) -> bool:
    return (
        min(s1[0], s2[0]) <= p[0] <= max(s1[0], s2[0])
        and min(s1[1], s2[1]) <= p[1] <= max(s1[1], s2[1])
    )


def scale_line(
    line: Sequence[Sequence[float]],
    ref_w: int,
    ref_h: int,
    frame_w: int,
    frame_h: int,
) -> List[List[float]]:
    """Rescale a line calibrated at one resolution to the actual frame."""
    sx = frame_w / float(ref_w) if ref_w else 1.0
    sy = frame_h / float(ref_h) if ref_h else 1.0
    return [[p[0] * sx, p[1] * sy] for p in line]


def pairwise_iom(boxes: np.ndarray) -> np.ndarray:
    """Vectorised IoM matrix for an (N, 4) array of xyxy boxes."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    smaller = np.minimum(areas[:, None], areas[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(smaller > 0, inter / smaller, 0.0)
    return out.astype(np.float32)
