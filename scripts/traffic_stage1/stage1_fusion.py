"""Detection-level box fusion, applied before the tracker sees anything.

Motivation (this is the concrete bug this module exists to fix): on a
3-minute CVC-10 clip, one physical bus filling most of the frame width was
detected as THREE boxes tiled front-to-back along its body -- front
section, door section, rear section. Ultralytics' NMS, class-agnostic or
not, never merged them because their mutual IoU is low (they barely
overlap each other), so ByteTrack created three independent tracks and the
counter counted one bus three times.

Fusion here uses intersection-over-minimum (containment) instead of IoU,
which is exactly the signal that separates "panel of a bigger vehicle"
from "a smaller vehicle that happens to be nearby".
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from stage1_config import FusionConfig
from stage1_geometry import area, containment, iom, union_box

logger = logging.getLogger("mvsa.traffic_stage1")


class _DisjointSet:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def should_fuse(box_a, box_b, config: FusionConfig) -> bool:
    """Two boxes are the same physical vehicle when the smaller is largely
    swallowed by the larger AND they are not wildly different in size.

    The size guard matters: a motorcycle stopped directly in front of a bus
    is heavily contained by the bus box, but its area ratio is enormous, so
    fusing them would delete a real vehicle. Panels of one mis-split
    vehicle are, by contrast, comparable in size to each other.
    """
    a_area, b_area = area(box_a), area(box_b)
    if a_area <= 0 or b_area <= 0:
        return False

    ratio = max(a_area, b_area) / min(a_area, b_area)
    if ratio > config.max_area_ratio:
        return False

    if iom(box_a, box_b) < config.iom_thresh:
        return False

    inner, outer = (box_a, box_b) if a_area <= b_area else (box_b, box_a)
    return containment(inner, outer) >= config.min_containment


def fuse_boxes(boxes: np.ndarray, confs: np.ndarray, config: FusionConfig):
    """Merge co-detected boxes belonging to one physical vehicle.

    Returns ``(fused_boxes, fused_confs, groups)`` where ``groups[i]`` lists
    the indices of the input boxes that produced output box ``i``. A fused
    box is the union of its members (the full vehicle extent, which is what
    Stage 2 wants to crop) and carries the max confidence of its members.
    """
    n = len(boxes)
    if not config.enabled or n <= 1:
        return boxes, confs, [[i] for i in range(n)]

    ds = _DisjointSet(n)
    # Fusion is transitive by construction: front-panel fuses with
    # door-panel, door-panel with rear-panel, so all three land in one
    # group even though front and rear may not overlap each other at all.
    for i in range(n):
        for j in range(i + 1, n):
            if should_fuse(boxes[i], boxes[j], config):
                ds.union(i, j)

    grouped = {}
    for i in range(n):
        grouped.setdefault(ds.find(i), []).append(i)

    out_boxes, out_confs, groups = [], [], []
    for members in grouped.values():
        members.sort()
        out_boxes.append(union_box([boxes[m] for m in members]))
        out_confs.append(float(max(confs[m] for m in members)))
        groups.append(members)

    fused = np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4)
    if len(fused) < n:
        logger.debug("Fused %d detections into %d", n, len(fused))
    return fused, np.asarray(out_confs, dtype=np.float32), groups
