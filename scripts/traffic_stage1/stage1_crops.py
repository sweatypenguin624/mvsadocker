"""Per-vehicle crop selection -- Stage 1's real product.

Stage 2 classifies vehicles from these crops, so what matters is that each
physical vehicle contributes exactly one folder holding its few clearest
views. Crops are buffered per RAW track id during the pass (the vehicle
identity is not known until the post-pass) and re-keyed to vehicle ids at
write time, so a vehicle merged from three raw tracks yields ONE folder,
not three.

Frames are scored on box area, detector confidence and sharpness, and only
the top N per vehicle survive -- saving every frame of every track would
produce hundreds of near-duplicate images per vehicle and make Stage 2
both slow and no more accurate.
"""

from __future__ import annotations

import heapq
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from stage1_config import CropsConfig

logger = logging.getLogger("mvsa.traffic_stage1")

# Variance-of-Laplacian above this counts as "fully sharp" for scoring.
BLUR_SHARPNESS_CAP = 400.0


@dataclass(order=True)
class CropCandidate:
    score: float
    seq: int
    frame_idx: int = field(compare=False)
    jpg: bytes = field(compare=False, repr=False)
    bbox: List[float] = field(compare=False, default_factory=list)
    conf: float = field(compare=False, default=0.0)


def quality_score(bbox: Sequence[float], conf: float, frame: np.ndarray) -> float:
    x1, y1, x2, y2 = _clamp(bbox, frame.shape[1], frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    h, w = frame.shape[:2]
    area_norm = min(1.0, ((x2 - x1) * (y2 - y1)) / float(w * h))
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    blur_norm = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / BLUR_SHARPNESS_CAP)
    return 0.4 * area_norm + 0.3 * float(conf) + 0.3 * blur_norm


def _clamp(bbox: Sequence[float], w: int, h: int):
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    return max(0, min(x1, w)), max(0, min(y1, h)), max(0, min(x2, w)), max(0, min(y2, h))


class CropCollector:
    def __init__(self, config: CropsConfig):
        self.config = config
        self._buffers: Dict[int, List[CropCandidate]] = {}
        self._seq = itertools.count()

    def consider(self, track_id: int, bbox: Sequence[float], conf: float, frame_idx: int, frame: np.ndarray) -> None:
        if not self.config.enabled:
            return

        h, w = frame.shape[:2]
        pad_x = (bbox[2] - bbox[0]) * self.config.pad_frac
        pad_y = (bbox[3] - bbox[1]) * self.config.pad_frac
        padded = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]
        x1, y1, x2, y2 = _clamp(padded, w, h)
        if (x2 - x1) < self.config.min_box_px or (y2 - y1) < self.config.min_box_px:
            return

        # Score on the unpadded box so padding cannot inflate the area term.
        score = quality_score(bbox, conf, frame)
        heap = self._buffers.setdefault(track_id, [])
        # Encode only when the frame will actually be kept -- JPEG encoding
        # every candidate frame of every track dominates runtime otherwise.
        if len(heap) >= self.config.max_per_vehicle and score <= heap[0].score:
            return

        ok, encoded = cv2.imencode(
            ".jpg", frame[y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            return

        candidate = CropCandidate(
            score=score,
            seq=next(self._seq),
            frame_idx=frame_idx,
            jpg=encoded.tobytes(),
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            conf=float(conf),
        )
        if len(heap) < self.config.max_per_vehicle:
            heapq.heappush(heap, candidate)
        else:
            heapq.heapreplace(heap, candidate)

    def write(self, out_dir: Path, vehicle_to_tracks: Dict[int, List[int]], counted_ids: Optional[set] = None) -> Dict[int, Dict]:
        """Write one folder per vehicle, pooling all its raw tracks' crops."""
        if not self.config.enabled:
            return {}

        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[int, Dict] = {}

        for vehicle_id, track_ids in sorted(vehicle_to_tracks.items()):
            if counted_ids is not None and vehicle_id not in counted_ids:
                continue
            pooled: List[CropCandidate] = []
            for tid in track_ids:
                pooled.extend(self._buffers.get(tid, []))
            if not pooled:
                continue
            # Re-rank the pooled set and re-trim: a vehicle merged from 3
            # raw tracks must still end up with max_per_vehicle crops.
            pooled.sort(key=lambda c: c.score, reverse=True)
            pooled = pooled[: self.config.max_per_vehicle]

            vdir = out_dir / f"vehicle_{vehicle_id:06d}"
            vdir.mkdir(parents=True, exist_ok=True)
            files = []
            for i, cand in enumerate(pooled, start=1):
                name = "best.jpg" if i == 1 else f"crop_{i:02d}.jpg"
                (vdir / name).write_bytes(cand.jpg)
                files.append(
                    {
                        "file": name,
                        "frame_idx": cand.frame_idx,
                        "score": round(cand.score, 4),
                        "bbox": [round(v, 1) for v in cand.bbox],
                        "conf": round(cand.conf, 4),
                    }
                )
            manifest[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "raw_track_ids": sorted(track_ids),
                "crop_count": len(files),
                "crops": files,
            }
            (vdir / "metadata.json").write_text(json.dumps(manifest[vehicle_id], indent=2), encoding="utf-8")

        logger.info("Wrote crops for %d vehicles under %s", len(manifest), out_dir)
        return manifest
