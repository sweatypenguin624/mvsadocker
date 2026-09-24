"""Representative-crop selection and saving for future Layer 2 consumption.

Only tracks observed with a bucket in ``crops.target_classes`` (Bus, by
default -- see config/layer1_config.yaml) are buffered at all, and each
track keeps only its top ``crops.max_per_track`` frames by a quality score
(box size + detector confidence + sharpness), never every frame it appeared
in -- the spec is explicit that saving "hundreds of duplicate frames per
track" is not acceptable (section 17/20).

Crops are JPEG-encoded immediately on capture (not kept as raw arrays) so
memory use stays proportional to (active target-class tracks) x
max_per_track x one small JPEG, not to video length.
"""

from __future__ import annotations

import heapq
import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import CropsConfig  # noqa: E402
from layer1_models import SourcedDetection, TrackState  # noqa: E402

logger = logging.getLogger("mvsa.traffic_layer1")

# Blur (variance-of-Laplacian) values above this are treated as "fully
# sharp" for scoring purposes -- an empirically reasonable ceiling for
# 720p-1080p traffic-camera crops, not a hard technical bound.
BLUR_SHARPNESS_CAP = 400.0


@dataclass(order=True)
class _CropCandidate:
    score: float
    seq: int = field(compare=True)
    frame_idx: int = field(compare=False)
    jpg_bytes: bytes = field(compare=False)
    # Clamped-to-frame bbox this crop was cut from, [x1, y1, x2, y2] in
    # source-frame pixel space -- Layer 2 (axle/viewpoint assessment,
    # tractor-trolley spatial relationship) needs real geometry, not just
    # the cropped JPEG, per the Layer 2 spec's track-level input requirement.
    bbox: List[float] = field(compare=False, default_factory=list)
    conf: float = field(compare=False, default=0.0)


def _quality_score(bbox, conf: float, frame: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    area_norm = min(1.0, ((x2 - x1) * (y2 - y1)) / float(w * h))

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_norm = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / BLUR_SHARPNESS_CAP)

    return 0.4 * area_norm + 0.3 * float(conf) + 0.3 * blur_norm


class CropManager:
    """Buffers bounded, top-scored crops per track, keyed by Layer 1's
    synthetic global track_id.
    """

    def __init__(self, config: CropsConfig):
        self.config = config
        self._buffers: Dict[int, List[_CropCandidate]] = {}
        self._seq = itertools.count()

    def consider(
        self, track_id: int, det: SourcedDetection, frame_idx: int, frame: np.ndarray
    ) -> None:
        if not self.config.enabled or det.bucket not in self.config.target_classes:
            return

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return

        score = _quality_score(det.bbox, det.conf, frame)
        ok, encoded = cv2.imencode(".jpg", frame[y1:y2, x1:x2])
        if not ok:
            return
        candidate = _CropCandidate(
            score=score,
            seq=next(self._seq),
            frame_idx=frame_idx,
            jpg_bytes=encoded.tobytes(),
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            conf=float(det.conf),
        )

        heap = self._buffers.setdefault(track_id, [])
        if len(heap) < self.config.max_per_track:
            heapq.heappush(heap, candidate)
        elif candidate.score > heap[0].score:
            heapq.heapreplace(heap, candidate)

    def write_all(self, run_dir: Path, tracks: Dict[object, TrackState]) -> Dict[int, int]:
        """Write crops for every buffered track whose FINAL voted class is
        still one of target_classes (a track can accumulate Bus-bucket
        candidate frames yet ultimately vote e.g. Car overall -- only write
        crops for tracks the vote actually confirmed).
        """
        if not self.config.enabled or not self._buffers:
            return {}

        final_class_by_id = {s.track_id: s.final_class for s in tracks.values()}
        written: Dict[int, int] = {}
        crops_root = run_dir / "bus_tracks"

        for track_id, heap in self._buffers.items():
            if final_class_by_id.get(track_id) not in self.config.target_classes:
                continue
            candidates = sorted(heap, key=lambda c: c.score, reverse=True)
            if not candidates:
                continue

            track_dir = crops_root / f"track_{track_id:06d}"
            track_dir.mkdir(parents=True, exist_ok=True)

            best = candidates[0]
            (track_dir / "best.jpg").write_bytes(best.jpg_bytes)
            frame_refs = []
            for i, cand in enumerate(candidates, start=1):
                name = f"frame_{i:03d}.jpg"
                (track_dir / name).write_bytes(cand.jpg_bytes)
                frame_refs.append({
                    "file": name,
                    "frame_idx": cand.frame_idx,
                    "score": round(cand.score, 4),
                    "bbox": [round(v, 1) for v in cand.bbox],
                    "conf": round(cand.conf, 4),
                })

            metadata = {
                "track_id": track_id,
                "final_class": final_class_by_id.get(track_id),
                "best_frame_idx": best.frame_idx,
                "best_score": round(best.score, 4),
                "crops": frame_refs,
            }
            (track_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            written[track_id] = len(candidates)

        logger.info("Wrote crops for %d track(s) under %s", len(written), crops_root)
        return written
