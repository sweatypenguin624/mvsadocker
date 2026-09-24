"""Representative-crop saving for Goods-Vehicle-shaped tracks produced by
this pipeline's own ByteTrack engine (tracker.py::VehicleTracker), for
Layer 2 (scripts/traffic_layer2/) to consume AND for full-coverage manual
annotation (every frame of every Goods Vehicle track, not just a
top-scored handful -- see the "extract ALL the frames" run).

Deliberately independent of scripts/traffic_layer1/crops.py (different
TrackState shape here -- class_history/conf_history + get_stable_class(),
not a final_class attribute) rather than forcing a shared interface across
two pipelines that don't share a TrackState class.

Two modes, both writing the same output shape
(track_XXXXXX/{best.jpg, frame_NNN.jpg, metadata.json}) so
scripts/traffic_layer2/layer2_io.py reads either unchanged:

  - Bounded (max_per_track > 0): same as before -- an in-memory top-K
    heap by quality score (box size + confidence + blur), written once at
    the end of the run.
  - Unbounded (max_per_track <= 0, "save all"): every considered frame is
    JPEG-encoded and written to disk IMMEDIATELY (not buffered in memory
    for the whole run -- a 45-minute, every-frame run can produce tens of
    thousands of crops, and buffering all of them as JPEG bytes until the
    end risks a large memory footprint and losing everything on a crash
    near the finish). Only lightweight per-frame metadata (score, bbox,
    conf, frame_idx) is kept in memory; write_all() then either finalizes
    a track's metadata.json (if its final class held) or deletes its
    speculative frames (if the track's class later flipped away from
    Goods-Vehicle-shaped).
"""

from __future__ import annotations

import heapq
import itertools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# UVH-26 native classes that map to the "Goods Vehicle" bucket -- kept in
# sync with config/layer1_class_mapping.yaml's uvh26 section by hand (this
# pipeline has no shared taxonomy module with Layer 1 to import from without
# creating a cross-pipeline dependency).
GOODS_VEHICLE_NATIVE_CLASSES = {"Truck", "LCV", "tempo-traveller"}

BLUR_SHARPNESS_CAP = 400.0


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


@dataclass(order=True)
class _CropCandidate:
    score: float
    seq: int = field(compare=True)
    frame_idx: int = field(compare=False)
    jpg_bytes: bytes = field(compare=False)
    bbox: List[float] = field(compare=False, default_factory=list)
    conf: float = field(compare=False, default=0.0)


@dataclass
class _SavedFrameRef:
    """Metadata for a frame already written to disk in "save all" mode --
    the image itself is not held in memory.
    """
    file: str
    frame_idx: int
    score: float
    bbox: List[float]
    conf: float


class GoodsCropManager:
    def __init__(self, max_per_track: int = 5, target_classes=None, all_classes: bool = False):
        self.max_per_track = max_per_track
        self.save_all = max_per_track is None or max_per_track <= 0
        # None (all_classes=True) = every class the detector emits, not
        # just the Goods-Vehicle-shaped ones -- see the "bigger run,
        # include all vehicles" request. Keeping the class name/attribute
        # as-is (not renamed) since it's still keyed off the same
        # GOODS_VEHICLE_NATIVE_CLASSES default for the common case.
        self.target_classes = None if all_classes else set(target_classes or GOODS_VEHICLE_NATIVE_CLASSES)

        # Bounded mode.
        self._buffers: Dict[int, List[_CropCandidate]] = {}
        self._seq = itertools.count()

        # Unbounded ("save all") mode.
        self._saved_refs: Dict[int, List[_SavedFrameRef]] = {}
        self._output_dir: Optional[Path] = None
        self._frame_seq: Dict[int, int] = {}

    def bind_output_dir(self, output_dir: Path) -> None:
        """Must be called before consider() in save-all mode, since crops
        are written to disk as they're seen rather than buffered.
        """
        self._output_dir = Path(output_dir) / "goods_tracks"

    def consider(self, track_id: int, cls_name: str, bbox, conf: float, frame_idx: int, frame: np.ndarray) -> None:
        if self.target_classes is not None and cls_name not in self.target_classes:
            return
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return

        score = _quality_score(bbox, conf, frame)
        ok, encoded = cv2.imencode(".jpg", frame[y1:y2, x1:x2])
        if not ok:
            return
        jpg_bytes = encoded.tobytes()
        bbox_out = [float(x1), float(y1), float(x2), float(y2)]

        if self.save_all:
            if self._output_dir is None:
                raise RuntimeError("GoodsCropManager.bind_output_dir() must be called before consider() in save-all mode")
            seq = self._frame_seq.get(track_id, 0) + 1
            self._frame_seq[track_id] = seq
            track_dir = self._output_dir / f"track_{track_id:06d}"
            track_dir.mkdir(parents=True, exist_ok=True)
            name = f"frame_{seq:05d}.jpg"
            (track_dir / name).write_bytes(jpg_bytes)
            self._saved_refs.setdefault(track_id, []).append(
                _SavedFrameRef(file=name, frame_idx=frame_idx, score=score, bbox=bbox_out, conf=float(conf))
            )
            return

        candidate = _CropCandidate(
            score=score, seq=next(self._seq), frame_idx=frame_idx,
            jpg_bytes=jpg_bytes, bbox=bbox_out, conf=float(conf),
        )
        heap = self._buffers.setdefault(track_id, [])
        if len(heap) < self.max_per_track:
            heapq.heappush(heap, candidate)
        elif candidate.score > heap[0].score:
            heapq.heapreplace(heap, candidate)

    def write_all(self, output_dir: Path, tracker, class_names: Dict[int, str]) -> Dict[int, int]:
        """tracker is a VehicleTracker instance (tracker.py); a track's
        crops are only kept if its FINAL stable class (confidence-weighted
        vote across its whole life, not the per-frame class that triggered
        buffering) is still in target_classes -- in save-all mode, tracks
        that flip away from a Goods-Vehicle-shaped class have their
        already-written speculative frames deleted here.
        """
        if self.save_all:
            return self._write_all_save_all_mode(tracker, class_names)
        return self._write_all_bounded_mode(output_dir, tracker, class_names)

    def _write_all_save_all_mode(self, tracker, class_names: Dict[int, str]) -> Dict[int, int]:
        written: Dict[int, int] = {}
        if not self._saved_refs or self._output_dir is None:
            return written

        for track_id, refs in self._saved_refs.items():
            track_dir = self._output_dir / f"track_{track_id:06d}"
            state = tracker.tracks.get(track_id)
            final_class = state.get_stable_class(class_names) if state is not None else None

            if self.target_classes is not None and final_class not in self.target_classes:
                shutil.rmtree(track_dir, ignore_errors=True)
                continue

            best = max(refs, key=lambda r: r.score)
            shutil.copy2(track_dir / best.file, track_dir / "best.jpg")

            frame_refs = [
                {"file": r.file, "frame_idx": r.frame_idx, "score": round(r.score, 4),
                 "bbox": [round(v, 1) for v in r.bbox], "conf": round(r.conf, 4)}
                for r in sorted(refs, key=lambda r: r.frame_idx)
            ]
            metadata = {
                "track_id": int(track_id),
                "final_class": final_class,
                "best_frame_idx": best.frame_idx,
                "best_score": round(best.score, 4),
                "crops": frame_refs,
            }
            (track_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            written[track_id] = len(refs)

        return written

    def _write_all_bounded_mode(self, output_dir: Path, tracker, class_names: Dict[int, str]) -> Dict[int, int]:
        written: Dict[int, int] = {}
        if not self._buffers:
            return written

        crops_root = Path(output_dir) / "goods_tracks"
        for track_id, heap in self._buffers.items():
            state = tracker.tracks.get(track_id)
            if state is None:
                continue
            final_class = state.get_stable_class(class_names)
            if self.target_classes is not None and final_class not in self.target_classes:
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
                    "file": name, "frame_idx": cand.frame_idx, "score": round(cand.score, 4),
                    "bbox": [round(v, 1) for v in cand.bbox], "conf": round(cand.conf, 4),
                })

            metadata = {
                "track_id": int(track_id),
                "final_class": final_class,
                "best_frame_idx": best.frame_idx,
                "best_score": round(best.score, 4),
                "crops": frame_refs,
            }
            (track_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            written[track_id] = len(candidates)

        return written
