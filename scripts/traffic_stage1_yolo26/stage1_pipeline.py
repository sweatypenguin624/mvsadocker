"""Stage 1 orchestration: video in, unique-vehicle crops + 15-min counts out.

Two passes over the data, not one:

  PASS 1 (streaming)  detect -> fuse -> track, recording every observation
                      and buffering candidate crops per raw track id.
  PASS 2 (post)       raw tracks -> vehicle identities (spatial dedup +
                      temporal stitching) -> line-crossing count -> crops.

The split is what makes "count each vehicle exactly once" achievable: both
identity corrections need evidence from frames after the tracker's
mistake, so no online counter can make them.

Stage 1 emits NO vehicle class. Every box is generic "vehicle"; Stage 2
consumes crops/vehicle_*/ and assigns 2/3/4-wheeler.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
for _p in (str(SCRIPTS_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from video_utils import VideoStreamReader, frame_idx_to_timestamp  # noqa: E402

from stage1_config import Stage1Config  # noqa: E402
from stage1_count import build_intervals, count_vehicles  # noqa: E402
from stage1_crops import CropCollector  # noqa: E402
from stage1_detect import VehicleDetector  # noqa: E402
from stage1_fusion import fuse_boxes  # noqa: E402
from stage1_appearance import RunningAppearance, compute as compute_appearance  # noqa: E402
from stage1_identity import Observation, RawTrack, resolve_identities  # noqa: E402
from stage1_track import VehicleTracker  # noqa: E402

logger = logging.getLogger("mvsa.traffic_stage1")


def run_stage1(
    video_path: Path,
    output_dir: Path,
    config: Stage1Config,
    start_time: datetime,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    progress_every: int = 500,
) -> Dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    # Sub-sampling shortens the effective track buffer and the temporal
    # stitching gap in wall-clock terms, so both are expressed in PROCESSED
    # frames and the stride is folded in here rather than silently ignored.
    effective_fps = fps / frame_stride
    logger.info(
        "Video %s: %dx%d @ %.2f fps (%s frames), stride=%d -> %.2f effective fps",
        video_path.name, frame_w, frame_h, fps, total_frames or "?", frame_stride, effective_fps,
    )

    detector = VehicleDetector(config)
    tracker = VehicleTracker(config.tracker, frame_rate=int(round(effective_fps)))
    crops = CropCollector(config.crops)

    raw_tracks: Dict[int, RawTrack] = {}
    appearances: Dict[int, RunningAppearance] = {}
    processed = 0
    detections_total = 0
    fused_total = 0
    last_frame_idx = 0

    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []
    batch_size = max(1, config.detector.batch_size)

    def flush_batch() -> None:
        nonlocal detections_total, fused_total
        if not batch_frames:
            return
        for (boxes, confs), frame, frame_idx in zip(
            detector.detect_batch(batch_frames), batch_frames, batch_indices
        ):
            detections_total += len(boxes)
            boxes, confs, _ = fuse_boxes(boxes, confs, config.fusion)
            fused_total += len(boxes)

            # The tracker advances its own frame counter on every update,
            # so it must be called even on empty frames or lost-track
            # ageing drifts out of sync with the video.
            for track_id, bbox, conf in tracker.update(boxes, confs, frame):
                raw_tracks.setdefault(track_id, RawTrack(track_id=track_id)).observations.append(
                    Observation(frame_idx=frame_idx, bbox=list(bbox), conf=conf)
                )
                appearances.setdefault(track_id, RunningAppearance()).add(
                    compute_appearance(frame, bbox)
                )
                crops.consider(track_id, bbox, conf, frame_idx, frame)
        batch_frames.clear()
        batch_indices.clear()

    with VideoStreamReader(video_path) as reader:
        for frame_idx, frame in reader.frames():
            if frame_stride > 1 and frame_idx % frame_stride:
                continue
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            last_frame_idx = frame_idx
            processed += 1

            if len(batch_frames) >= batch_size:
                flush_batch()

            if progress_every and processed % progress_every == 0:
                elapsed = time.time() - started
                logger.info(
                    "  %d frames processed (%.1f fps), %d live tracks",
                    processed, processed / max(elapsed, 1e-6), len(raw_tracks),
                )
            if max_frames and processed >= max_frames:
                break
        flush_batch()

    for track_id, track in raw_tracks.items():
        track.appearance = appearances[track_id].mean
    fingerprinted = sum(1 for t in raw_tracks.values() if t.appearance is not None)
    logger.info(
        "Pass 1 done: %d frames, %d raw tracks (%d with appearance fingerprints)",
        processed, len(raw_tracks), fingerprinted,
    )

    # --- Pass 2: identity resolution, counting, output -------------------
    identities, identity_stats = resolve_identities(raw_tracks, config.identity)

    def frame_time(frame_idx: int) -> datetime:
        return frame_idx_to_timestamp(frame_idx, fps, start_time)

    counted, rejected, line, calibrated = count_vehicles(
        identities, config.counting, frame_w, frame_h, frame_time
    )

    video_end = frame_time(last_frame_idx)
    intervals = build_intervals(counted, config.counting, start_time, video_end)

    counted_ids = {c.vehicle_id for c in counted}
    vehicle_to_tracks = {i.vehicle_id: i.raw_track_ids for i in identities}
    # Only counted vehicles get crops: an identity that never crossed the
    # line is not part of any 15-min total, so Stage 2 must not classify it.
    crop_manifest = crops.write(output_dir / "crops", vehicle_to_tracks, counted_ids)

    vehicles_path = output_dir / "vehicles.jsonl"
    with vehicles_path.open("w", encoding="utf-8") as fh:
        for c in counted:
            fh.write(json.dumps({
                "vehicle_id": c.vehicle_id,
                "raw_track_ids": c.raw_track_ids,
                "crossing_frame": c.crossing_frame,
                "crossing_time": c.crossing_time.isoformat(),
                "interval": c.interval,
                "direction": c.direction,
                "frames_seen": c.frames_seen,
                "first_frame": c.first_frame,
                "last_frame": c.last_frame,
                "best_bbox": c.best_bbox,
                "mean_conf": c.mean_conf,
                "merge_reasons": c.merge_reasons,
                "crop_dir": f"crops/vehicle_{c.vehicle_id:06d}",
                "crop_count": crop_manifest.get(c.vehicle_id, {}).get("crop_count", 0),
            }) + "\n")

    elapsed = time.time() - started
    summary = {
        "stage": 1,
        "note": "Stage 1 is class-agnostic. Vehicle type (2/3/4-wheeler) is assigned by Stage 2 from crops/.",
        "output_dir": str(output_dir),
        "start_time": start_time.isoformat(),
        "end_time": video_end.isoformat(),
        "video": {
            "path": str(video_path), "fps": round(fps, 3), "width": frame_w, "height": frame_h,
            "total_frames": total_frames, "frames_processed": processed, "frame_stride": frame_stride,
        },
        "counting_line": {
            "points": [[round(v, 1) for v in p] for p in line],
            "calibrated": calibrated,
            "warning": None if calibrated else "Uncalibrated fallback line -- run calibrate_line.py for this camera.",
        },
        "detection": {
            "raw_detections": detections_total,
            "after_fusion": fused_total,
            "fused_away": detections_total - fused_total,
            "dropped_by_size_filter": detector.dropped_by_size,
        },
        "tracking": identity_stats,
        "counting": {
            "counted_vehicles": len(counted),
            "rejected": rejected,
            "vehicles_with_crops": len(crop_manifest),
        },
        "intervals": intervals,
        "runtime_seconds": round(elapsed, 1),
        "processing_fps": round(processed / max(elapsed, 1e-6), 2),
    }
    (output_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Stage 1 complete in %.1fs: %d unique vehicles counted across %d intervals -> %s",
        elapsed, len(counted), len(intervals), output_dir,
    )
    return summary
