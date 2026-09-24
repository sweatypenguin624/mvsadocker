"""RUN_MODE == "collect_crops": sample bus crops per track for later labeling."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from config import PipelineConfig
from crops import crop_bus


def run_collect_crops(detector, bus_class_id: int, video_path: Path, output_dir: Path,
                       cfg: PipelineConfig, device: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving crops to: {output_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    last_saved_time = {}
    saved_count_per_track = {}
    total_saved = 0
    frame_number = 0
    start_time = time.time()

    try:
        while total_saved < cfg.MAX_TOTAL_CROPS:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_seconds = frame_number / video_fps
            try:
                results = detector.track(
                    source=frame, persist=True, tracker=cfg.TRACKER_CONFIG,
                    conf=cfg.DETECTION_CONFIDENCE, classes=[bus_class_id],
                    device=device, verbose=False,
                )
            except Exception as e:
                print(f"[WARN] frame {frame_number}: track() failed: {e}")
                frame_number += 1
                continue

            r = results[0]
            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes_xyxy, track_ids):
                    track_id = int(track_id)
                    if saved_count_per_track.get(track_id, 0) >= cfg.MAX_CROPS_PER_TRACK:
                        continue
                    if track_id in last_saved_time and (timestamp_seconds - last_saved_time[track_id]) < cfg.CROP_EVERY_SECONDS:
                        continue
                    crop = crop_bus(frame, box.tolist(), cfg)
                    if crop is None:
                        continue
                    out_name = f"track{track_id:04d}_t{timestamp_seconds:07.1f}.jpg"
                    cv2.imwrite(os.path.join(output_dir, out_name), crop)
                    last_saved_time[track_id] = timestamp_seconds
                    saved_count_per_track[track_id] = saved_count_per_track.get(track_id, 0) + 1
                    total_saved += 1
                    if total_saved >= cfg.MAX_TOTAL_CROPS:
                        break

            frame_number += 1
            if frame_number % cfg.PROGRESS_EVERY_N_FRAMES == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_number}/{video_frame_count} | t={timestamp_seconds:.1f}s | "
                      f"tracks seen={len(saved_count_per_track)} | crops saved={total_saved} | proc FPS={fps:.1f}")
    finally:
        cap.release()

    print(f"\nDone. Saved {total_saved} crops from {len(saved_count_per_track)} distinct tracks.")
    print(f"Crops directory: {output_dir}")
    print("\nNext: sort these images into 4 folders --")
    for c in cfg.COLOR_CLASSES:
        print(f"  <COLOR_DATASET_DIR>/train/{c}/")
    print("Aim for roughly 50-150+ images per color, then run in mode='train_classifier'.")
