"""Production mode: one video in, per-track vehicle-group counts (and, for
buses, a color classification second layer) out, aggregated into camera +
15-minute wall-clock windows.

Multi-class: with a UVH-26 fine-tuned detector (see uvh26_train.py), the
model itself distinguishes two-wheelers, auto-rickshaws (three-wheelers),
four-wheelers (hatchback/sedan/SUV/MUV/van), buses (bus/mini-bus/tempo-
traveller), and trucks (truck/LCV) -- see config.py: VEHICLE_GROUPS. This
mode tracks every group in one pass; earlier revisions tracked only 'bus'
plus a cross-class suppression workaround for COCO detectors that had no
finer-grained classes. That workaround is gone -- once the detector natively
tells vehicle types apart, trusting its own class label per group is more
reliable than trying to suppress one class using another's competing score.

Other differences from full_pipeline (mode_full_pipeline.py):
  - Classifies each bus track exactly ONCE, from its single best frame (see
    best_frame.py) -- not repeatedly across its lifetime.
  - Buckets by real wall-clock 15-minute windows ("09:00-09:15"), derived
    from the video's inferred real-world start time (video_discovery.py),
    not seconds-since-video-start.
  - Drops detections below a per-group minimum box-area fraction before they
    ever get a track ID (config.py: GROUP_MIN_AREA_FRAC).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from best_frame import BestFrameStore, score_detection
from classify import classify_bus_crop
from config import PipelineConfig
from crops import crop_bus


def _window_bounds(dt: datetime, interval_minutes: int):
    floor_minute = (dt.minute // interval_minutes) * interval_minutes
    start = dt.replace(minute=floor_minute, second=0, microsecond=0)
    end = start + timedelta(minutes=interval_minutes)
    return start, end


def build_class_id_to_group(detector, vehicle_groups: dict) -> dict:
    """Resolve config.py's VEHICLE_GROUPS (class *names*) to a class-id ->
    group-name map using this detector's own names. Class names not found in
    the detector are skipped with a warning (e.g. a COCO model won't have
    'Three-wheeler' at all)."""
    by_name = {str(v).strip().lower(): k for k, v in detector.names.items()}
    class_id_to_group = {}
    for group, class_names in vehicle_groups.items():
        for name in class_names:
            cid = by_name.get(name.strip().lower())
            if cid is None:
                print(f"[WARN] class '{name}' (group '{group}') not found in detector.names; skipping it.")
                continue
            class_id_to_group[cid] = group
    return class_id_to_group


def process_video(detector, class_id_to_group: dict, video_path: Path, camera_name: str,
                   start_dt: datetime, output_dir: Path, cfg: PipelineConfig, device: str,
                   start_offset_seconds: float = 0.0, duration_seconds: float | None = None) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not video_fps or video_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS reading {video_path}")

    frame_number = 0
    if start_offset_seconds > 0:
        frame_number = int(start_offset_seconds * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    max_timestamp_seconds = (start_offset_seconds + duration_seconds) if duration_seconds is not None else None

    track_classes = sorted(class_id_to_group.keys())
    groups_present = sorted(set(class_id_to_group.values()))

    best_frames = BestFrameStore()
    track_first_seen: dict = {}
    track_last_seen: dict = {}
    track_windows = defaultdict(set)       # track_id -> set of window_start datetimes
    track_group_votes = defaultdict(Counter)  # track_id -> Counter({group: n_frames})
    rejected_tiny = 0

    print(f"\n[{camera_name}] Processing {video_path.name} "
          f"(start={start_dt.isoformat()}, fps={video_fps:.2f}, frames={video_frame_count}"
          + (f", window={start_offset_seconds:.0f}s-{max_timestamp_seconds:.0f}s" if max_timestamp_seconds else "")
          + f", groups={groups_present})")

    t0 = time.time()
    try:
        while True:
            if max_timestamp_seconds is not None and (frame_number / video_fps) >= max_timestamp_seconds:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_seconds = frame_number / video_fps
            wallclock = start_dt + timedelta(seconds=timestamp_seconds)

            try:
                track_kwargs = dict(
                    source=frame, persist=True, tracker=cfg.TRACKER_CONFIG,
                    conf=cfg.DETECTION_CONFIDENCE, classes=track_classes,
                    device=device, verbose=False,
                )
                if cfg.IMGSZ:
                    track_kwargs["imgsz"] = cfg.IMGSZ
                results = detector.track(**track_kwargs)
            except Exception as e:
                print(f"  [WARN] frame {frame_number}: detector.track() failed: {e}")
                frame_number += 1
                continue

            r = results[0]
            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [None] * len(track_ids)
                frame_area = frame.shape[0] * frame.shape[1]

                for box, track_id, cls, conf in zip(boxes_xyxy, track_ids, classes, confs):
                    group = class_id_to_group.get(int(cls))
                    if group is None:
                        continue  # ungrouped class (Bicycle/Others) -- tracked by BoT-SORT, not counted
                    track_id = int(track_id)
                    box_list = box.tolist()
                    x1, y1, x2, y2 = box_list
                    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    min_area_frac = cfg.GROUP_MIN_AREA_FRAC.get(group, cfg.GROUP_MIN_AREA_FRAC["default"])
                    if frame_area > 0 and (box_area / frame_area) < min_area_frac:
                        rejected_tiny += 1
                        continue

                    win_start, _ = _window_bounds(wallclock, cfg.INTERVAL_MINUTES)
                    track_windows[track_id].add(win_start)
                    track_group_votes[track_id][group] += 1
                    track_first_seen.setdefault(track_id, wallclock)
                    track_last_seen[track_id] = wallclock

                    crop = crop_bus(frame, box_list, cfg)
                    if crop is not None:
                        score = score_detection(frame.shape, box_list, conf, crop, cfg)
                        best_frames.offer(track_id, score, crop, {
                            "frame_number": frame_number,
                            "timestamp_seconds": timestamp_seconds,
                            "wallclock": wallclock,
                            "confidence": float(conf) if conf is not None else None,
                            "group": group,
                        })

            frame_number += 1
            if frame_number % cfg.PROGRESS_EVERY_N_FRAMES == 0:
                elapsed = time.time() - t0
                fps_proc = frame_number / elapsed if elapsed > 0 else 0.0
                print(f"  frame {frame_number}/{video_frame_count} | wallclock={wallclock.strftime('%H:%M:%S')} | "
                      f"tracks so far={len(track_windows)} | proc FPS={fps_proc:.1f}")
    finally:
        cap.release()

    elapsed = time.time() - t0
    proc_fps = frame_number / elapsed if elapsed > 0 else 0.0
    print(f"  Done: {frame_number} frames in {elapsed:.1f}s ({proc_fps:.1f} FPS). "
          f"{len(track_windows)} tracks, {rejected_tiny} tiny-box detections rejected.")

    best_dir = output_dir / "best_frames"
    if cfg.SAVE_BEST_FRAME_CROPS:
        for g in groups_present:
            if g == cfg.COLOR_CLASSIFIED_GROUP:
                for c in cfg.COLOR_CLASSES:
                    (best_dir / f"{g}_{c}").mkdir(parents=True, exist_ok=True)
            else:
                (best_dir / g).mkdir(parents=True, exist_ok=True)

    track_rows = []
    # window -> group -> count, plus a nested bus-color breakdown
    window_group_counts = defaultdict(lambda: {g: 0 for g in groups_present})
    window_bus_color_counts = defaultdict(lambda: {c: 0 for c in cfg.COLOR_CLASSES})

    for track_id, windows in track_windows.items():
        group = track_group_votes[track_id].most_common(1)[0][0]
        best = best_frames.get(track_id)
        color, color_conf = "", ""

        if group == cfg.COLOR_CLASSIFIED_GROUP:
            if best is None:
                color, color_conf = cfg.FALLBACK_COLOR, None
            else:
                score, crop, best_meta = best
                color, color_conf = classify_bus_crop(crop, cfg)
                if color is None:
                    color, color_conf = cfg.FALLBACK_COLOR, None
                if cfg.SAVE_BEST_FRAME_CROPS:
                    out_name = f"{camera_name}_{video_path.stem}_track{track_id:05d}_{color}.jpg"
                    cv2.imwrite(str(best_dir / f"{group}_{color}" / out_name), crop)
        elif best is not None and cfg.SAVE_BEST_FRAME_CROPS:
            score, crop, best_meta = best
            out_name = f"{camera_name}_{video_path.stem}_track{track_id:05d}.jpg"
            cv2.imwrite(str(best_dir / group / out_name), crop)

        for win_start in windows:
            window_group_counts[win_start][group] += 1
            if group == cfg.COLOR_CLASSIFIED_GROUP:
                window_bus_color_counts[win_start][color] += 1

        best_meta = best[2] if best is not None else {}
        best_wallclock = best_meta.get("wallclock")
        best_conf = best_meta.get("confidence")
        track_rows.append({
            "camera": camera_name,
            "video": video_path.name,
            "track_id": track_id,
            "vehicle_group": group,
            "first_seen": track_first_seen[track_id].isoformat(),
            "last_seen": track_last_seen[track_id].isoformat(),
            "best_frame_wallclock": best_wallclock.isoformat() if best_wallclock else "",
            "best_frame_detection_confidence": round(best_conf, 4) if best_conf is not None else "",
            "color": color,
            "color_confidence": round(color_conf, 4) if isinstance(color_conf, float) else "",
        })

    window_rows = []
    all_windows = sorted(set(window_group_counts.keys()) | set(window_bus_color_counts.keys()))
    for win_start in all_windows:
        win_end = win_start + timedelta(minutes=cfg.INTERVAL_MINUTES)
        row = {
            "camera": camera_name,
            "video": video_path.name,
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
        }
        group_counts = window_group_counts.get(win_start, {g: 0 for g in groups_present})
        row.update(group_counts)
        row["total_vehicles"] = sum(group_counts.values())
        bus_colors = window_bus_color_counts.get(win_start, {c: 0 for c in cfg.COLOR_CLASSES})
        for c in cfg.COLOR_CLASSES:
            row[f"bus_{c}"] = bus_colors.get(c, 0)
        window_rows.append(row)

    return {
        "track_rows": track_rows,
        "window_rows": window_rows,
        "groups_present": groups_present,
        "frame_count": frame_number,
        "processing_seconds": elapsed,
        "rejected_tiny_boxes": rejected_tiny,
        "unique_tracks": len(track_windows),
    }
