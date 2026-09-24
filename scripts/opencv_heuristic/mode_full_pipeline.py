"""RUN_MODE == "full_pipeline": track buses, classify color per-track over
time, and aggregate into 15-minute interval counts.

Tracker: BoT-SORT. BoT-SORT can bridge short occlusions via its tracking
buffer; a long disappearance can result in a new track ID on reappearance.
No custom identity-merging is applied in this version.

save_checkpoint() below is an intermediate-results backup only -- NOT a true
BoT-SORT resume. Restarting from a checkpoint re-initializes the tracker and
will likely assign new track IDs to buses already in view; identity
continuity is not guaranteed.
"""

from __future__ import annotations

import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from classify import TrackColorAccumulator, classify_bus_crop
from config import PipelineConfig
from crops import crop_bus


def run_full_pipeline(detector, bus_class_id: int, video_path: Path, output_dir: Path,
                       cfg: PipelineConfig, device: str, detector_model_name: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_crops_dir = output_dir / "debug_crops"
    if cfg.SAVE_DEBUG_CROPS:
        for c in cfg.COLOR_CLASSES:
            os.makedirs(debug_crops_dir / c, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path} for full processing.")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration_seconds = (video_frame_count / video_fps) if video_frame_count > 0 else 0.0

    track_info = {}
    color_acc = TrackColorAccumulator(cfg)
    last_classification_time = {}
    interval_track_ids = defaultdict(set)
    debug_crop_counts = defaultdict(int)

    def get_interval(timestamp_seconds):
        return int(timestamp_seconds / (cfg.INTERVAL_MINUTES * 60))

    def save_debug_crop(crop, color):
        if not cfg.SAVE_DEBUG_CROPS or debug_crop_counts[color] >= cfg.MAX_DEBUG_CROPS_PER_CLASS:
            return
        idx = debug_crop_counts[color]
        out_path = debug_crops_dir / color / f"{color}_{idx:04d}.jpg"
        try:
            cv2.imwrite(str(out_path), crop)
            debug_crop_counts[color] += 1
        except Exception as e:
            if cfg.DEBUG:
                print(f"  [WARN] failed to save debug crop: {e}")

    def save_checkpoint(frame_number):
        ckpt = {
            "frame_number": frame_number,
            "track_info": track_info,
            "last_classification_time": last_classification_time,
            "track_color_prediction_count": dict(color_acc.prediction_count),
            "interval_track_ids": {k: list(v) for k, v in interval_track_ids.items()},
        }
        with open(output_dir / "checkpoint.pkl", "wb") as f:
            pickle.dump(ckpt, f)

    annotated_writer = None
    annotated_path = output_dir / "annotated.mp4"
    if cfg.SAVE_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(str(annotated_path), fourcc, video_fps, (video_width, video_height))
        if not annotated_writer.isOpened():
            print("[WARN] Could not open annotated video writer; disabling annotated video output.")
            annotated_writer = None

    frame_number = 0
    processing_start_time = time.time()

    print("Starting full video processing...")
    print(f"Tracker: BoT-SORT ({cfg.TRACKER_CONFIG})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_seconds = frame_number / video_fps
            current_interval = get_interval(timestamp_seconds)

            try:
                results = detector.track(
                    source=frame, persist=True, tracker=cfg.TRACKER_CONFIG,
                    conf=cfg.DETECTION_CONFIDENCE, classes=[bus_class_id],
                    device=device, verbose=False,
                )
            except Exception as e:
                print(f"[WARN] frame {frame_number}: detector.track() failed: {e}")
                frame_number += 1
                continue

            r = results[0]
            active_track_count = 0

            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                active_track_count = len(track_ids)

                for box, track_id in zip(boxes_xyxy, track_ids):
                    track_id = int(track_id)
                    if track_id not in track_info:
                        track_info[track_id] = {
                            "first_seen": timestamp_seconds, "last_seen": timestamp_seconds,
                            "first_frame": frame_number, "last_frame": frame_number,
                            "intervals": set(),
                        }
                    else:
                        track_info[track_id]["last_seen"] = timestamp_seconds
                        track_info[track_id]["last_frame"] = frame_number

                    track_info[track_id]["intervals"].add(current_interval)
                    interval_track_ids[current_interval].add(track_id)

                    should_classify = (
                        track_id not in last_classification_time
                        or (timestamp_seconds - last_classification_time[track_id]) >= cfg.CLASSIFY_INTERVAL_SECONDS
                    )
                    predicted_color, predicted_conf = None, None
                    if should_classify:
                        crop = crop_bus(frame, box.tolist(), cfg)
                        if crop is not None:
                            predicted_color, predicted_conf = classify_bus_crop(crop, cfg)
                            if predicted_color is not None:
                                color_acc.update(track_id, predicted_color, predicted_conf)
                                last_classification_time[track_id] = timestamp_seconds
                                if cfg.SAVE_DEBUG_CROPS:
                                    save_debug_crop(crop, predicted_color)

                    if annotated_writer is not None:
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_color = predicted_color if predicted_color else "?"
                        label_conf = predicted_conf if predicted_conf else 0.0
                        label = f"ID: {track_id} | {label_color.upper()} | {label_conf:.2f}"
                        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if annotated_writer is not None:
                annotated_writer.write(frame)

            frame_number += 1
            if frame_number % cfg.PROGRESS_EVERY_N_FRAMES == 0:
                elapsed = time.time() - processing_start_time
                proc_fps = frame_number / elapsed if elapsed > 0 else 0.0
                print(f"Processing frame {frame_number} / {video_frame_count} | "
                      f"t={timedelta(seconds=int(timestamp_seconds))} | "
                      f"active tracks={active_track_count} | proc FPS={proc_fps:.1f}")
            if frame_number % cfg.CHECKPOINT_EVERY_N_FRAMES == 0:
                save_checkpoint(frame_number)
    finally:
        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()

    total_processing_seconds = time.time() - processing_start_time
    overall_processing_fps = frame_number / total_processing_seconds if total_processing_seconds > 0 else 0.0
    save_checkpoint(frame_number)
    print(f"\nProcessing complete. {frame_number} frames in {total_processing_seconds:.1f}s ({overall_processing_fps:.1f} FPS).")

    final_track_colors = {tid: color_acc.final_color(tid) for tid in track_info}

    track_rows = []
    for track_id, info in track_info.items():
        color, color_conf = final_track_colors[track_id]
        duration = info["last_seen"] - info["first_seen"]
        for interval_idx in sorted(info["intervals"]):
            interval_start = interval_idx * cfg.INTERVAL_MINUTES * 60
            interval_end = interval_start + cfg.INTERVAL_MINUTES * 60
            track_rows.append({
                "track_id": track_id,
                "interval_start": str(timedelta(seconds=interval_start)),
                "interval_end": str(timedelta(seconds=interval_end)),
                "first_seen": str(timedelta(seconds=int(info["first_seen"]))),
                "last_seen": str(timedelta(seconds=int(info["last_seen"]))),
                "duration_seconds": round(duration, 2),
                "color": color,
                "color_confidence": round(color_conf, 4),
                "number_of_color_predictions": color_acc.prediction_count.get(track_id, 0),
            })

    track_results_df = pd.DataFrame(track_rows).sort_values(["track_id", "interval_start"]).reset_index(drop=True)
    track_results_path = output_dir / "track_results.csv"
    track_results_df.to_csv(track_results_path, index=False)
    print(f"Saved: {track_results_path}")

    if track_info:
        max_interval = max(get_interval(info["last_seen"]) for info in track_info.values())
    else:
        max_interval = get_interval(video_duration_seconds)
    all_intervals = list(range(0, max_interval + 1))

    interval_color_counts = {i: {c: 0 for c in cfg.COLOR_CLASSES} for i in all_intervals}
    for track_id, info in track_info.items():
        color, _ = final_track_colors[track_id]
        for interval_idx in info["intervals"]:
            interval_color_counts[interval_idx][color] += 1

    counts_rows = []
    for interval_idx in all_intervals:
        interval_start = interval_idx * cfg.INTERVAL_MINUTES * 60
        interval_end = interval_start + cfg.INTERVAL_MINUTES * 60
        counts = interval_color_counts[interval_idx]
        total = sum(counts.values())
        row = {
            "interval_start": str(timedelta(seconds=interval_start)),
            "interval_end": str(timedelta(seconds=interval_end)),
        }
        row.update({c: counts[c] for c in cfg.COLOR_CLASSES})
        row["total"] = total
        counts_rows.append(row)

    counts_df = pd.DataFrame(counts_rows)
    counts_path = output_dir / "15min_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"Saved: {counts_path}")

    total_unique_tracks = len(track_info)
    total_bus_appearances = sum(len(info["intervals"]) for info in track_info.values())
    totals_by_color = {c: int(counts_df[c].sum()) for c in cfg.COLOR_CLASSES}

    summary_lines = [
        "CPM Hubli-Dharwad Bus Color Classification Run", "=" * 47, "",
        f"Video: {os.path.basename(video_path)}",
        f"Video duration: {timedelta(seconds=int(video_duration_seconds))}",
        f"FPS: {video_fps:.3f}", f"Resolution: {video_width}x{video_height}",
        f"Total frames: {video_frame_count}", "",
        f"Detector model: {detector_model_name}",
        f"Detector bus class ID: {bus_class_id}", "",
        "Color classifier: OpenCV HSV heuristic (no trained model)",
        f"HSV color ranges: {cfg.HSV_COLOR_RANGES}",
        f"HSV min saturation/value: {cfg.HSV_MIN_SATURATION}/{cfg.HSV_MIN_VALUE}", "",
        "Tracker: BoT-SORT", f"Tracker config: {cfg.TRACKER_CONFIG}", "",
        f"Detection confidence: {cfg.DETECTION_CONFIDENCE}",
        f"Classification interval: {cfg.CLASSIFY_INTERVAL_SECONDS}s",
        f"15-minute interval: {cfg.INTERVAL_MINUTES} min", "",
        f"Device: {device}", "",
        f"Total unique track IDs: {total_unique_tracks}",
        f"Total bus appearances across intervals: {total_bus_appearances}", "",
    ] + [f"{c.capitalize()}: {totals_by_color[c]}" for c in cfg.COLOR_CLASSES] + [
        "",
        f"Processing time: {timedelta(seconds=int(total_processing_seconds))}",
        f"Processing FPS: {overall_processing_fps:.1f}", "",
        f"Output directory: {output_dir}",
    ]
    summary_path = output_dir / "run_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {summary_path}")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(counts_df))
    for color in cfg.COLOR_CLASSES:
        ax1.plot(x, counts_df[color], marker="o", label=color)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(counts_df["interval_start"], rotation=45, ha="right")
    ax1.set_xlabel("Interval start"); ax1.set_ylabel("Bus count")
    ax1.set_title("Bus counts per 15-minute interval by color"); ax1.legend()
    fig1.tight_layout()
    fig1.savefig(output_dir / "counts_by_color.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x, counts_df["total"], color="gray")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(counts_df["interval_start"], rotation=45, ha="right")
    ax2.set_xlabel("Interval start"); ax2.set_ylabel("Total buses")
    ax2.set_title("Total buses per 15-minute interval")
    fig2.tight_layout()
    fig2.savefig(output_dir / "counts_total.png")
    plt.close(fig2)

    print("=" * 60)
    print("CPM HUBLI-DHARWAD BUS COLOR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"Duration: {timedelta(seconds=int(video_duration_seconds))}")
    print(f"Processing time: {timedelta(seconds=int(total_processing_seconds))}")
    print(f"Processing FPS: {overall_processing_fps:.1f}")
    print(f"\n15-minute intervals: {len(all_intervals)}")
    print(f"Unique track IDs: {total_unique_tracks}")
    for c in cfg.COLOR_CLASSES:
        print(f"{c.capitalize()}: {totals_by_color[c]}")
    print(f"\nTotal bus appearances: {total_bus_appearances}")
    print("\nOutputs:\n15min_counts.csv\ntrack_results.csv\nrun_summary.txt\ncounts_by_color.png\ncounts_total.png")
    if cfg.SAVE_ANNOTATED_VIDEO and annotated_writer is not None:
        print("annotated.mp4")
    if cfg.SAVE_DEBUG_CROPS:
        print("debug_crops/")
    print(f"\nSaved to:\n{output_dir}")
    print("=" * 60)

    return {
        "track_results_path": str(track_results_path),
        "counts_path": str(counts_path),
        "summary_path": str(summary_path),
        "total_unique_tracks": total_unique_tracks,
        "totals_by_color": totals_by_color,
    }
