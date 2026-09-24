#!/usr/bin/env python3

from ultralytics import YOLO
from shapely.geometry import Point, Polygon

from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import csv
import os
import re
import time


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = (
    "/home/users/oauser/mvsa/videos/actual_test/"
    "08.00.00-09.00.00.mp4"
)

MODEL_PATH = (
    "/home/users/oauser/mvsa/models/yolo11x.pt"
)

OUTPUT_DIR = (
    "/home/users/oauser/mvsa/results/"
    "actual_08-09"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "tracked.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "hourly_pedestrian_counts.csv"
)

CHECKPOINT_CSV = os.path.join(
    OUTPUT_DIR,
    "checkpoint.csv"
)


# Detection settings
CONF_THRESHOLD = 0.35
IMG_SIZE = 1280

# Process every frame.
PROCESS_EVERY_N_FRAMES = 2

# Save checkpoint every N frames.
CHECKPOINT_EVERY_FRAMES = 1000


# ============================================================
# ROI
# ============================================================

# Replace these with the exact coordinates from:
#
# ~/mvsa/results/pedestrian_roi.txt
#
# These are the coordinates you provided earlier.

PEDESTRIAN_ROI = Polygon([
    (0, 340),
    (180, 275),
    (410, 225),
    (545, 260),
    (430, 400),
    (280, 560),
    (140, 740),
    (0, 880),
])


# ============================================================
# SURVEY DATE
# ============================================================

VIDEO_DATE = datetime(
    2026,
    7,
    28
)


# ============================================================
# TIMESTAMP
# ============================================================

def get_start_time_from_filename(video_path):

    filename = os.path.basename(video_path)

    match = re.search(
        r"(\d{2})\.(\d{2})\.(\d{2})-"
        r"(\d{2})\.(\d{2})\.(\d{2})",
        filename
    )

    if not match:
        raise ValueError(
            f"Could not extract time from filename: {filename}"
        )

    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))

    return datetime(
        VIDEO_DATE.year,
        VIDEO_DATE.month,
        VIDEO_DATE.day,
        hour,
        minute,
        second
    )


def get_frame_timestamp(
    frame_idx,
    fps,
    start_time
):

    return (
        start_time
        + timedelta(
            seconds=frame_idx / fps
        )
    )


def hour_bucket(timestamp):

    return timestamp.strftime(
        "%Y-%m-%d %H:00"
    )


# ============================================================
# GEOMETRY
# ============================================================

def bottom_center(box):

    x1, y1, x2, y2 = box

    return Point(
        (x1 + x2) / 2,
        y2
    )


# ============================================================
# DRAW ROI
# ============================================================

def draw_roi(
    frame,
    roi
):

    points = [
        [int(x), int(y)]
        for x, y in roi.exterior.coords
    ]

    points = points[:-1]

    pts = cv2.UMat(
        __import__("numpy").array(
            points,
            dtype=__import__("numpy").int32
        )
    )

    # Transparent fill
    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [pts.get()],
        (0, 255, 255)
    )

    frame = cv2.addWeighted(
        overlay,
        0.15,
        frame,
        0.85,
        0
    )

    # Polygon boundary
    cv2.polylines(
        frame,
        [pts.get()],
        True,
        (0, 255, 255),
        3
    )

    return frame


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    counted_ids_per_hour
):

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "hour",
            "pedestrian_entries"
        ])

        for hour in sorted(
            counted_ids_per_hour.keys()
        ):

            writer.writerow([
                hour,
                len(
                    counted_ids_per_hour[hour]
                )
            ])


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    counted_ids_per_hour,
    frame_idx,
    total_frames
):

    with open(
        CHECKPOINT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "total_frames",
            "pedestrian_entries"
        ])

        total = sum(
            len(ids)
            for ids
            in counted_ids_per_hour.values()
        )

        writer.writerow([
            frame_idx,
            total_frames,
            total
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("MVSA YOLO11x PEDESTRIAN TRACKING")
    print("=" * 70)
    print()

    print(
        f"Input video : {VIDEO_PATH}"
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Output video: {OUTPUT_VIDEO}"
    )

    print()

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not os.path.exists(
        VIDEO_PATH
    ):
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Start time
    # --------------------------------------------------------

    start_time = get_start_time_from_filename(
        VIDEO_PATH
    )

    print(
        "Video start:",
        start_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print()
    print("Loading YOLO11x...")

    model = YOLO(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    print("Opening video...")

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 20.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
    )

    print()
    print("Video information")
    print("------------------")
    print(
        f"Resolution : {width}x{height}"
    )
    print(
        f"FPS        : {fps:.2f}"
    )
    print(
        f"Frames     : {total_frames}"
    )
    print(
        f"Duration   : {duration / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    # mp4v is widely available on cluster systems.
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    print()
    print(
        f"Saving annotated video to:"
    )

    print(
        OUTPUT_VIDEO
    )

    print()

    # --------------------------------------------------------
    # Counting state
    # --------------------------------------------------------

    counted_ids_per_hour = defaultdict(
        set
    )

    counted_track_ids = set()

    frame_idx = 0
    total_entries = 0

    start_wall_time = time.time()
    last_progress_time = start_wall_time

    # --------------------------------------------------------
    # YOLO tracking
    # --------------------------------------------------------

    print(
        "Starting YOLO11x + ByteTrack..."
    )

    print()

    results_generator = model.track(

        source=VIDEO_PATH,

        classes=[0],

        conf=CONF_THRESHOLD,

        imgsz=IMG_SIZE,

        tracker="bytetrack.yaml",

        persist=True,

        stream=True,

        device=0,

        verbose=False
    )

    try:

        for result in results_generator:

            frame_idx += 1

            # ------------------------------------------------
            # Frame skipping
            # ------------------------------------------------

            if (
                frame_idx %
                PROCESS_EVERY_N_FRAMES != 0
            ):

                continue

            timestamp = get_frame_timestamp(
                frame_idx,
                fps,
                start_time
            )

            hour = hour_bucket(
                timestamp
            )

            # ------------------------------------------------
            # Start with YOLO annotated frame
            # ------------------------------------------------

            annotated = result.plot(
                boxes=True,
                labels=True,
                conf=True
            )

            # ------------------------------------------------
            # ROI
            # ------------------------------------------------

            annotated = draw_roi(
                annotated,
                PEDESTRIAN_ROI
            )

            # ------------------------------------------------
            # Process detections
            # ------------------------------------------------

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences
                ):

                    point = bottom_center(
                        box
                    )

                    if not PEDESTRIAN_ROI.contains(
                        point
                    ):
                        continue

                    if track_id in counted_track_ids:
                        continue

                    counted_ids_per_hour[
                        hour
                    ].add(
                        track_id
                    )

                    counted_track_ids.add(
                        track_id
                    )

                    total_entries += 1

                    print(
                        f"[COUNT] "
                        f"{timestamp.strftime('%H:%M:%S')} "
                        f"ID={track_id} "
                        f"conf={confidence:.2f} "
                        f"hour={hour} "
                        f"total={total_entries}"
                    )

            # ------------------------------------------------
            # Add information overlay
            # ------------------------------------------------

            current_hour_count = len(
                counted_ids_per_hour[hour]
            )

            cv2.rectangle(
                annotated,
                (10, 10),
                (430, 105),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Hour count: {current_hour_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Total: {total_entries}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Write annotated frame
            # ------------------------------------------------

            writer.write(
                annotated
            )

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                frame_idx %
                CHECKPOINT_EVERY_FRAMES == 0
            ):

                save_checkpoint(
                    counted_ids_per_hour,
                    frame_idx,
                    total_frames
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            now = time.time()

            if (
                now - last_progress_time
                >= 10
            ):

                progress = (
                    frame_idx /
                    total_frames *
                    100
                )

                elapsed = (
                    now -
                    start_wall_time
                )

                if frame_idx > 0:

                    estimated_total = (
                        elapsed *
                        total_frames /
                        frame_idx
                    )

                    eta = (
                        estimated_total -
                        elapsed
                    )

                else:

                    eta = 0

                print(
                    f"[PROGRESS] "
                    f"{progress:6.2f}% | "
                    f"{frame_idx}/{total_frames} | "
                    f"entries={total_entries} | "
                    f"ETA={eta / 60:.1f} min"
                )

                last_progress_time = now

    except KeyboardInterrupt:

        print()
        print(
            "Interrupted."
        )

        print(
            "Saving current results..."
        )

    finally:

        cap.release()
        writer.release()

    # --------------------------------------------------------
    # Final CSV
    # --------------------------------------------------------

    save_csv(
        counted_ids_per_hour
    )

    save_checkpoint(
        counted_ids_per_hour,
        frame_idx,
        total_frames
    )

    elapsed = (
        time.time()
        - start_wall_time
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Frames processed : {frame_idx}"
    )

    print(
        f"Processing time  : "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Total entries    : "
        f"{total_entries}"
    )

    print()

    print(
        "Hourly counts:"
    )

    for hour in sorted(
        counted_ids_per_hour.keys()
    ):

        print(
            f"  {hour} -> "
            f"{len(counted_ids_per_hour[hour])}"
        )

    print()

    print(
        f"Video: {OUTPUT_VIDEO}"
    )

    print(
        f"CSV:   {OUTPUT_CSV}"
    )

    print(
        f"Check: {CHECKPOINT_CSV}"
    )

    print()


if __name__ == "__main__":
    main()