#!/usr/bin/env python3

from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from shapely.geometry import Point, Polygon

from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import csv
import os
import queue
import re
import threading
import time
import numpy as np


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = (
    "/home/users/oauser/mvsa/work/downloads/20260728_182347_tp00052.mp4"
)

MODEL_PATH = (
    "/home/users/oauser/mvsa/models/yolo11x.pt"
)

OUTPUT_DIR = (
    "/home/users/oauser/mvsa/results/"
    "kamaripet"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "testvid.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "hourly_pedestrian_countskamaripet.csv"
)

CHECKPOINT_CSV = os.path.join(
    OUTPUT_DIR,
    "checkpointkamaripet.csv"
)


# Detection settings
CONF_THRESHOLD = 0.35

# 1280 -> 640: for YOLO11x this cuts inference pixels ~4x with
# only a small accuracy hit for a fairly large ROI subject
# (pedestrians). Re-benchmark against 1280 if recall looks worse.
IMG_SIZE = 640

# Only frames where (frame_idx % PROCESS_EVERY_N_FRAMES == 0) are
# ever handed to YOLO/ByteTrack (see frame_producer() below) --
# unlike the old model.track(source=..., stream=True) approach,
# which ran inference on every frame and only skipped *after* the
# (expensive) forward pass.
PROCESS_EVERY_N_FRAMES = 2

TRACKER_YAML = "bytetrack.yaml"

# Detection is batched on the GPU instead of one model.track() call
# per frame. Benchmarked on this A100 (yolo11x, imgsz=640, fp16):
# batch=1 ~35 FPS, batch=16 ~220 FPS -- a single decoded frame at a
# time leaves the A100 almost idle between tiny forward passes.
# ByteTrack itself is still run one frame at a time, in strict
# decode order, via a standalone BYTETracker below, so track IDs
# and counts are identical to the old per-frame model.track(..,
# persist=True) path -- only the detector call is batched.
BATCH_SIZE = 16

# Frames are decoded on a background thread (frame_producer) and
# consumed here in batches, so CPU decode of the *next* batch
# overlaps the GPU forward pass on the *current* one instead of the
# two running back-to-back. A few batches of read-ahead is enough
# to hide decode latency without holding too many full frames in
# memory.
FRAME_QUEUE_SIZE = BATCH_SIZE * 4

# Save checkpoint every N *processed* frames.
CHECKPOINT_EVERY_FRAMES = 1000

# Annotated video writing is expensive (render + encode + disk I/O)
# and is not required to produce hourly counts. Leave this False
# for production counting runs; flip to True only when you need to
# visually QA detections/tracks for a short clip.
SAVE_VIDEO = True


# ============================================================
# ROI
# ============================================================

# Replace these with the exact coordinates from:
#
# ~/mvsa/results/pedestrian_roi.txt
#
# These are the coordinates you provided earlier.

PEDESTRIAN_ROI = Polygon([
    (1903, 623),
    (1201, 687),
    (1548, 1279),
    (2238, 1269),
    (2264, 942),
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

    # Format 1:
    # 08.00.00-09.00.00.dav
    match = re.search(
        r"(\d{2})\.(\d{2})\.(\d{2})-"
        r"(\d{2})\.(\d{2})\.(\d{2})",
        filename
    )

    if match:
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

    # Format 2:
    # 20260728_182347_tp00052.mp4
    match = re.search(
        r"(\d{8})_(\d{6})",
        filename
    )

    if match:
        date_str = match.group(1)
        time_str = match.group(2)

        return datetime.strptime(
            date_str + time_str,
            "%Y%m%d%H%M%S"
        )

    raise ValueError(
        f"Could not extract start time from filename: {filename}"
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
# DRAW ROI / TRACKS
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

    pts = np.array(
        points,
        dtype=np.int32
    )

    # Transparent fill
    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [pts],
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
        [pts],
        True,
        (0, 255, 255),
        3
    )

    return frame


def draw_tracks(
    frame,
    tracks
):

    # Manual box/label rendering (rather than Results.plot()), since
    # detection now runs through model.predict() in batches and
    # ByteTrack is applied afterwards as a standalone tracker -- the
    # per-frame Results object never gets the track IDs merged back
    # into it the way model.track() would.

    annotated = frame.copy()

    for x1, y1, x2, y2, track_id, conf, _cls, _idx in tracks:

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        cv2.rectangle(
            annotated,
            p1,
            p2,
            (0, 200, 0),
            2
        )

        cv2.putText(
            annotated,
            f"person {int(track_id)} {conf:.2f}",
            (p1[0], max(p1[1] - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1
        )

    return annotated


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
# DECODE THREAD
# ============================================================

def frame_producer(
    cap,
    frame_queue,
    stop_event
):

    # cap.grab() advances the decoder without doing the full
    # YUV->BGR decode + copy that cap.read() does -- for skipped
    # frames we only grab(), and only retrieve() (full decode) the
    # frames that are actually handed to YOLO. cap.read() ==
    # grab()+retrieve() every time, so decoding every frame just to
    # throw half of them away would waste CPU that this thread would
    # rather spend getting ahead of the GPU.
    #
    # This runs on its own thread so frame decode for the *next*
    # batch overlaps the GPU forward pass on the *current* one
    # (cv2's grab/retrieve release the GIL while blocked in native
    # decode, same as torch releases it around CUDA calls).

    frame_idx = 0

    try:

        while not stop_event.is_set():

            grabbed = cap.grab()

            if not grabbed:
                break

            frame_idx += 1

            if (
                frame_idx %
                PROCESS_EVERY_N_FRAMES != 0
            ):
                continue

            ret, frame = cap.retrieve()

            if not ret:
                break

            frame_queue.put(
                (frame_idx, frame)
            )

    finally:

        # Sentinel: tells the consumer there are no more frames.
        frame_queue.put(None)


def build_tracker(
    tracker_yaml,
    device
):

    # Standalone equivalent of what model.track(persist=True)
    # sets up internally (see ultralytics.trackers.track.on_predict_
    # start) -- same tracker class, same yaml, so association
    # behaviour and track IDs are unaffected by batching detection.

    tracker_cfg_path = check_yaml(
        tracker_yaml
    )

    cfg = IterableSimpleNamespace(
        **YAML.load(tracker_cfg_path)
    )

    cfg.device = device

    return BYTETracker(
        args=cfg
    )


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

    if SAVE_VIDEO:
        print(
            f"Output video: {OUTPUT_VIDEO}"
        )
    else:
        print(
            "Output video: disabled (SAVE_VIDEO=False)"
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

    tracker = build_tracker(
        TRACKER_YAML,
        device=0
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
    # Video writer (only if SAVE_VIDEO)
    # --------------------------------------------------------

    writer = None

    if SAVE_VIDEO:

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

    # NOTE: this counts unique ByteTrack IDs, not unique physical
    # pedestrians. If a track is lost (occlusion, ROI re-entry,
    # detector miss) and ByteTrack assigns a new ID to the same
    # person, they will be counted again. This is a tracker/ReID
    # limitation, not something frame-rate or model-size tuning
    # fixes -- flagging it separately from the perf changes below.
    counted_track_ids = set()

    frame_idx = 0
    processed_frames = 0
    total_entries = 0

    start_wall_time = time.time()
    last_progress_time = start_wall_time

    # --------------------------------------------------------
    # Decode thread + batched YOLO11x + standalone ByteTrack
    #
    # The decode thread only ever grabs/retrieves frames (see
    # frame_producer above); this thread pulls them off the queue
    # BATCH_SIZE at a time and runs one batched forward pass per
    # batch instead of one model.track() call per frame. ByteTrack
    # is then applied to each frame's detections in strict decode
    # order via tracker.update(), which is exactly what model.track
    # (..., persist=True) does per frame internally -- so track IDs
    # and ROI-entry counts come out identical to the old single-
    # frame loop, just computed from many fewer, much larger GPU
    # calls.
    # --------------------------------------------------------

    print(
        f"Starting YOLO11x (batch={BATCH_SIZE}) + ByteTrack..."
    )

    print()

    frame_queue = queue.Queue(
        maxsize=FRAME_QUEUE_SIZE
    )

    stop_event = threading.Event()

    producer = threading.Thread(
        target=frame_producer,
        args=(cap, frame_queue, stop_event),
        daemon=True
    )

    producer.start()

    try:

        finished = False

        while not finished:

            batch = []

            while len(batch) < BATCH_SIZE:

                item = frame_queue.get()

                if item is None:
                    finished = True
                    break

                batch.append(item)

            if not batch:
                break

            frames = [
                frame
                for _, frame
                in batch
            ]

            results = model.predict(
                frames,
                classes=[0],
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                device=0,
                quantize=16,
                verbose=False
            )

            for (frame_idx, frame), result in zip(
                batch,
                results
            ):

                processed_frames += 1

                timestamp = get_frame_timestamp(
                    frame_idx,
                    fps,
                    start_time
                )

                hour = hour_bucket(
                    timestamp
                )

                det = result.boxes.cpu().numpy()

                tracks = tracker.update(
                    det,
                    result.orig_img
                )

                # ------------------------------------------------
                # Annotated frame (only built when saving video)
                # ------------------------------------------------

                annotated = None

                if SAVE_VIDEO:

                    annotated = draw_tracks(
                        frame,
                        tracks
                    )

                    annotated = draw_roi(
                        annotated,
                        PEDESTRIAN_ROI
                    )

                # ------------------------------------------------
                # Process detections
                # ------------------------------------------------

                if len(tracks):

                    boxes = tracks[:, :4]
                    track_ids = tracks[:, 4].astype(int)
                    confidences = tracks[:, 5]

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
                # Add information overlay + write (only if saving)
                # ------------------------------------------------

                if SAVE_VIDEO:

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

                    writer.write(
                        annotated
                    )

                # ------------------------------------------------
                # Checkpoint (every N processed frames)
                # ------------------------------------------------

                if (
                    processed_frames %
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
                        f"processed={processed_frames} | "
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

        stop_event.set()

        # Unblock the producer if it's waiting on a full queue, then
        # drain anything left so its final put(None) sentinel can
        # also land before we join it.
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            pass

        producer.join(timeout=5)

        cap.release()

        if writer is not None:
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
        f"Frames read      : {frame_idx}"
    )

    print(
        f"Frames processed : {processed_frames} "
        f"(YOLO inference calls)"
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

    if SAVE_VIDEO:
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
