"""
Hourly unique-pedestrian counter using YOLO11 + ByteTrack, with rider/vehicle
filtering so motorcycle/car/bus occupants aren't counted as pedestrians.

Core logic:
  - Run detection+tracking on person + vehicle classes together.
  - For each person track_id, check if their feet-point falls inside (or near)
    a vehicle box. If so, they're a rider/occupant -> skip.
  - Otherwise: count the track_id ONCE per hour bucket, the moment it first
    enters the pedestrian ROI. Later frames of the same track_id are ignored.
  - Timestamps come from video start time + frame index / fps. Swap
    `get_frame_timestamp` for OCR'd timestamps if frame drops are a concern.

Two run modes:
  DEBUG_MODE = True   -> writes an annotated video (person boxes, vehicle
                         boxes, ROI outline, and which people got rejected as
                         riders) so you can visually validate the filtering
                         before trusting the counts. Use this first, on ~25s
                         of footage, per the tuning step.
  DEBUG_MODE = False  -> runs the full pass and writes hourly_pedestrian_counts.csv

Install:
    pip install ultralytics shapely --break-system-packages
"""

from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import defaultdict
from shapely.geometry import Point, box as shapely_box
import cv2
import csv
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these for your camera
# ---------------------------------------------------------------------------

VIDEO_PATH = "videos/sample.mp4"
MODEL_PATH = "yolo11m.pt"          # start with 'm', not 'x' -- resolution matters more than model size here
CONF_THRESHOLD = 0.35
IMG_SIZE = 1280                     # higher res helps with small/far pedestrians
PROCESS_EVERY_N_FRAMES = 3          # skip frames to save compute; ByteTrack handles gaps

# COCO classes we need: person + everything a person could be "riding in/on"
PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck
ALL_CLASSES = [PERSON_CLASS] + list(VEHICLE_CLASSES)

# How close a person's feet-point must be to a vehicle box to be rejected
# as a rider rather than a pedestrian. Tune this during the debug pass --
# too small and riders leak through as false pedestrians, too large and
# people walking past parked bikes get wrongly excluded.
RIDER_REJECT_BUFFER_PX = 25

# Pedestrian ROI, hand-traced against the CVC4 sample frame (1920x1080).
# Fixed camera -- this does not need to change unless the camera moves.
from shapely.geometry import Polygon
PEDESTRIAN_ROI = Polygon([
    (14, 508),
    (57, 1024),
    (805, 331),
    (684, 221),
])


VIDEO_START_TIME = datetime(2026, 7, 18, 15, 0, 0)
OUTPUT_CSV = "hourly_pedestrian_counts.csv"

DEBUG_MODE = True                    # <-- flip to False once filtering looks right
DEBUG_OUTPUT_VIDEO = "debug_annotated.mp4"
DEBUG_MAX_FRAMES = 750               # ~25s at 30fps; enough for the manual tuning pass

# ---------------------------------------------------------------------------


def get_frame_timestamp(frame_idx: int, fps: float) -> datetime:
    """Map a frame index to a real-world timestamp. Swap this for OCR'd
    timestamps or a sidecar metadata file if the video ever drops frames."""
    seconds_elapsed = frame_idx / fps
    return VIDEO_START_TIME + timedelta(seconds=seconds_elapsed)


def hour_bucket(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:00")


def feet_point(box) -> Point:
    x1, y1, x2, y2 = box
    # Bottom-center of the box -- the ground-contact point, not the box centroid.
    return Point((x1 + x2) / 2, y2)


def is_rider(person_feet: Point, vehicle_boxes) -> bool:
    """True if this person's feet-point falls inside or near any vehicle box --
    i.e. they're almost certainly riding/sitting in it, not walking beside it."""
    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vbox
        vehicle_poly = shapely_box(vx1, vy1, vx2, vy2)
        if vehicle_poly.buffer(RIDER_REJECT_BUFFER_PX).contains(person_feet):
            return True
    return False


def run_debug_pass():
    """Annotates a short clip: person boxes (green=counted pedestrian,
    red=rejected as rider), vehicle boxes (blue), and the ROI outline (yellow).
    Watch this before trusting the real counting pass."""
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(DEBUG_OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    roi_pts = [(int(x), int(y)) for x, y in PEDESTRIAN_ROI.exterior.coords]

    results_gen = model.track(
        source=VIDEO_PATH, classes=ALL_CLASSES, conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE, tracker="bytetrack.yaml", persist=True,
        stream=True, verbose=False,
    )

    frame_idx = 0
    for result in results_gen:
        frame_idx += 1
        if frame_idx > DEBUG_MAX_FRAMES:
            break

        frame = result.orig_img.copy()
        cv2.polylines(frame, [np.array(roi_pts)],
                      isClosed=True, color=(0, 255, 255), thickness=2)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            vehicle_boxes = [b for b, c in zip(boxes, cls_ids) if c in VEHICLE_CLASSES]

            for b, c, tid in zip(boxes, cls_ids, track_ids):
                x1, y1, x2, y2 = map(int, b)
                if c in VEHICLE_CLASSES:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    continue
                if c != PERSON_CLASS:
                    continue

                fp = feet_point(b)
                rejected = is_rider(fp, vehicle_boxes)
                in_roi = PEDESTRIAN_ROI.contains(fp)

                if rejected:
                    color = (0, 0, 255)     # red = rejected as rider
                    label = f"id{tid} RIDER"
                elif in_roi:
                    color = (0, 255, 0)     # green = counted pedestrian
                    label = f"id{tid} COUNT"
                else:
                    color = (0, 165, 255)   # orange = person, outside ROI
                    label = f"id{tid} outROI"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (int(fp.x), int(fp.y)), 4, color, -1)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Debug video written: {DEBUG_OUTPUT_VIDEO}. "
          f"Check red (rejected riders) vs green (counted pedestrians) boxes.")


def run_counting_pass():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    counted_ids_per_hour = defaultdict(set)
    seen_ids_ever = set()
    frame_idx = 0

    results_gen = model.track(
        source=VIDEO_PATH, classes=ALL_CLASSES, conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE, tracker="bytetrack.yaml", persist=True,
        stream=True, verbose=False,
    )

    for result in results_gen:
        frame_idx += 1
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            continue
        if result.boxes is None or result.boxes.id is None:
            continue

        ts = get_frame_timestamp(frame_idx, fps)
        hkey = hour_bucket(ts)

        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        vehicle_boxes = [b for b, c in zip(boxes, cls_ids) if c in VEHICLE_CLASSES]

        for b, c, tid in zip(boxes, cls_ids, track_ids):
            if c != PERSON_CLASS:
                continue
            if tid in seen_ids_ever:
                continue

            fp = feet_point(b)
            if is_rider(fp, vehicle_boxes):
                continue  # rider/occupant, not a pedestrian -- never count this id
            if PEDESTRIAN_ROI.contains(fp):
                counted_ids_per_hour[hkey].add(tid)
                seen_ids_ever.add(tid)

    cap.release()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "unique_pedestrian_count"])
        for hkey in sorted(counted_ids_per_hour.keys()):
            writer.writerow([hkey, len(counted_ids_per_hour[hkey])])

    print(f"Done. Wrote {OUTPUT_CSV}")
    for hkey in sorted(counted_ids_per_hour.keys()):
        print(f"{hkey} -> {len(counted_ids_per_hour[hkey])} unique pedestrians")


if __name__ == "__main__":
    if DEBUG_MODE:
        run_debug_pass()
    else:
        run_counting_pass()

# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# 1. Workflow: run with DEBUG_MODE=True first on a representative ~25s clip.
#    Watch debug_annotated.mp4 -- confirm riders show red, walking pedestrians
#    show green. Adjust RIDER_REJECT_BUFFER_PX up/down accordingly, re-run
#    debug, repeat. Only then flip DEBUG_MODE=False for the real pass.
#
# 2. This is a spatial heuristic, not true rider detection. It will misfire
#    on: a pedestrian standing directly behind/beside a parked vehicle such
#    that their feet-point lands inside the vehicle's box (false reject), or
#    a rider whose person-box barely overlaps their bike at a distance (false
#    accept). The buffer tuning step exists specifically to minimize both.
#    If false rejects are common in your footage (e.g. people browsing next
#    to parked bikes outside the shops), consider requiring the rejection to
#    hold for >=2 consecutive frames before applying it, rather than any
#    single frame.
#
# 3. Model size / resolution notes from before still apply: start with
#    yolo11m at imgsz=1280, only move to 'x' if verified misses on 'm'.
#
# 4. bbox->feet-point and hour-boundary-by-first-entry logic are unchanged
#    from the previous version -- see inline comments.
#
# 5. Install: pip install ultralytics shapely --break-system-packages