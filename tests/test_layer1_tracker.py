"""Tests for scripts/traffic_layer1/tracker.py::Layer1Tracker -- ROI-entry
counting across two detector sources at once, with per-source track-id
namespacing. No GPU/model dependencies. Detections here are already
bucket-tagged (as taxonomy.py::tag_and_filter would produce), matching what
the tracker actually receives from pipeline.py.
"""

from datetime import datetime

from layer1_models import SourcedDetection, TRACK_HISTORY_MAXLEN
from roi import PedestrianROI
from layer1_tracker import Layer1Tracker

SQUARE_ROI = PedestrianROI(
    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], reference_width=100, reference_height=100
)
T0 = datetime(2026, 7, 28, 8, 0, 0)
INSIDE_BBOX = (40.0, 40.0, 60.0, 60.0)  # bottom-center (50, 60) is inside
OUTSIDE_BBOX = (150.0, 150.0, 170.0, 170.0)


def _det(source, raw_track_id, bucket, bbox=INSIDE_BBOX, conf=0.9, native_class=None):
    return SourcedDetection(
        source=source,
        native_class=native_class or bucket,
        bbox=bbox,
        conf=conf,
        raw_track_id=raw_track_id,
        bucket=bucket,
    )


def test_track_counted_once_when_entering_roi():
    tracker = Layer1Tracker(SQUARE_ROI, frame_width=100, frame_height=100)

    newly = tracker.update(0, T0, [_det("uvh26", 1, "Car")])
    assert len(newly) == 1
    assert newly[0].final_class == "Car"
    assert newly[0].counted is True

    newly_again = tracker.update(1, T0, [_det("uvh26", 1, "Car")])
    assert newly_again == []
    assert sum(1 for s in tracker.tracks.values() if s.counted) == 1


def test_track_outside_roi_never_counted():
    tracker = Layer1Tracker(SQUARE_ROI, frame_width=100, frame_height=100)

    newly = tracker.update(0, T0, [_det("uvh26", 2, "Auto", bbox=OUTSIDE_BBOX)])
    assert newly == []
    assert all(not s.counted for s in tracker.tracks.values())


def test_detection_with_no_bucket_is_ignored():
    tracker = Layer1Tracker(SQUARE_ROI, frame_width=100, frame_height=100)

    newly = tracker.update(0, T0, [_det("uvh26", 3, None)])
    assert newly == []
    assert tracker.tracks == {}


def test_detection_without_track_id_is_skipped():
    tracker = Layer1Tracker(SQUARE_ROI, frame_width=100, frame_height=100)
    det = SourcedDetection(source="uvh26", native_class="Car", bbox=INSIDE_BBOX, conf=0.9, raw_track_id=None, bucket="Car")

    newly = tracker.update(0, T0, [det])
    assert newly == []
    assert tracker.tracks == {}


def test_same_raw_track_id_from_different_sources_does_not_collide():
    tracker = Layer1Tracker(SQUARE_ROI, frame_width=100, frame_height=100)

    # UVH-26 track 7 and COCO track 7 are unrelated physical objects.
    newly = tracker.update(
        0,
        T0,
        [_det("uvh26", 7, "Car"), _det("coco", 7, "Pedestrian")],
    )
    assert len(newly) == 2
    global_ids = {s.track_id for s in newly}
    assert len(global_ids) == 2  # distinct synthetic global ids, no collision
    buckets = {s.final_class for s in newly}
    assert buckets == {"Car", "Pedestrian"}


def test_track_history_is_bounded():
    tracker = Layer1Tracker(SQUARE_ROI, frame_width=100, frame_height=100)

    for frame_idx in range(TRACK_HISTORY_MAXLEN + 50):
        tracker.update(frame_idx, T0, [_det("uvh26", 9, "Car")])

    state = tracker.tracks[("uvh26", 9)]
    assert len(state.bbox_history) == TRACK_HISTORY_MAXLEN
    assert len(state.center_history) == TRACK_HISTORY_MAXLEN
    assert len(state.confidence_history) == TRACK_HISTORY_MAXLEN
    assert state.frames_seen == TRACK_HISTORY_MAXLEN + 50
