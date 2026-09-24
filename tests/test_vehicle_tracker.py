"""Tests for scripts/vehicle_counting/vehicle_tracker.py -- ROI-entry
counting per vehicle class. No GPU/model dependencies required.
"""

from datetime import datetime

from models import Detection
from roi import PedestrianROI
from vehicle_tracker import VehicleTracker

SQUARE_ROI = PedestrianROI(
    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], reference_width=100, reference_height=100
)
CLASS_MAP = {1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS"}
T0 = datetime(2026, 7, 28, 8, 0, 0)


def _det(track_id, cls_id, bbox):
    return Detection(bbox=bbox, conf=0.9, cls_id=cls_id, track_id=track_id)


def test_track_counted_once_when_entering_roi():
    tracker = VehicleTracker(SQUARE_ROI, frame_width=100, frame_height=100)
    inside_bbox = (40.0, 40.0, 60.0, 60.0)  # bottom-center (50, 60) is inside

    newly = tracker.update(0, T0, [_det(1, 2, inside_bbox)], CLASS_MAP)
    assert len(newly) == 1
    assert newly[0].bucket == "CAR"
    assert 1 in tracker.counted_ids

    # Same track seen again inside the ROI must NOT be counted a second time.
    newly_again = tracker.update(1, T0, [_det(1, 2, inside_bbox)], CLASS_MAP)
    assert newly_again == []
    assert len(tracker.counted_ids) == 1


def test_track_outside_roi_never_counted():
    tracker = VehicleTracker(SQUARE_ROI, frame_width=100, frame_height=100)
    outside_bbox = (150.0, 150.0, 170.0, 170.0)

    newly = tracker.update(0, T0, [_det(2, 3, outside_bbox)], CLASS_MAP)
    assert newly == []
    assert tracker.counted_ids == set()


def test_detections_of_unmapped_class_are_ignored():
    tracker = VehicleTracker(SQUARE_ROI, frame_width=100, frame_height=100)
    inside_bbox = (40.0, 40.0, 60.0, 60.0)

    # cls_id 7 (truck) is not in CLASS_MAP -- must not raise or count.
    newly = tracker.update(0, T0, [_det(3, 7, inside_bbox)], CLASS_MAP)
    assert newly == []
    assert tracker.tracks == {}


def test_different_classes_tracked_independently():
    tracker = VehicleTracker(SQUARE_ROI, frame_width=100, frame_height=100)
    inside_bbox = (40.0, 40.0, 60.0, 60.0)

    dets = [_det(1, 2, inside_bbox), _det(2, 5, inside_bbox), _det(3, 3, inside_bbox)]
    newly = tracker.update(0, T0, dets, CLASS_MAP)
    buckets = {s.bucket for s in newly}
    assert buckets == {"CAR", "BUS", "MOTORCYCLE"}
    assert len(tracker.counted_ids) == 3


def test_detection_without_track_id_is_skipped():
    tracker = VehicleTracker(SQUARE_ROI, frame_width=100, frame_height=100)
    inside_bbox = (40.0, 40.0, 60.0, 60.0)
    det = Detection(bbox=inside_bbox, conf=0.9, cls_id=2, track_id=None)

    newly = tracker.update(0, T0, [det], CLASS_MAP)
    assert newly == []
    assert tracker.tracks == {}
