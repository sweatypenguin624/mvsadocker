"""Tests for scripts/roi.py -- point-in-polygon inclusion and bottom-center
bbox math. No GPU/model dependencies required.
"""

from roi import PedestrianROI, bottom_center, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside_square():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_outside_square():
    assert point_in_polygon((15, 5), SQUARE) is False
    assert point_in_polygon((-5, 5), SQUARE) is False
    assert point_in_polygon((5, -5), SQUARE) is False
    assert point_in_polygon((5, 15), SQUARE) is False


def test_bottom_center_of_bbox():
    bbox = (100.0, 200.0, 140.0, 260.0)
    x, y = bottom_center(bbox)
    assert x == 120.0
    assert y == 260.0


def test_pedestrian_roi_contains_point_reference_resolution():
    roi = PedestrianROI(polygon=SQUARE, reference_width=1920, reference_height=1080)
    assert roi.contains_point((5, 5)) is True
    assert roi.contains_point((50, 50)) is False


def test_pedestrian_roi_scales_to_different_frame_size():
    # ROI defined at 1920x1080; a point that would be inside at native
    # resolution should still be classified correctly at half resolution.
    polygon_native = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
    roi = PedestrianROI(polygon=polygon_native, reference_width=1920, reference_height=1080)
    assert roi.contains_point((480, 270), frame_width=960, frame_height=540) is True
    assert roi.contains_point((2000, 270), frame_width=960, frame_height=540) is False


def test_roi_rejects_degenerate_polygon():
    import pytest

    with pytest.raises(ValueError):
        PedestrianROI(polygon=[(0, 0), (1, 1)])


def test_bottom_center_used_for_roi_membership_not_box_center():
    # A tall bbox whose center is outside the ROI but whose feet point is
    # inside must be classified as "inside" -- this is the whole point of
    # using bottom-center instead of the box centroid.
    roi = PedestrianROI(polygon=SQUARE)
    tall_bbox = (4.0, -50.0, 6.0, 9.0)  # center y = -20.5 (outside), feet y = 9 (inside)
    feet = bottom_center(tall_bbox)
    assert roi.contains_point(feet) is True
