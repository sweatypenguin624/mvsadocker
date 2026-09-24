"""Stage 1 unit tests.

Focused on the three places a vehicle can be double-counted or lost:
box fusion, identity merging, and line crossing.
"""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

STAGE1 = Path(__file__).resolve().parent.parent / "scripts" / "traffic_stage1"
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))

from stage1_config import CountingConfig, FusionConfig, IdentityConfig  # noqa: E402
from stage1_count import build_intervals, count_vehicles, interval_label  # noqa: E402
from stage1_fusion import fuse_boxes, should_fuse  # noqa: E402
from stage1_geometry import containment, iom, iou, segment_intersects, union_box  # noqa: E402
from stage1_identity import Observation, RawTrack, resolve_identities  # noqa: E402


# --------------------------------------------------------------- geometry
def test_iom_catches_containment_that_iou_misses():
    big = [0, 0, 400, 100]
    panel = [10, 10, 110, 90]
    assert iou(big, panel) < 0.25          # IoU-NMS would never suppress this
    assert iom(big, panel) > 0.95          # ...but containment is obvious


def test_segment_intersects_respects_finite_line():
    line = ([0, 100], [100, 100])
    assert segment_intersects([50, 50], [50, 150], *line)     # crosses within span
    assert not segment_intersects([500, 50], [500, 150], *line)  # past the endpoint


# ----------------------------------------------------------------- fusion
def test_fuses_a_duplicate_box_on_one_vehicle():
    """Two boxes on one vehicle, one largely swallowing the other."""
    cfg = FusionConfig()
    boxes = np.array([[100, 300, 500, 700], [140, 320, 480, 690]], dtype=np.float32)
    confs = np.array([0.9, 0.7], dtype=np.float32)
    fused, fused_confs, groups = fuse_boxes(boxes, confs, cfg)
    assert len(fused) == 1
    assert groups[0] == [0, 1]
    assert fused[0].tolist() == [100, 300, 500, 700]
    assert fused_confs[0] == pytest.approx(0.9)


def test_tiled_panels_are_beyond_per_frame_fusion():
    """Documents WHY lockstep identity merging exists.

    One bus detected as three boxes tiled front-to-back: adjacent panels
    overlap only slightly and the outer two not at all, so neither IoU nor
    IoM can group them from a single frame without also merging genuinely
    adjacent vehicles. This case is handled over time instead -- see
    test_lockstep_merges_tiled_panels_of_one_bus.
    """
    cfg = FusionConfig()
    boxes = np.array([
        [100, 300, 500, 700],   # front section
        [420, 300, 820, 700],   # door section
        [740, 300, 1140, 700],  # rear section
    ], dtype=np.float32)
    assert iom(boxes[0], boxes[1]) < cfg.iom_thresh
    assert iom(boxes[0], boxes[2]) == 0.0
    fused, _, _ = fuse_boxes(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), cfg)
    assert len(fused) == 3


def test_does_not_fuse_a_motorcycle_in_front_of_a_bus():
    cfg = FusionConfig()
    bus = [0, 0, 800, 600]
    motorcycle = [300, 400, 380, 560]   # fully inside the bus box, tiny
    assert containment(motorcycle, bus) == pytest.approx(1.0)
    assert not should_fuse(bus, motorcycle, cfg)   # area ratio guard saves it

    fused, _, _ = fuse_boxes(
        np.array([bus, motorcycle], dtype=np.float32), np.array([0.9, 0.8], dtype=np.float32), cfg
    )
    assert len(fused) == 2


def test_two_separate_vehicles_are_left_alone():
    cfg = FusionConfig()
    boxes = np.array([[0, 0, 100, 100], [300, 0, 400, 100]], dtype=np.float32)
    fused, _, _ = fuse_boxes(boxes, np.array([0.9, 0.9], dtype=np.float32), cfg)
    assert len(fused) == 2


def test_fusion_disabled_is_a_passthrough():
    cfg = FusionConfig(enabled=False)
    boxes = np.array([[0, 0, 400, 100], [10, 10, 110, 90]], dtype=np.float32)
    fused, _, groups = fuse_boxes(boxes, np.array([0.9, 0.8], dtype=np.float32), cfg)
    assert len(fused) == 2 and groups == [[0], [1]]


# --------------------------------------------------------------- identity
def _track(tid, start, count, x0, y0, dx=0.0, dy=10.0, w=80, h=80, conf=0.9):
    obs = [
        Observation(start + i, [x0 + dx * i, y0 + dy * i, x0 + dx * i + w, y0 + dy * i + h], conf)
        for i in range(count)
    ]
    return RawTrack(track_id=tid, observations=obs)


def test_spatial_dedup_merges_two_tracks_riding_one_vehicle():
    cfg = IdentityConfig()
    a = _track(1, 0, 30, 100, 100)
    b = _track(2, 0, 30, 110, 105)   # same object, slightly offset box
    vehicles, stats = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 1
    assert stats["spatial_merges"] == 1
    assert vehicles[0].raw_track_ids == [1, 2]
    # One observation per frame after merging, not two.
    assert len({o.frame_idx for o in vehicles[0].observations}) == len(vehicles[0].observations)


def test_two_genuinely_separate_vehicles_stay_separate():
    cfg = IdentityConfig()
    a = _track(1, 0, 30, 100, 100)
    b = _track(2, 0, 30, 900, 100)
    vehicles, stats = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2
    assert stats["spatial_merges"] == 0


def test_briefly_overlapping_vehicles_are_not_merged():
    """One vehicle passing another overlaps for a few frames only."""
    cfg = IdentityConfig()
    a = _track(1, 0, 40, 100, 0, dy=20)
    b = _track(2, 0, 40, 100, 400, dy=20)   # ahead, same speed -> never really overlaps
    vehicles, _ = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2


def test_lockstep_merges_tiled_panels_of_one_bus():
    """THE regression case: one bus carrying three simultaneous track ids.

    Front/door/rear panel boxes barely overlap, so per-frame fusion cannot
    touch them -- but they are bolted to one body, so their relative
    offsets never change as the bus moves.
    """
    cfg = IdentityConfig()
    front = _track(1, 0, 30, 100, 300, dx=6, dy=4, w=400, h=400)
    door = _track(2, 0, 30, 420, 300, dx=6, dy=4, w=400, h=400)
    rear = _track(3, 0, 30, 740, 300, dx=6, dy=4, w=400, h=400)
    vehicles, stats = resolve_identities({1: front, 2: door, 3: rear}, cfg)
    assert len(vehicles) == 1
    assert vehicles[0].raw_track_ids == [1, 2, 3]
    assert stats["lockstep_merges"] == 2
    # The merged observation must span the whole bus, not one panel.
    box = vehicles[0].observations[0].bbox
    assert box[0] == pytest.approx(100) and box[2] == pytest.approx(1140)


def test_lockstep_leaves_adjacent_lane_vehicles_alone():
    """Two vehicles side by side at similar speeds must stay separate."""
    cfg = IdentityConfig()
    a = _track(1, 0, 30, 100, 300, dx=0, dy=12, w=200, h=200)
    b = _track(2, 0, 30, 340, 300, dx=0, dy=9, w=200, h=200)  # drifts apart
    vehicles, stats = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2
    assert stats["lockstep_merges"] == 0


def test_lockstep_ignores_non_touching_boxes():
    """Perfect lockstep but a real gap between them -> two vehicles."""
    cfg = IdentityConfig()
    a = _track(1, 0, 30, 100, 300, dx=5, dy=5, w=200, h=200)
    b = _track(2, 0, 30, 800, 300, dx=5, dy=5, w=200, h=200)
    vehicles, stats = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2
    assert stats["lockstep_merges"] == 0


def test_temporal_stitch_recovers_an_id_switch():
    """Track dies under an occlusion, reappears with a new id."""
    cfg = IdentityConfig()
    a = _track(1, 0, 20, 100, 0, dy=10)              # ends at frame 19, y ~190
    b = _track(2, 30, 20, 100, 300, dy=10)           # born at frame 30 where a would be
    vehicles, stats = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 1
    assert stats["temporal_merges"] == 1
    assert vehicles[0].raw_track_ids == [1, 2]


def test_no_stitch_across_an_implausible_jump():
    cfg = IdentityConfig()
    a = _track(1, 0, 20, 100, 0, dy=10)
    b = _track(2, 25, 20, 1600, 900, dy=10)   # nowhere near the prediction
    vehicles, stats = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2
    assert stats["temporal_merges"] == 0


def test_no_stitch_when_directions_disagree():
    cfg = IdentityConfig()
    a = _track(1, 0, 20, 100, 0, dy=10)       # heading down
    b = _track(2, 22, 20, 100, 200, dy=-10)   # heading up
    vehicles, _ = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2


def test_identity_disabled_keeps_raw_tracks():
    cfg = IdentityConfig(enabled=False)
    a = _track(1, 0, 30, 100, 100)
    b = _track(2, 0, 30, 110, 105)
    vehicles, _ = resolve_identities({1: a, 2: b}, cfg)
    assert len(vehicles) == 2


# --------------------------------------------------------------- counting
def _frame_time(fps=25.0, start=datetime(2026, 7, 20, 8, 0, 0)):
    from datetime import timedelta
    return lambda idx: start + timedelta(seconds=idx / fps)


def test_vehicle_crossing_the_line_counts_exactly_once():
    cfg = CountingConfig(line=[[0, 500], [1920, 500]], min_track_frames=6)
    v, _ = resolve_identities({1: _track(1, 0, 40, 900, 200, dy=20)}, IdentityConfig())
    counted, rejected, _, calibrated = count_vehicles(v, cfg, 1920, 1080, _frame_time())
    assert calibrated
    assert len(counted) == 1
    assert counted[0].direction == "down"


def test_vehicle_that_never_reaches_the_line_is_not_counted():
    cfg = CountingConfig(line=[[0, 900], [1920, 900]], min_track_frames=6)
    v, _ = resolve_identities({1: _track(1, 0, 20, 900, 100, dy=5)}, IdentityConfig())
    counted, rejected, _, _ = count_vehicles(v, cfg, 1920, 1080, _frame_time())
    assert counted == []
    assert rejected["never_crossed"] == 1


def test_box_jittering_on_the_line_does_not_count():
    """Ground point oscillating across the line must not register."""
    cfg = CountingConfig(line=[[0, 500], [1920, 500]], min_track_frames=6, min_frames_each_side=3)
    obs = []
    for i in range(20):
        y = 418 if i % 2 == 0 else 422      # bottom edge straddles y=500
        obs.append(Observation(i, [900, y, 980, y + 82], 0.9))
    v, _ = resolve_identities({1: RawTrack(1, obs)}, IdentityConfig())
    counted, rejected, _, _ = count_vehicles(v, cfg, 1920, 1080, _frame_time())
    assert counted == []
    assert rejected["insufficient_dwell"] == 1


def test_flicker_track_below_min_frames_is_rejected():
    cfg = CountingConfig(line=[[0, 500], [1920, 500]], min_track_frames=6)
    v, _ = resolve_identities({1: _track(1, 0, 3, 900, 400, dy=60)}, IdentityConfig())
    counted, rejected, _, _ = count_vehicles(v, cfg, 1920, 1080, _frame_time())
    assert counted == []
    assert rejected["too_few_frames"] == 1


def test_uncalibrated_fallback_line_is_flagged():
    cfg = CountingConfig(line=None)
    v, _ = resolve_identities({1: _track(1, 0, 40, 900, 200, dy=20)}, IdentityConfig())
    _, _, line, calibrated = count_vehicles(v, cfg, 1920, 1080, _frame_time())
    assert not calibrated
    assert line[0][1] == pytest.approx(1080 * 0.6)


def test_line_is_rescaled_from_reference_resolution():
    cfg = CountingConfig(line=[[0, 500], [1920, 500]], reference_width=1920, reference_height=1080)
    _, _, line, _ = count_vehicles([], cfg, 3840, 2160, _frame_time())
    assert line == [[0.0, 1000.0], [3840.0, 1000.0]]


# --------------------------------------------------------------- intervals
def test_interval_label_floors_to_15_minutes():
    assert interval_label(datetime(2026, 7, 20, 8, 14, 59), 15) == "2026-07-20 08:00"
    assert interval_label(datetime(2026, 7, 20, 8, 15, 0), 15) == "2026-07-20 08:15"
    assert interval_label(datetime(2026, 7, 20, 8, 59, 59), 15) == "2026-07-20 08:45"


def test_intervals_are_dense_including_empty_windows():
    cfg = CountingConfig(interval_minutes=15)
    rows = build_intervals([], cfg, datetime(2026, 7, 20, 8, 0), datetime(2026, 7, 20, 9, 0))
    assert [r["interval_start"] for r in rows] == [
        "2026-07-20 08:00", "2026-07-20 08:15", "2026-07-20 08:30",
        "2026-07-20 08:45", "2026-07-20 09:00",
    ]
    assert all(r["total_vehicles"] == 0 for r in rows)


# ------------------------------------------------------------ size filter
def test_size_filter_drops_scene_wide_and_speck_boxes():
    """Observed on real footage: one box spanning a whole intersection was
    tracked and counted as a vehicle."""
    from stage1_config import DetectorConfig
    from stage1_detect import VehicleDetector

    detector = VehicleDetector.__new__(VehicleDetector)   # no weights needed
    detector.cfg = DetectorConfig()
    detector.dropped_by_size = 0

    frame_area = 1920.0 * 1080.0
    boxes = np.array([
        [100, 100, 300, 280],        # a normal vehicle -> kept
        [0, 0, 1900, 1000],          # scene-wide hallucination -> dropped
        [500, 500, 503, 503],        # speck -> dropped
    ], dtype=np.float32)
    confs = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    kept_boxes, kept_confs = detector._size_filter(boxes, confs, frame_area)
    assert len(kept_boxes) == 1
    assert kept_boxes[0].tolist() == [100, 100, 300, 280]
    assert kept_confs.tolist() == [pytest.approx(0.9)]
    assert detector.dropped_by_size == 2


# ------------------------------------------------------- appearance gate
def _solid_hist(hue: int):
    """Fingerprint of a uniformly coloured vehicle."""
    from stage1_appearance import compute
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:, :] = cv2.cvtColor(
        np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    return compute(frame, [0, 0, 200, 200])


def test_appearance_separates_differently_coloured_vehicles():
    from stage1_appearance import similarity
    red, silver_ish = _solid_hist(0), _solid_hist(100)
    assert similarity(red, red) == pytest.approx(1.0, abs=1e-5)
    assert similarity(red, silver_ish) < 0.80


def test_appearance_similarity_is_none_when_unknown():
    """Missing fingerprints must read as 'no opinion', never as a mismatch."""
    from stage1_appearance import similarity
    assert similarity(None, _solid_hist(0)) is None
    assert similarity(None, None) is None


def test_appearance_veto_blocks_the_two_scooter_over_merge():
    """REGRESSION: caught on real GPU footage.

    A silver scooter and a red scooter travelled the same lane ~1.5s apart.
    Geometrically the second was a near-perfect continuation of the first
    (same path, same speed, landing 57px from the prediction), so it was
    stitched on and two vehicles were counted as one. Colour separates them.
    """
    cfg = IdentityConfig()
    first = _track(1, 0, 25, 400, 100, dy=12, w=180, h=180)
    second = _track(2, 64, 25, 400, 570, dy=12, w=180, h=180)   # 39-frame gap

    # Same colour -> still stitched (a genuine occlusion recovery).
    first.appearance = second.appearance = _solid_hist(0)
    vehicles, stats = resolve_identities({1: first, 2: second}, cfg)
    assert stats["temporal_merges"] == 1

    # Different colour -> vetoed, counted as two vehicles.
    first.appearance, second.appearance = _solid_hist(0), _solid_hist(100)
    vehicles, stats = resolve_identities({1: first, 2: second}, cfg)
    assert stats["temporal_merges"] == 0
    assert len(vehicles) == 2


def test_stitch_tolerance_scales_with_vehicle_size_not_gap_length():
    """The old flat per-frame pixel budget allowed ~1750px of slack across
    a 39-frame gap -- most of the frame width."""
    cfg = IdentityConfig()
    old = _track(1, 0, 25, 400, 100, dy=10, w=100, h=100)
    # Continuation predicted near y=590; place the new track far off to the
    # side, well beyond a few body-lengths but inside the old flat budget.
    new = _track(2, 60, 25, 1400, 590, dy=10, w=100, h=100)
    vehicles, stats = resolve_identities({1: old, 2: new}, cfg)
    assert stats["temporal_merges"] == 0
    assert len(vehicles) == 2


# --------------------------------------------------------- calibrate_line
def test_set_line_actually_persists_the_line(tmp_path):
    """REGRESSION: `dict.setdefault(k, {}) or {}` silently discarded the
    dict it just created (an empty dict is falsy), so the written config
    always had an empty `counting: {}` block and the line was lost."""
    import yaml
    from calibrate_line import set_line

    config_path = tmp_path / "stage1_config.yaml"
    config_path.write_text("cameras: {}\n", encoding="utf-8")

    set_line(config_path, "cam_a", [0.0, 520.0, 1920.0, 430.0], 1920, 1080)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    line = data["cameras"]["cam_a"]["counting"]["line"]
    assert line == [[0.0, 520.0], [1920.0, 430.0]]


def test_set_line_on_a_second_camera_does_not_clobber_the_first(tmp_path):
    import yaml
    from calibrate_line import set_line

    config_path = tmp_path / "stage1_config.yaml"
    config_path.write_text("cameras: {}\n", encoding="utf-8")

    set_line(config_path, "cam_a", [0.0, 100.0, 100.0, 100.0], 1920, 1080)
    set_line(config_path, "cam_b", [0.0, 200.0, 200.0, 200.0], 1920, 1080)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["cameras"]["cam_a"]["counting"]["line"] == [[0.0, 100.0], [100.0, 100.0]]
    assert data["cameras"]["cam_b"]["counting"]["line"] == [[0.0, 200.0], [200.0, 200.0]]
