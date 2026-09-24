"""Tests for scripts/traffic_layer1/aggregator.py -- configurable time-bucket
flooring and per-class/directional totals arithmetic. No GPU/model
dependencies.
"""

from datetime import datetime

import pytest

from aggregator import (
    build_class_counts,
    build_direction_counts,
    build_direction_totals,
    build_time_buckets,
    time_bucket_key,
)
from layer1_models import TrackState

T0 = datetime(2026, 7, 28, 8, 0, 0)


def _state(track_id, final_class, direction_label, entered_roi_time):
    return TrackState(
        track_id=track_id,
        source="uvh26",
        raw_track_id=track_id,
        first_seen_frame=0,
        first_seen_time=entered_roi_time,
        last_seen_frame=1,
        last_seen_time=entered_roi_time,
        counted=True,
        final_class=final_class,
        direction_label=direction_label,
        entered_roi_time=entered_roi_time,
    )


def test_time_bucket_key_hourly_matches_hour_bucket_format():
    ts = datetime(2026, 7, 28, 8, 47, 12)
    assert time_bucket_key(ts, 60) == "2026-07-28 08:00"


def test_time_bucket_key_15_minute_floor():
    assert time_bucket_key(datetime(2026, 7, 28, 8, 47, 0), 15) == "2026-07-28 08:45"
    assert time_bucket_key(datetime(2026, 7, 28, 8, 44, 59), 15) == "2026-07-28 08:30"
    assert time_bucket_key(datetime(2026, 7, 28, 8, 0, 0), 15) == "2026-07-28 08:00"


def test_time_bucket_key_rejects_non_positive_minutes():
    with pytest.raises(ValueError):
        time_bucket_key(T0, 0)


def test_build_class_counts_counts_unique_tracks_only():
    states = [
        _state(1, "Car", "indeterminate", T0),
        _state(2, "Car", "indeterminate", T0),
        _state(3, "Bus", "indeterminate", T0),
    ]
    counts = build_class_counts(states)
    assert counts["Car"] == 2
    assert counts["Bus"] == 1
    assert counts["Pedestrian"] == 0  # present, zero, never omitted


def test_build_direction_counts_per_class_totals():
    states = [
        _state(1, "Car", "toward_junction", T0),
        _state(2, "Car", "toward_junction", T0),
        _state(3, "Car", "left_to_right", T0),
    ]
    counts = build_direction_counts(states)
    assert counts["Car"]["toward_junction"] == 2
    assert counts["Car"]["left_to_right"] == 1
    assert counts["Car"]["total"] == 3
    assert "Bus" not in counts  # no counted Bus tracks -> no row


def test_build_direction_totals_sums_across_all_classes():
    states = [
        _state(1, "Car", "toward_junction", T0),
        _state(2, "Bus", "toward_junction", T0),
        _state(3, "Pedestrian", "indeterminate", T0),
    ]
    totals = build_direction_totals(states)
    assert totals["toward_junction"] == 2
    assert totals["indeterminate"] == 1


def test_build_time_buckets_groups_by_entered_roi_time():
    states = [
        _state(1, "Car", "indeterminate", datetime(2026, 7, 28, 8, 10)),
        _state(2, "Car", "indeterminate", datetime(2026, 7, 28, 8, 50)),
        _state(3, "Bus", "indeterminate", datetime(2026, 7, 28, 9, 5)),
    ]
    buckets = build_time_buckets(states, minutes=60)
    assert buckets["2026-07-28 08:00"]["Car"] == 2
    assert buckets["2026-07-28 08:00"]["total"] == 2
    assert buckets["2026-07-28 09:00"]["Bus"] == 1
