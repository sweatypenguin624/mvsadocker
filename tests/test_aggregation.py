"""Tests for scripts/aggregation.py (track-level demographic aggregation)
and the weighted-stats helpers in scripts/utils.py. No GPU/model
dependencies required -- FaceSample objects are constructed directly.
"""

from aggregation import TrackDemographicAggregator
from config_loader import DemographicsConfig
from models import (
    SOURCE_BODY,
    SOURCE_FACE,
    STATUS_INSUFFICIENT_QUALITY,
    STATUS_NOT_SAMPLED,
    STATUS_NO_FACE_DETECTED,
    STATUS_VALID,
    FaceSample,
)
from utils import weighted_majority_vote, weighted_median


def make_config(min_valid_samples=1) -> DemographicsConfig:
    return DemographicsConfig(
        enabled=True,
        sample_interval=10,
        insightface_model_name="buffalo_l",
        insightface_root="models/insightface",
        device="cpu",
        det_size=[640, 640],
        min_valid_samples=min_valid_samples,
    )


def usable_sample(frame_idx, gender, age, quality=0.8, det_score=0.9):
    return FaceSample(
        frame_idx=frame_idx,
        face_detected=True,
        usable=True,
        gender=gender,
        age=age,
        quality_score=quality,
        det_score=det_score,
        confidence_proxy=0.5 * quality + 0.5 * det_score,
    )


def unusable_sample(frame_idx, face_detected=True, quality=0.1):
    return FaceSample(
        frame_idx=frame_idx,
        face_detected=face_detected,
        usable=False,
        gender=None,
        age=None,
        quality_score=quality,
        det_score=0.3,
        confidence_proxy=None,
    )


def body_sample(frame_idx, gender, age_group, confidence=0.8):
    return FaceSample(
        frame_idx=frame_idx,
        face_detected=True,
        usable=True,
        gender=gender,
        age=None,
        quality_score=confidence,
        det_score=0.0,
        confidence_proxy=confidence,
        source=SOURCE_BODY,
        age_group=age_group,
    )


# -- weighted stats helpers --------------------------------------------------


def test_weighted_median_basic():
    assert weighted_median([10, 20, 30], [1, 1, 1]) == 20


def test_weighted_median_favors_heavier_weight():
    # Two low-weight outliers should not move the median away from the
    # single, heavily-weighted central value.
    result = weighted_median([5, 25, 26], [0.01, 10.0, 0.01])
    assert result == 25


def test_weighted_median_empty_returns_none():
    assert weighted_median([], []) is None


def test_weighted_majority_vote_picks_higher_weight():
    label, share = weighted_majority_vote(["male", "female"], [0.9, 0.2])
    assert label == "male"
    assert 0.8 < share <= 1.0


def test_weighted_majority_vote_empty_returns_none():
    label, share = weighted_majority_vote([], [])
    assert label is None
    assert share == 0.0


# -- TrackDemographicAggregator ----------------------------------------------


def test_no_samples_at_all_is_not_sampled():
    agg = TrackDemographicAggregator(make_config())
    record = agg.finalize_track(1, "2026-07-28 08:00:00", "2026-07-28 08:00:05")
    assert record.status == STATUS_NOT_SAMPLED
    assert record.gender == "unknown"
    assert record.age_group == "unknown"
    assert record.total_face_attempts == 0
    assert record.valid_face_samples == 0


def test_all_samples_unusable_with_face_detected_is_insufficient_quality():
    agg = TrackDemographicAggregator(make_config())
    agg.add_sample(2, unusable_sample(10, face_detected=True))
    agg.add_sample(2, unusable_sample(20, face_detected=True))
    record = agg.finalize_track(2, "t0", "t1")
    assert record.status == STATUS_INSUFFICIENT_QUALITY
    assert record.gender == "unknown"
    assert record.total_face_attempts == 2
    assert record.valid_face_samples == 0


def test_no_face_ever_detected_is_no_face_detected_status():
    agg = TrackDemographicAggregator(make_config())
    agg.add_sample(3, unusable_sample(10, face_detected=False))
    record = agg.finalize_track(3, "t0", "t1")
    assert record.status == STATUS_NO_FACE_DETECTED


def test_gender_majority_vote_weighted_by_confidence():
    agg = TrackDemographicAggregator(make_config())
    # Three "female" votes at low confidence, one "male" vote at very high
    # confidence -- majority-by-count would say female, weighted vote must
    # say male.
    agg.add_sample(4, usable_sample(1, "female", 30, quality=0.2, det_score=0.2))
    agg.add_sample(4, usable_sample(2, "female", 30, quality=0.2, det_score=0.2))
    agg.add_sample(4, usable_sample(3, "female", 30, quality=0.2, det_score=0.2))
    agg.add_sample(4, usable_sample(4, "male", 30, quality=0.99, det_score=0.99))
    record = agg.finalize_track(4, "t0", "t1")
    assert record.gender == "male"
    assert record.status == STATUS_VALID


def test_age_uses_weighted_median_not_mean():
    agg = TrackDemographicAggregator(make_config())
    agg.add_sample(5, usable_sample(1, "male", 20, quality=0.9, det_score=0.9))
    agg.add_sample(5, usable_sample(2, "male", 22, quality=0.9, det_score=0.9))
    agg.add_sample(5, usable_sample(3, "male", 90, quality=0.9, det_score=0.9))  # outlier
    record = agg.finalize_track(5, "t0", "t1")
    # Mean would be ~44; median-based aggregation should stay close to the
    # cluster of consistent estimates instead of being dragged by the outlier.
    assert record.age_estimate is not None
    assert record.age_estimate < 30


def test_one_record_per_track_not_per_sample():
    agg = TrackDemographicAggregator(make_config())
    for i in range(20):
        agg.add_sample(6, usable_sample(i, "male", 25))
    record = agg.finalize_track(6, "t0", "t1")
    assert record.track_id == 6
    assert record.valid_face_samples == 20
    # finalize_track always returns exactly one record, regardless of
    # how many samples were accumulated.
    assert record.age_estimate == 25


def test_min_valid_samples_threshold_forces_unknown():
    agg = TrackDemographicAggregator(make_config(min_valid_samples=5))
    agg.add_sample(7, usable_sample(1, "male", 25))
    record = agg.finalize_track(7, "t0", "t1")
    assert record.status == STATUS_INSUFFICIENT_QUALITY
    assert record.gender == "unknown"
    assert record.age_group == "unknown"


def test_mixed_usable_and_unusable_samples_only_counts_usable():
    agg = TrackDemographicAggregator(make_config())
    agg.add_sample(8, usable_sample(1, "female", 40))
    agg.add_sample(8, unusable_sample(2))
    agg.add_sample(8, unusable_sample(3))
    record = agg.finalize_track(8, "t0", "t1")
    assert record.total_face_attempts == 3
    assert record.valid_face_samples == 1
    assert record.gender == "female"


# -- multi-source (face + body) aggregation ----------------------------------


def test_body_only_track_gets_bucket_but_no_numeric_age():
    # No face ever resolved for this track -- only body-attribute samples.
    # age_group must still come out populated (voted from the body samples'
    # own buckets), but age_estimate must stay None: MVSA never fabricates
    # a numeric age from a bucket.
    agg = TrackDemographicAggregator(make_config())
    agg.add_sample(9, body_sample(1, "male", "18-60", confidence=0.9))
    agg.add_sample(9, body_sample(2, "male", "18-60", confidence=0.7))
    record = agg.finalize_track(9, "t0", "t1")
    assert record.status == STATUS_VALID
    assert record.gender == "male"
    assert record.age_group == "18-60"
    assert record.age_estimate is None
    assert record.age_confidence is not None


def test_face_and_body_samples_combine_in_one_vote():
    agg = TrackDemographicAggregator(make_config())
    # A single low-confidence face sample plus several higher-confidence
    # body samples all agreeing on "female"/"0-18" should win the vote even
    # though the face sample alone would have said something else.
    agg.add_sample(10, usable_sample(1, "male", 10, quality=0.1, det_score=0.1))
    agg.add_sample(10, body_sample(2, "female", "0-18", confidence=0.9))
    agg.add_sample(10, body_sample(3, "female", "0-18", confidence=0.9))
    record = agg.finalize_track(10, "t0", "t1")
    assert record.gender == "female"
    assert record.age_group == "0-18"
    assert record.total_face_attempts == 3
    assert record.valid_face_samples == 3
