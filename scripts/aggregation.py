"""Track-level demographic aggregation.

A track may generate dozens of face samples across its lifetime. This module
turns that list of per-frame FaceSample attempts into exactly ONE
TrackDemographicRecord per track_id -- never one record per frame. Gender is
resolved via a quality/confidence-weighted majority vote; age via a
quality/confidence-weighted median (robust to a handful of bad frames, unlike
a plain mean).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from config_loader import DemographicsConfig
from models import (
    STATUS_INSUFFICIENT_QUALITY,
    STATUS_NOT_SAMPLED,
    STATUS_NO_FACE_DETECTED,
    STATUS_VALID,
    FaceSample,
    TrackDemographicRecord,
    age_to_group,
)
from utils import weighted_majority_vote, weighted_median


class TrackDemographicAggregator:
    """Accumulates FaceSamples per track_id and produces final aggregated
    demographic records.
    """

    def __init__(self, config: DemographicsConfig):
        self.config = config
        self.samples: Dict[int, List[FaceSample]] = defaultdict(list)

    def add_sample(self, track_id: int, sample: FaceSample) -> None:
        self.samples[track_id].append(sample)

    def has_samples(self, track_id: int) -> bool:
        return track_id in self.samples and len(self.samples[track_id]) > 0

    def finalize_track(
        self, track_id: int, first_seen: str, last_seen: str
    ) -> TrackDemographicRecord:
        """Aggregate all samples collected for ``track_id`` into a single
        TrackDemographicRecord. Safe to call even if no samples were ever
        collected (e.g. a track that was counted but disappeared before any
        demographic sampling occurred) -- returns a "not_sampled" record.
        """
        track_samples = self.samples.get(track_id, [])
        total_attempts = len(track_samples)

        if total_attempts == 0:
            return TrackDemographicRecord(
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                gender="unknown",
                gender_confidence=None,
                age_estimate=None,
                age_group="unknown",
                age_confidence=None,
                valid_face_samples=0,
                total_face_attempts=0,
                average_face_quality=0.0,
                status=STATUS_NOT_SAMPLED,
            )

        average_quality = sum(s.quality_score for s in track_samples) / total_attempts
        usable_samples = [s for s in track_samples if s.usable]
        valid_count = len(usable_samples)

        if valid_count == 0:
            any_face_detected = any(s.face_detected for s in track_samples)
            status = STATUS_INSUFFICIENT_QUALITY if any_face_detected else STATUS_NO_FACE_DETECTED
            return TrackDemographicRecord(
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                gender="unknown",
                gender_confidence=None,
                age_estimate=None,
                age_group="unknown",
                age_confidence=None,
                valid_face_samples=0,
                total_face_attempts=total_attempts,
                average_face_quality=average_quality,
                status=status,
            )

        if valid_count < self.config.min_valid_samples:
            return TrackDemographicRecord(
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                gender="unknown",
                gender_confidence=None,
                age_estimate=None,
                age_group="unknown",
                age_confidence=None,
                valid_face_samples=valid_count,
                total_face_attempts=total_attempts,
                average_face_quality=average_quality,
                status=STATUS_INSUFFICIENT_QUALITY,
            )

        gender_labels = [s.gender for s in usable_samples if s.gender is not None]
        gender_weights = [
            (s.confidence_proxy if s.confidence_proxy is not None else s.quality_score)
            for s in usable_samples
            if s.gender is not None
        ]
        gender, gender_share = weighted_majority_vote(gender_labels, gender_weights)
        gender = gender or "unknown"

        # Numeric age (and the mean confidence over the samples that produced
        # it) only ever comes from face samples -- the body estimator never
        # emits a numeric age, only a bucket (see models.FaceSample.age_group
        # docstring). age_estimate is therefore None whenever a track's only
        # usable samples are body-sourced -- this is intentional honesty,
        # not a bug: we do not fabricate a numeric age from a bucket
        # midpoint.
        age_values = [s.age for s in usable_samples if s.age is not None]
        age_weights = [
            (s.confidence_proxy if s.confidence_proxy is not None else s.quality_score)
            for s in usable_samples
            if s.age is not None
        ]
        age_estimate = weighted_median(age_values, age_weights)
        numeric_age_confidence = (sum(age_weights) / len(age_weights)) if age_weights else None

        # The bucket itself is decided by a confidence-weighted majority
        # vote across EVERY usable sample regardless of source -- a face
        # sample votes age_to_group(its numeric age), a body sample votes
        # its own directly-predicted bucket. This is the same voting
        # mechanism already used for gender, generalized so heterogeneous
        # sources (numeric-age face samples, bucket-only body samples) can
        # be combined without conflating their different confidence scales
        # into a fake shared numeric age.
        age_group_labels = []
        age_group_weights = []
        for s in usable_samples:
            label = s.age_group if s.age_group is not None else age_to_group(s.age)
            if label in (None, "unknown"):
                continue
            age_group_labels.append(label)
            age_group_weights.append(
                s.confidence_proxy if s.confidence_proxy is not None else s.quality_score
            )
        age_group, age_group_share = weighted_majority_vote(age_group_labels, age_group_weights)
        age_group = age_group or "unknown"

        age_confidence = numeric_age_confidence if numeric_age_confidence is not None else (
            age_group_share if age_group_labels else None
        )

        return TrackDemographicRecord(
            track_id=track_id,
            first_seen=first_seen,
            last_seen=last_seen,
            gender=gender,
            gender_confidence=gender_share if gender_labels else None,
            age_estimate=age_estimate,
            age_group=age_group,
            age_confidence=age_confidence,
            valid_face_samples=valid_count,
            total_face_attempts=total_attempts,
            average_face_quality=average_quality,
            status=STATUS_VALID,
        )
