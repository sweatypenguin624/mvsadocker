"""Track-level (not per-frame) classification via confidence-weighted
majority vote.

Generalizes the one working temporal-vote implementation already in the
repo -- scripts/vehicle_counting/Vclassification/tracker.py::TrackState.
get_stable_class (confidence-weighted argmax over a track's raw per-frame
class predictions) -- into a standalone function that votes over Layer 1's
*broad buckets* (post scripts/traffic_layer1/taxonomy.py mapping) instead of
raw model classes, since that duplicate-but-unused
scripts/vehicle_counting/Vclassification/classification/temporal_voting.py
was never wired into anything.
"""

from __future__ import annotations

from typing import Dict, Optional


def vote_class(class_votes: Dict[str, float]) -> Optional[str]:
    """Return the broad bucket with the highest summed confidence across a
    track's observation history, or None if the track has no observations
    at all (an edge case a caller should not normally hit, since a track
    only exists after at least one detection updated it).

    A track's final class stays stable even if one or two frames produced
    an incorrect per-frame classification, because a handful of stray votes
    rarely outweigh the sum accumulated by the track's true class -- the
    same rationale the spec's section 7 example gives for temporal
    aggregation over per-frame majority voting alone.
    """
    if not class_votes:
        return None
    return max(class_votes.items(), key=lambda kv: kv[1])[0]
