"""Track-level temporal aggregation (spec section 14): combine one stage's
per-frame predictions for a track into a single label + confidence. One bad
frame must not override the whole track.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from layer2_models import StagePrediction


def aggregate_predictions(
    predictions: List[StagePrediction], frame_weights: List[float]
) -> Tuple[str, float, Dict[str, float]]:
    """Weighted average of each frame's full probability distribution
    (weight = that frame's crop quality score, so a sharp/large/confident
    crop counts more than a blurry/small one), then argmax -- a strict
    generalization of confidence-weighted majority voting, since it uses
    the full distribution rather than collapsing each frame to one label
    first.

    Returns (label, confidence, averaged_probs). Raises ValueError on empty
    input -- callers must handle "no usable frames" before calling this.
    """
    if not predictions:
        raise ValueError("aggregate_predictions called with no predictions")
    if len(predictions) != len(frame_weights):
        raise ValueError("predictions and frame_weights must be the same length")

    total_weight = sum(frame_weights) or 1.0
    all_labels = sorted({label for p in predictions for label in p.probs.keys()})

    averaged: Dict[str, float] = {label: 0.0 for label in all_labels}
    for pred, weight in zip(predictions, frame_weights):
        w = weight / total_weight
        for label, prob in pred.probs.items():
            averaged[label] += prob * w

    best_label = max(averaged.items(), key=lambda kv: kv[1])[0]
    return best_label, averaged[best_label], averaged
