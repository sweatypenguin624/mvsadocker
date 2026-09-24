"""Tests for scripts/traffic_layer1/classifier.py::vote_class -- track-level
confidence-weighted majority vote over broad-bucket class history.
"""

from classifier import vote_class


def test_vote_class_single_class():
    assert vote_class({"Car": 3.5}) == "Car"


def test_vote_class_picks_highest_summed_confidence_not_highest_frame_count():
    # "2-Wheeler" seen more often but with low confidence each time; "Bus"
    # seen fewer times but with high confidence -- summed confidence should
    # still favor whichever bucket accumulates more total, matching the
    # spec's "stable even if one or two frames misclassify" requirement.
    votes = {"2-Wheeler": 0.2 + 0.2 + 0.2, "Bus": 0.9 + 0.9}
    assert vote_class(votes) == "Bus"


def test_vote_class_minority_misclassified_frames_do_not_flip_result():
    votes = {"Car": 0.81 + 0.87 + 0.91 + 0.89 + 0.94, "Others": 0.4}
    assert vote_class(votes) == "Car"


def test_vote_class_empty_history_returns_none():
    assert vote_class({}) is None
