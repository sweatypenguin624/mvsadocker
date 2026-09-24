"""Tests for scripts/vehicle_counting/vehicle_classes.py -- the category ->
COCO class mapping for the vehicle-counting pilot. No GPU/model dependencies
required.
"""

from vehicle_classes import (
    PENDING_FINE_TUNE,
    REQUESTED_VEHICLE_LABELS,
    SUPPORTED_CLASSES,
    bucket_names,
    class_id_to_bucket,
    coverage_report,
    detector_class_ids,
)


def test_detector_class_ids_are_the_four_supported_coco_classes():
    # bicycle=1, car=2, motorcycle=3, bus=5 -- truck (7) is intentionally
    # excluded since "Goods Vehicle" was not mapped to it.
    assert detector_class_ids() == [1, 2, 3, 5]


def test_class_id_to_bucket_routes_every_supported_class():
    mapping = class_id_to_bucket()
    assert mapping[1] == "BICYCLE"
    assert mapping[2] == "CAR"
    assert mapping[3] == "MOTORCYCLE"
    assert mapping[5] == "BUS"
    assert 7 not in mapping


def test_bucket_names_match_supported_classes():
    assert bucket_names() == ["BICYCLE", "MOTORCYCLE", "CAR", "BUS"]


def test_bus_subtypes_all_collapse_into_one_bucket():
    bus = next(vc for vc in SUPPORTED_CLASSES if vc.name == "BUS")
    for subtype in ("School Bus", "Private Bus", "Mini Bus", "KSRTC Bus", "City Bus", "BRTS Bus"):
        assert subtype in bus.requested_labels


def test_taxi_and_evcar_fold_into_car():
    car = next(vc for vc in SUPPORTED_CLASSES if vc.name == "CAR")
    assert "TAXI" in car.requested_labels
    assert "evCAR" in car.requested_labels


def test_pending_fine_tune_labels_have_no_bucket():
    covered_labels = {label for vc in SUPPORTED_CLASSES for label in vc.requested_labels}
    for label in PENDING_FINE_TUNE:
        assert label not in covered_labels


def test_coverage_report_accounts_for_every_requested_label():
    report = coverage_report()
    status = report["requested_label_status"]
    assert set(status.keys()) == set(REQUESTED_VEHICLE_LABELS)
    for label in PENDING_FINE_TUNE:
        assert status[label] == "pending_fine_tune"
    assert status["CAR"] == "CAR"
    assert status["TAXI"] == "CAR"
    assert status["KSRTC Bus"] == "BUS"
    assert "MOTORCYCLE" in status["2 WHEELER"]
    assert "BICYCLE" in status["2 WHEELER"]
