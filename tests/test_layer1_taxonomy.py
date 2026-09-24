"""Tests for scripts/traffic_layer1/taxonomy.py -- class mapping load,
unmapped-class fallback, and coverage reporting. No GPU/model dependencies.
"""

import pytest

from config import ConfidenceConfig
from layer1_models import SourcedDetection
from taxonomy import BROAD_CLASSES, ClassMapping, load_class_mapping, tag_and_filter

MAPPING_YAML = """
uvh26:
  Hatchback: Car
  Two-wheeler: 2-Wheeler
  Bus: Bus
coco:
  person: Pedestrian
pending_model_support:
  - "Emergency Vehicle"
  - "Tractor"
"""


def _det(source, native_class, conf=0.9, bbox=(0.0, 0.0, 10.0, 10.0), track_id=1):
    return SourcedDetection(source=source, native_class=native_class, bbox=bbox, conf=conf, raw_track_id=track_id)


def test_load_class_mapping_parses_sections(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text(MAPPING_YAML)
    mapping = load_class_mapping(path)

    assert mapping.by_source["uvh26"]["Hatchback"] == "Car"
    assert mapping.by_source["coco"]["person"] == "Pedestrian"
    assert mapping.pending_model_support == ["Emergency Vehicle", "Tractor"]


def test_load_class_mapping_rejects_bucket_outside_taxonomy(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text("uvh26:\n  Hatchback: NotARealBucket\n")
    with pytest.raises(ValueError):
        load_class_mapping(path)


def test_map_class_known_and_unknown(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text(MAPPING_YAML)
    mapping = load_class_mapping(path)

    assert mapping.map_class("uvh26", "Bus") == "Bus"
    # A native class with no mapping entry falls back to "Others", not an error.
    assert mapping.map_class("uvh26", "Truck") == "Others"


def test_coverage_report_lists_detected_and_pending(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text(MAPPING_YAML)
    mapping = load_class_mapping(path)
    report = mapping.coverage_report()

    assert set(report["detected_buckets"]) == {"Car", "2-Wheeler", "Bus", "Pedestrian"}
    assert report["pending_model_support"] == ["Emergency Vehicle", "Tractor"]
    assert set(BROAD_CLASSES) >= set(report["detected_buckets"])


def test_tag_and_filter_sets_bucket_and_drops_below_threshold(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text(MAPPING_YAML)
    mapping = load_class_mapping(path)
    confidence = ConfidenceConfig(default=0.30, per_bucket={"Bus": 0.80})

    dets = [
        _det("uvh26", "Hatchback", conf=0.5),  # Car, passes default 0.30
        _det("uvh26", "Bus", conf=0.5),  # Bus, below per-bucket 0.80 -> dropped
        _det("uvh26", "Bus", conf=0.9),  # Bus, passes per-bucket 0.80
    ]
    kept = tag_and_filter(dets, mapping, confidence)

    assert [d.bucket for d in kept] == ["Car", "Bus"]
    assert [d.conf for d in kept] == [0.5, 0.9]
