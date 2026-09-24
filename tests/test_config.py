"""Tests for scripts/config_loader.py. No GPU/model dependencies required."""

from pathlib import Path

import pytest
import yaml

from config_loader import ConfigError, load_config

MINIMAL_YAML = {
    "video": {"path": "videos/x.mp4", "fps": 20, "start_datetime": "2026-07-28 08:00:00"},
    "model": {
        "yolo_weights": "models/yolo11x.pt",
        "device": "0",
        "confidence": 0.35,
        "imgsz": 1280,
        "person_class_id": 0,
        "tracker": "bytetrack.yaml",
    },
    "roi": {
        "reference_width": 1920,
        "reference_height": 1080,
        "polygon": [[0, 0], [10, 0], [10, 10]],
    },
}


def _write_config(tmp_path: Path, data: dict) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f)
    return config_path


def test_load_minimal_config_applies_defaults(tmp_path):
    config_path = _write_config(tmp_path, MINIMAL_YAML)
    config = load_config(config_path)

    assert config.video.fps == 20
    assert config.model.confidence == 0.35
    assert len(config.roi.polygon) == 3
    # Defaults for sections omitted entirely from the YAML.
    assert config.rider_filter.enabled is False
    assert config.demographics.sample_interval == 10
    assert config.face_quality.min_face_width == 40
    assert config.checkpoint.interval_frames == 500


def test_project_root_is_parent_of_config_dir(tmp_path):
    config_path = _write_config(tmp_path, MINIMAL_YAML)
    config = load_config(config_path)
    assert config.project_root == tmp_path


def test_resolve_relative_path_against_project_root(tmp_path):
    config_path = _write_config(tmp_path, MINIMAL_YAML)
    config = load_config(config_path)
    resolved = config.resolve("models/yolo11x.pt")
    assert resolved == (tmp_path / "models" / "yolo11x.pt").resolve()


def test_resolve_absolute_path_passthrough(tmp_path):
    config_path = _write_config(tmp_path, MINIMAL_YAML)
    config = load_config(config_path)
    abs_path = Path("/some/absolute/path.mp4")
    assert config.resolve(str(abs_path)) == abs_path


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_missing_required_section_raises(tmp_path):
    data = dict(MINIMAL_YAML)
    del data["video"]
    config_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_required_key_within_section_raises(tmp_path):
    import copy

    data = copy.deepcopy(MINIMAL_YAML)
    del data["model"]["confidence"]
    config_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_roi_polygon_too_short_raises(tmp_path):
    import copy

    data = copy.deepcopy(MINIMAL_YAML)
    data["roi"]["polygon"] = [[0, 0], [1, 1]]
    config_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_rider_filter_overrides_are_respected(tmp_path):
    import copy

    data = copy.deepcopy(MINIMAL_YAML)
    data["rider_filter"] = {"enabled": True, "rider_overlap_ratio": 0.6}
    config_path = _write_config(tmp_path, data)
    config = load_config(config_path)
    assert config.rider_filter.enabled is True
    assert config.rider_filter.rider_overlap_ratio == 0.6
    # Untouched keys in the same section still fall back to defaults.
    assert config.rider_filter.uncertain_overlap_ratio == 0.15
