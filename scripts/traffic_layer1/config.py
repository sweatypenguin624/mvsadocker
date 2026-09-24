"""Loads config/layer1_config.yaml (+ config/layer1_class_mapping.yaml) into
typed, validated dataclasses.

Deliberately its own file/schema, same convention scripts/config_loader.py
and scripts/vehicle_counting/vehicle_config_loader.py already established:
each pipeline owns its own config surface so none can accidentally drift
another's.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class Layer1ConfigError(ValueError):
    """Raised when layer1_config.yaml is missing required fields or invalid."""


@dataclass
class VideoConfig:
    path: str
    fps: float
    start_datetime: str


@dataclass
class DetectorModelConfig:
    yolo_weights: str
    device: str
    imgsz: int
    tracker: str
    # Restrict this detector instance to a fixed set of native class ids.
    # None means "ask for every class the model knows" (used for UVH-26).
    class_ids: Optional[List[int]] = None
    # NMS tuning -- see scripts/detector.py::PersonDetector for why these
    # exist. Defaults here are lower/more aggressive than Ultralytics' own
    # (iou=0.7, agnostic_nms=False) because UVH-26 was observed emitting
    # multiple overlapping boxes (same class tiled along a large/close
    # vehicle, and different classes on the same object) that the stock
    # defaults don't suppress.
    iou: float = 0.45
    agnostic_nms: bool = True


@dataclass
class ModelsConfig:
    uvh26: DetectorModelConfig
    person: DetectorModelConfig


@dataclass
class ConfidenceConfig:
    default: float
    per_bucket: Dict[str, float] = field(default_factory=dict)

    def threshold_for(self, bucket: str) -> float:
        return self.per_bucket.get(bucket, self.default)


@dataclass
class ROIConfig:
    polygon: List[List[float]]
    reference_width: int
    reference_height: int
    roi_by_camera_path: str


@dataclass
class DirectionConfig:
    min_displacement_fraction_of_frame_diagonal: float


@dataclass
class AggregationConfig:
    time_bucket_minutes: int


@dataclass
class CropsConfig:
    enabled: bool
    max_per_track: int
    target_classes: List[str]


@dataclass
class OutputConfig:
    results_dir: str
    save_annotated_video: bool
    debug_mode: bool
    video_codec: str


@dataclass
class LoggingConfig:
    level: str
    filename: str
    progress_interval_frames: int


@dataclass
class ToolsConfig:
    ffmpeg: str
    ffprobe: str


@dataclass
class Layer1Config:
    project_root: Path
    video: VideoConfig
    models: ModelsConfig
    confidence: ConfidenceConfig
    class_mapping_path: str
    roi: ROIConfig
    direction: DirectionConfig
    aggregation: AggregationConfig
    crops: CropsConfig
    output: OutputConfig
    logging: LoggingConfig
    tools: ToolsConfig

    def resolve(self, relative_path: str) -> Path:
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def _require(d: dict, key: str, section: str) -> object:
    if key not in d:
        raise Layer1ConfigError(f"layer1_config.yaml section '{section}' is missing required key '{key}'")
    return d[key]


def _load_detector_model(raw: dict, section: str) -> DetectorModelConfig:
    class_id = raw.get("class_id")
    class_ids = [int(class_id)] if class_id is not None else None
    return DetectorModelConfig(
        yolo_weights=str(_require(raw, "yolo_weights", section)),
        device=str(_require(raw, "device", section)),
        imgsz=int(_require(raw, "imgsz", section)),
        tracker=str(raw.get("tracker", "bytetrack.yaml")),
        class_ids=class_ids,
        iou=float(raw.get("iou", 0.45)),
        agnostic_nms=bool(raw.get("agnostic_nms", True)),
    )


def load_layer1_config(config_path: Path) -> Layer1Config:
    """Load and validate layer1_config.yaml.

    ``project_root`` is derived as the parent of the ``config/`` directory
    containing the file, same convention as the other config loaders.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise Layer1ConfigError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise Layer1ConfigError("layer1_config.yaml did not parse into a mapping")

    project_root = config_path.parent.parent

    video_raw = _require(raw, "video", "root")
    video = VideoConfig(
        path=_require(video_raw, "path", "video"),
        fps=float(_require(video_raw, "fps", "video")),
        start_datetime=_require(video_raw, "start_datetime", "video"),
    )

    models_raw = _require(raw, "models", "root")
    models = ModelsConfig(
        uvh26=_load_detector_model(_require(models_raw, "uvh26", "models"), "models.uvh26"),
        person=_load_detector_model(_require(models_raw, "person", "models"), "models.person"),
    )

    conf_raw = raw.get("confidence", {})
    confidence = ConfidenceConfig(
        default=float(conf_raw.get("default", 0.30)),
        per_bucket={k: float(v) for k, v in (conf_raw.get("per_bucket") or {}).items()},
    )

    class_mapping_path = str(raw.get("class_mapping_path", "config/layer1_class_mapping.yaml"))

    roi_raw = _require(raw, "roi", "root")
    polygon = _require(roi_raw, "polygon", "roi")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise Layer1ConfigError("roi.polygon must be a list of at least 3 [x, y] pairs")
    roi = ROIConfig(
        polygon=polygon,
        reference_width=int(roi_raw.get("reference_width", 1920)),
        reference_height=int(roi_raw.get("reference_height", 1080)),
        roi_by_camera_path=str(roi_raw.get("roi_by_camera_path", "config/roi/roi_by_camera.yaml")),
    )

    dir_raw = raw.get("direction", {})
    direction = DirectionConfig(
        min_displacement_fraction_of_frame_diagonal=float(
            dir_raw.get("min_displacement_fraction_of_frame_diagonal", 0.015)
        )
    )

    agg_raw = raw.get("aggregation", {})
    aggregation = AggregationConfig(time_bucket_minutes=int(agg_raw.get("time_bucket_minutes", 60)))

    crops_raw = raw.get("crops", {})
    crops = CropsConfig(
        enabled=bool(crops_raw.get("enabled", False)),
        max_per_track=int(crops_raw.get("max_per_track", 5)),
        target_classes=list(crops_raw.get("target_classes", ["Bus"])),
    )

    out_raw = raw.get("output", {})
    output = OutputConfig(
        results_dir=str(out_raw.get("results_dir", "results")),
        save_annotated_video=bool(out_raw.get("save_annotated_video", False)),
        debug_mode=bool(out_raw.get("debug_mode", False)),
        video_codec=str(out_raw.get("video_codec", "mp4v")),
    )

    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=str(log_raw.get("level", "INFO")),
        filename=str(log_raw.get("filename", "run.log")),
        progress_interval_frames=int(log_raw.get("progress_interval_frames", 2000)),
    )

    tools_raw = raw.get("tools", {})
    tools = ToolsConfig(
        ffmpeg=str(tools_raw.get("ffmpeg", "tools/ffmpeg")),
        ffprobe=str(tools_raw.get("ffprobe", "tools/ffprobe")),
    )

    return Layer1Config(
        project_root=project_root,
        video=video,
        models=models,
        confidence=confidence,
        class_mapping_path=class_mapping_path,
        roi=roi,
        direction=direction,
        aggregation=aggregation,
        crops=crops,
        output=output,
        logging=logging_cfg,
        tools=tools,
    )
