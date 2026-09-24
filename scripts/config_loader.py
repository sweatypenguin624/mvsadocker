"""Loads config/config.yaml into typed, validated dataclasses.

This is the single place that reads YAML. Every other module receives an
already-parsed config object -- nothing else in the codebase should call
``yaml.safe_load`` directly, and no thresholds should be hard-coded
elsewhere (per the project requirement to centralize configuration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


class ConfigError(ValueError):
    """Raised when config.yaml is missing required fields or is invalid."""


@dataclass
class VideoConfig:
    path: str
    fps: float
    start_datetime: str


@dataclass
class ModelConfig:
    yolo_weights: str
    device: str
    confidence: float
    imgsz: int
    person_class_id: int
    tracker: str


@dataclass
class ROIConfig:
    polygon: List[List[float]]
    reference_width: int
    reference_height: int


@dataclass
class RiderFilterConfig:
    enabled: bool
    vehicle_class_ids: List[int]
    uncertain_overlap_ratio: float
    rider_overlap_ratio: float
    exclude_riders_from_count: bool
    # If true, tracks whose final rider_status == "uncertain" are ALSO
    # excluded from the pedestrian count (only "pedestrian"-classified
    # tracks count) -- distinct from exclude_riders_from_count, which only
    # ever excludes "rider". "uncertain" tracks are still reported (e.g. in
    # classification_summary.json) regardless of this flag; it only affects
    # what counts as a confirmed pedestrian entry.
    require_confident_pedestrian: bool
    # A track's final rider_status is a majority vote over every frame it
    # was seen (see aggregate_track_rider_status). That vote is unreliable
    # when a track has very few frames -- a single frame's overlap-ratio
    # measurement is all that decides it, with nothing to average out a
    # momentary miss (e.g. a rider's body occluding their own vehicle from
    # the detector at the instant the track was seen). Tracks with fewer
    # than this many frames get the stricter check in
    # short_track_max_clean_ratio applied before their "pedestrian" verdict
    # is trusted.
    min_frames_for_confident_pedestrian: int
    # For a track shorter than min_frames_for_confident_pedestrian, its
    # "pedestrian" verdict is only trusted as-is if every frame's vehicle
    # overlap ratio stayed at or below this value (i.e. no vehicle was ever
    # measurably close) -- otherwise it's downgraded to "uncertain" rather
    # than counted as a confident pedestrian. Must be <= uncertain_overlap_ratio.
    short_track_max_clean_ratio: float
    # Overlap with a vehicle that's been sitting within
    # parked_vehicle_max_drift_px of the same spot for at least this many
    # seconds no longer counts as rider evidence (see vehicle_motion.py) --
    # a bbox-overlap heuristic can't otherwise tell a pedestrian standing or
    # walking next to a parked two-wheeler from someone riding one. 0
    # disables this check entirely (the pre-existing behavior).
    parked_vehicle_stationary_seconds: float
    # Max pixel drift (in the video's native resolution) a vehicle's
    # detection center may wander while still counting as "the same spot"
    # for parked_vehicle_stationary_seconds purposes -- absorbs detection
    # jitter, not real movement.
    parked_vehicle_max_drift_px: float


@dataclass
class DemographicsConfig:
    enabled: bool
    sample_interval: int
    insightface_model_name: str
    insightface_root: str
    device: str
    det_size: List[int]
    min_valid_samples: int


@dataclass
class BodyAttributesConfig:
    enabled: bool
    model_path: str
    device: str
    input_width: int
    input_height: int
    gender_threshold: float
    min_crop_width: int
    min_crop_height: int


@dataclass
class FaceQualityConfig:
    min_face_width: int
    min_face_height: int
    min_face_confidence: float
    min_sharpness: float
    check_brightness: bool
    min_brightness: float
    max_brightness: float


@dataclass
class OutputConfig:
    results_dir: str
    save_annotated_video: bool
    save_debug_face_crops: bool
    max_debug_crops: int
    video_codec: str
    show_demographics_min_quality: float


@dataclass
class MovementConfig:
    # A track's net motion along its own best-fit direction (see
    # scripts/movement_direction.py) must clear this fraction of the
    # frame's diagonal before a direction is trusted -- below it, reported
    # "indeterminate" rather than guessed, since at that scale the motion
    # is dominated by detector/tracker jitter, not real walking. Expressed
    # as a fraction of frame diagonal (not raw pixels) so the same default
    # is meaningful across cameras of different resolution.
    min_displacement_fraction_of_frame_diagonal: float


@dataclass
class CheckpointConfig:
    interval_frames: int
    filename: str


@dataclass
class LoggingConfig:
    level: str
    filename: str


@dataclass
class DemographicsTestConfig:
    max_tracks: int
    tail_frames: int


@dataclass
class ToolsConfig:
    ffmpeg: str
    ffprobe: str


@dataclass
class Config:
    project_root: Path
    video: VideoConfig
    model: ModelConfig
    roi: ROIConfig
    rider_filter: RiderFilterConfig
    demographics: DemographicsConfig
    body_attributes: BodyAttributesConfig
    face_quality: FaceQualityConfig
    output: OutputConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    demographics_test: DemographicsTestConfig
    tools: ToolsConfig
    movement: MovementConfig
    # Per-camera movement-direction calibration (see scripts/movement_direction.py),
    # as the raw "movement_axis" dict from that camera's entry in
    # config/roi/roi_by_camera.yaml. None means direction classification is
    # off for this run (the default -- opt-in per camera, same as ROI).
    # Deliberately a raw dict rather than a MovementAxis here so this loader
    # doesn't need to import movement_direction.py; pipeline.py converts it
    # via movement_direction.movement_axis_from_config().
    movement_axis: Optional[dict] = None

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def _require(d: dict, key: str, section: str) -> object:
    if key not in d:
        raise ConfigError(f"config.yaml section '{section}' is missing required key '{key}'")
    return d[key]


def load_config(config_path: Path) -> Config:
    """Load and validate config.yaml.

    ``project_root`` is derived as the parent of the ``config/`` directory
    containing the file (i.e. for .../mvsa/config/config.yaml, project_root
    is .../mvsa). All relative paths in the YAML (video path, model
    weights, output dir, etc.) are resolved against project_root.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml did not parse into a mapping")

    project_root = config_path.parent.parent

    video_raw = _require(raw, "video", "root")
    video = VideoConfig(
        path=_require(video_raw, "path", "video"),
        fps=float(_require(video_raw, "fps", "video")),
        start_datetime=_require(video_raw, "start_datetime", "video"),
    )

    model_raw = _require(raw, "model", "root")
    model = ModelConfig(
        yolo_weights=_require(model_raw, "yolo_weights", "model"),
        device=str(_require(model_raw, "device", "model")),
        confidence=float(_require(model_raw, "confidence", "model")),
        imgsz=int(_require(model_raw, "imgsz", "model")),
        person_class_id=int(_require(model_raw, "person_class_id", "model")),
        tracker=_require(model_raw, "tracker", "model"),
    )

    roi_raw = _require(raw, "roi", "root")
    polygon = _require(roi_raw, "polygon", "roi")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ConfigError("roi.polygon must be a list of at least 3 [x, y] pairs")
    roi = ROIConfig(
        polygon=polygon,
        reference_width=int(roi_raw.get("reference_width", 1920)),
        reference_height=int(roi_raw.get("reference_height", 1080)),
    )

    rider_raw = raw.get("rider_filter", {})
    rider_filter = RiderFilterConfig(
        enabled=bool(rider_raw.get("enabled", False)),
        vehicle_class_ids=list(rider_raw.get("vehicle_class_ids", [1, 2, 3, 5, 7])),
        uncertain_overlap_ratio=float(rider_raw.get("uncertain_overlap_ratio", 0.15)),
        rider_overlap_ratio=float(rider_raw.get("rider_overlap_ratio", 0.4)),
        exclude_riders_from_count=bool(rider_raw.get("exclude_riders_from_count", False)),
        require_confident_pedestrian=bool(rider_raw.get("require_confident_pedestrian", False)),
        min_frames_for_confident_pedestrian=int(rider_raw.get("min_frames_for_confident_pedestrian", 1)),
        short_track_max_clean_ratio=float(rider_raw.get("short_track_max_clean_ratio", 0.0)),
        parked_vehicle_stationary_seconds=float(rider_raw.get("parked_vehicle_stationary_seconds", 0.0)),
        parked_vehicle_max_drift_px=float(rider_raw.get("parked_vehicle_max_drift_px", 20.0)),
    )

    demo_raw = raw.get("demographics", {})
    demographics = DemographicsConfig(
        enabled=bool(demo_raw.get("enabled", True)),
        sample_interval=int(demo_raw.get("sample_interval", 10)),
        insightface_model_name=str(demo_raw.get("insightface_model_name", "buffalo_l")),
        insightface_root=str(demo_raw.get("insightface_root", "models/insightface")),
        device=str(demo_raw.get("device", "cpu")),
        det_size=list(demo_raw.get("det_size", [640, 640])),
        min_valid_samples=int(demo_raw.get("min_valid_samples", 1)),
    )

    body_raw = raw.get("body_attributes", {})
    body_attributes = BodyAttributesConfig(
        enabled=bool(body_raw.get("enabled", True)),
        model_path=str(body_raw.get("model_path", "models/person_attribute/person_attribute.onnx")),
        device=str(body_raw.get("device", "cpu")),
        input_width=int(body_raw.get("input_width", 192)),
        input_height=int(body_raw.get("input_height", 256)),
        gender_threshold=float(body_raw.get("gender_threshold", 0.5)),
        min_crop_width=int(body_raw.get("min_crop_width", 15)),
        min_crop_height=int(body_raw.get("min_crop_height", 40)),
    )

    fq_raw = raw.get("face_quality", {})
    face_quality = FaceQualityConfig(
        min_face_width=int(fq_raw.get("min_face_width", 40)),
        min_face_height=int(fq_raw.get("min_face_height", 40)),
        min_face_confidence=float(fq_raw.get("min_face_confidence", 0.5)),
        min_sharpness=float(fq_raw.get("min_sharpness", 60.0)),
        check_brightness=bool(fq_raw.get("check_brightness", True)),
        min_brightness=float(fq_raw.get("min_brightness", 30.0)),
        max_brightness=float(fq_raw.get("max_brightness", 225.0)),
    )

    out_raw = raw.get("output", {})
    output = OutputConfig(
        results_dir=str(out_raw.get("results_dir", "results")),
        save_annotated_video=bool(out_raw.get("save_annotated_video", True)),
        save_debug_face_crops=bool(out_raw.get("save_debug_face_crops", False)),
        max_debug_crops=int(out_raw.get("max_debug_crops", 20)),
        video_codec=str(out_raw.get("video_codec", "mp4v")),
        show_demographics_min_quality=float(out_raw.get("show_demographics_min_quality", 0.4)),
    )

    ckpt_raw = raw.get("checkpoint", {})
    checkpoint = CheckpointConfig(
        interval_frames=int(ckpt_raw.get("interval_frames", 500)),
        filename=str(ckpt_raw.get("filename", "checkpoint.json")),
    )

    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=str(log_raw.get("level", "INFO")),
        filename=str(log_raw.get("filename", "run.log")),
    )

    dtest_raw = raw.get("demographics_test", {})
    demographics_test = DemographicsTestConfig(
        max_tracks=int(dtest_raw.get("max_tracks", 20)),
        tail_frames=int(dtest_raw.get("tail_frames", 300)),
    )

    tools_raw = raw.get("tools", {})
    tools = ToolsConfig(
        ffmpeg=str(tools_raw.get("ffmpeg", "tools/ffmpeg")),
        ffprobe=str(tools_raw.get("ffprobe", "tools/ffprobe")),
    )

    movement_raw = raw.get("movement", {})
    movement = MovementConfig(
        min_displacement_fraction_of_frame_diagonal=float(
            movement_raw.get("min_displacement_fraction_of_frame_diagonal", 0.015)
        ),
    )

    return Config(
        project_root=project_root,
        video=video,
        model=model,
        roi=roi,
        rider_filter=rider_filter,
        demographics=demographics,
        body_attributes=body_attributes,
        face_quality=face_quality,
        output=output,
        checkpoint=checkpoint,
        logging=logging_cfg,
        demographics_test=demographics_test,
        tools=tools,
        movement=movement,
    )
