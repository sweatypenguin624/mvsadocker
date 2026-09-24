"""Production run configuration: YAML-driven, describes *what* to process
(a folder of videos, or a single video -- e.g. one 15-minute clip for a
quick test run), which detector/tracker/device to use, and how to name
cameras. Distinct from PipelineConfig (config.py), which holds the
detection/tracking/classification *algorithm* knobs (RunConfig is "what to
run", PipelineConfig is "how").

See production_config.example.yaml for a documented example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    mode: str                                  # "folder" or "single_video"
    input_folder: Path | None = None            # required if mode == "folder"
    input_video: Path | None = None             # required if mode == "single_video"
    output_dir: Path = Path("results/bus_color/production_run")
    camera_name: str | None = None              # None => derive per-video (see video_discovery.py)
    video_globs: list = field(default_factory=lambda: ["*.mp4", "*.dav", "*.avi", "*.mkv"])
    detector_model: str = "yolo26s.pt"
    tracker_config: str = "botsort.yaml"
    device: str | None = None                    # None => auto (cuda:0 if available else cpu)
    detection_confidence: float | None = None     # None => PipelineConfig default
    interval_minutes: int | None = None            # None => PipelineConfig default (15)
    imgsz: int | None = None                       # None => ultralytics default (640). Set this to the
                                                     # resolution the detector was trained at (the UVH-26
                                                     # fine-tune uses 960) -- a mismatch costs accuracy.
    save_best_frame_crops: bool | None = None       # None => PipelineConfig default (on). Turn off for a
                                                     # counts-only run -- no per-track image files written.
    video_start_times: dict = field(default_factory=dict)  # filename -> ISO datetime override
    start_offset_seconds: float = 0.0              # skip this many seconds into each video before processing
    duration_seconds: float | None = None          # None => process to the end; else stop after this many
                                                     # seconds of video (from start_offset_seconds). Lets a
                                                     # quick test slice run straight off the source video --
                                                     # no need to pre-cut/re-encode a separate clip file.

    @staticmethod
    def from_yaml(path: Path) -> "RunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        mode = raw.get("mode")
        if mode not in ("folder", "single_video"):
            raise ValueError(f"config 'mode' must be 'folder' or 'single_video', got {mode!r}")

        cfg = RunConfig(
            mode=mode,
            input_folder=Path(raw["input_folder"]) if raw.get("input_folder") else None,
            input_video=Path(raw["input_video"]) if raw.get("input_video") else None,
            output_dir=Path(raw.get("output_dir", "results/bus_color/production_run")),
            camera_name=raw.get("camera_name"),
            video_globs=raw.get("video_globs", ["*.mp4", "*.dav", "*.avi", "*.mkv"]),
            detector_model=raw.get("detector_model", "yolo26s.pt"),
            tracker_config=raw.get("tracker_config", "botsort.yaml"),
            device=raw.get("device"),
            detection_confidence=raw.get("detection_confidence"),
            interval_minutes=raw.get("interval_minutes"),
            imgsz=raw.get("imgsz"),
            save_best_frame_crops=raw.get("save_best_frame_crops"),
            video_start_times=raw.get("video_start_times", {}),
            start_offset_seconds=raw.get("start_offset_seconds", 0.0),
            duration_seconds=raw.get("duration_seconds"),
        )
        if cfg.mode == "folder" and cfg.input_folder is None:
            raise ValueError("mode='folder' requires 'input_folder' in the config")
        if cfg.mode == "single_video" and cfg.input_video is None:
            raise ValueError("mode='single_video' requires 'input_video' in the config")
        return cfg
