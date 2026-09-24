"""Filesystem writer for one Layer 1 run: summary.json, tracks.jsonl,
counts.csv, config.json, hourly/time-bucket csv, direction csv.

Kept independent of scripts/output.py and scripts/vehicle_counting/
vehicle_output.py (each pipeline's own writer, same convention those two
already established) since none of those own Layer 1's tracks.jsonl / Bus
crop bookkeeping shape.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import cv2

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import Layer1Config  # noqa: E402
from layer1_models import CountedRecord, TrackState  # noqa: E402
from taxonomy import BROAD_CLASSES  # noqa: E402

logger = logging.getLogger("mvsa.traffic_layer1")


def to_counted_record(state: TrackState, has_bus_crops: bool) -> CountedRecord:
    return CountedRecord(
        track_id=state.track_id,
        source=state.source,
        cls=state.final_class,
        confidence=state.mean_confidence,
        direction=state.direction_label or "indeterminate",
        first_seen=state.first_seen_time,
        last_seen=state.last_seen_time,
        counted_at=state.entered_roi_time or state.last_seen_time,
        frames_seen=state.frames_seen,
        has_bus_crops=has_bus_crops,
    )


class Layer1OutputWriter:
    def __init__(self, run_dir: Path, config: Layer1Config):
        self.run_dir = Path(run_dir)
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    @property
    def video_path(self) -> Path:
        return self.run_dir / "tracked.mp4"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def tracks_jsonl_path(self) -> Path:
        return self.run_dir / "tracks.jsonl"

    @property
    def counts_csv_path(self) -> Path:
        return self.run_dir / "counts.csv"

    @property
    def hourly_csv_path(self) -> Path:
        return self.run_dir / "hourly_counts.csv"

    @property
    def direction_csv_path(self) -> Path:
        return self.run_dir / "direction_counts.csv"

    @property
    def config_json_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "logs" / self.config.logging.filename

    # -- video writer --------------------------------------------------

    def open_video_writer(self, width: int, height: int, fps: float) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
        writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.video_path}")
        return writer

    # -- writes -----------------------------------------------------------

    def write_summary(self, summary: dict) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote %s", self.summary_path)

    def write_tracks_jsonl(self, counted: List[TrackState], bus_crop_track_ids: set) -> None:
        with open(self.tracks_jsonl_path, "w", encoding="utf-8") as f:
            for state in counted:
                record = to_counted_record(state, has_bus_crops=state.track_id in bus_crop_track_ids)
                f.write(json.dumps(record.to_dict()) + "\n")
        logger.info("Wrote %s", self.tracks_jsonl_path)

    def write_counts_csv(self, class_counts: Dict[str, int]) -> None:
        with open(self.counts_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count"])
            for cls in BROAD_CLASSES:
                writer.writerow([cls, class_counts.get(cls, 0)])
            writer.writerow(["TOTAL", sum(class_counts.values())])
        logger.info("Wrote %s", self.counts_csv_path)

    def write_time_bucket_csv(self, buckets: Dict[str, Dict[str, int]]) -> None:
        header = ["bucket"] + BROAD_CLASSES + ["total"]
        with open(self.hourly_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for key in sorted(buckets):
                row = buckets[key]
                writer.writerow([key] + [row.get(c, 0) for c in BROAD_CLASSES] + [row.get("total", 0)])
        logger.info("Wrote %s", self.hourly_csv_path)

    def write_direction_csv(self, direction_counts: Dict[str, Dict[str, int]]) -> None:
        from movement_direction import ALL_LABELS  # local import: keeps output.py importable without cv2-heavy deps at module load if ever reused standalone

        header = ["class"] + list(ALL_LABELS) + ["total"]
        with open(self.direction_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for cls, row in direction_counts.items():
                writer.writerow([cls] + [row.get(label, 0) for label in ALL_LABELS] + [row.get("total", 0)])
        logger.info("Wrote %s", self.direction_csv_path)

    def write_config_json(self, config_dict: dict) -> None:
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info("Wrote %s", self.config_json_path)
