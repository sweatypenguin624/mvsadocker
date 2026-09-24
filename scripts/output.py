"""Output writing: annotated video, CSVs, run summary, track position log.

Nothing here persists face images or embeddings by default -- only
aggregate/track-level numeric and text attributes ever reach CSV. Debug face
crops are strictly opt-in (config.output.save_debug_face_crops) and capped
(max_debug_crops), written to a clearly separate debug_faces/ subdirectory
so they're never mistaken for a normal pipeline output.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config_loader import Config
from models import TRACK_DEMOGRAPHICS_CSV_HEADER, BBox, TrackDemographicRecord
from roi import PedestrianROI

logger = logging.getLogger("mvsa")

HOURLY_DEMOGRAPHICS_HEADER = [
    "hour",
    "pedestrian_entries",
    "male",
    "female",
    "unknown_gender",
    "age_0_18",
    "age_18_60",
    "age_60_plus",
    "unknown_age",
]

HOURLY_COUNTS_HEADER = ["hour", "pedestrian_entries"]

# Order matches movement_direction.ALL_LABELS minus INDETERMINATE at the
# front-of-mind positions -- kept explicit here (rather than imported) so
# output.py doesn't need to depend on movement_direction.py just to know a
# column order; pipeline.py is the module that actually imports both.
HOURLY_DIRECTION_COUNTS_HEADER = [
    "hour",
    "toward_junction",
    "away_from_junction",
    "left_to_right",
    "right_to_left",
    "indeterminate",
    "total",
]

TRACKS_CSV_HEADER = [
    "track_id",
    "first_seen_frame",
    "first_seen_time",
    "last_seen_frame",
    "last_seen_time",
    "entered_roi",
    "entered_roi_frame",
    "rider_status",
    "movement_direction",
    "movement_displacement_px",
]


class OutputWriter:
    """Owns all filesystem writes for one pipeline run."""

    def __init__(self, run_dir: Path, config: Config):
        self.run_dir = Path(run_dir)
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._debug_crop_count = 0
        if config.output.save_debug_face_crops:
            (self.run_dir / "debug_faces").mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    @property
    def video_path(self) -> Path:
        return self.run_dir / "tracked.mp4"

    @property
    def hourly_counts_path(self) -> Path:
        return self.run_dir / "hourly_pedestrian_counts.csv"

    @property
    def tracks_path(self) -> Path:
        return self.run_dir / "tracks.csv"

    @property
    def track_demographics_path(self) -> Path:
        return self.run_dir / "track_demographics.csv"

    @property
    def hourly_demographics_path(self) -> Path:
        return self.run_dir / "hourly_demographics.csv"

    @property
    def hourly_direction_counts_path(self) -> Path:
        return self.run_dir / "hourly_pedestrian_direction_counts.csv"

    @property
    def run_summary_path(self) -> Path:
        return self.run_dir / "run_summary.json"

    @property
    def track_positions_path(self) -> Path:
        return self.run_dir / "track_positions.json"

    @property
    def classification_summary_path(self) -> Path:
        return self.run_dir / "classification_summary.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "logs" / self.config.logging.filename

    # -- video writer ------------------------------------------------------

    def open_video_writer(self, width: int, height: int, fps: float) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
        writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.video_path}")
        return writer

    # -- CSVs --------------------------------------------------------------

    def write_hourly_counts_csv(self, hourly_counts: Dict[str, int]) -> None:
        with open(self.hourly_counts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HOURLY_COUNTS_HEADER)
            for hour in sorted(hourly_counts):
                writer.writerow([hour, hourly_counts[hour]])
        logger.info("Wrote %s", self.hourly_counts_path)

    def write_tracks_csv(self, rows: List[dict]) -> None:
        with open(self.tracks_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(TRACKS_CSV_HEADER)
            for row in rows:
                writer.writerow([row.get(col, "") for col in TRACKS_CSV_HEADER])
        logger.info("Wrote %s", self.tracks_path)

    def write_track_demographics_csv(self, records: List[TrackDemographicRecord]) -> None:
        with open(self.track_demographics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(TRACK_DEMOGRAPHICS_CSV_HEADER)
            for record in records:
                writer.writerow(record.to_row())
        logger.info("Wrote %s", self.track_demographics_path)

    def write_hourly_demographics_csv(self, hourly_demo: Dict[str, dict]) -> None:
        with open(self.hourly_demographics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HOURLY_DEMOGRAPHICS_HEADER)
            for hour in sorted(hourly_demo):
                row = hourly_demo[hour]
                writer.writerow(
                    [
                        hour,
                        row.get("pedestrian_entries", 0),
                        row.get("male", 0),
                        row.get("female", 0),
                        row.get("unknown_gender", 0),
                        row.get("age_0_18", 0),
                        row.get("age_18_60", 0),
                        row.get("age_60_plus", 0),
                        row.get("unknown_age", 0),
                    ]
                )
        logger.info("Wrote %s", self.hourly_demographics_path)

    def write_hourly_direction_counts_csv(self, hourly_directions: Dict[str, Dict[str, int]]) -> None:
        """hourly_directions: {hour: {label: count}} for
        movement_direction.ALL_LABELS. Only written when the run's camera
        has a movement_axis calibration (see pipeline.py) -- absent
        otherwise, same opt-in convention as hourly_demographics.csv.
        """
        with open(self.hourly_direction_counts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HOURLY_DIRECTION_COUNTS_HEADER)
            for hour in sorted(hourly_directions):
                counts = hourly_directions[hour]
                row_counts = [counts.get(label, 0) for label in HOURLY_DIRECTION_COUNTS_HEADER[1:-1]]
                writer.writerow([hour] + row_counts + [sum(row_counts)])
        logger.info("Wrote %s", self.hourly_direction_counts_path)

    def write_run_summary(self, summary: dict) -> None:
        with open(self.run_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote %s", self.run_summary_path)

    def write_classification_summary(self, summary: dict) -> None:
        with open(self.classification_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote %s", self.classification_summary_path)

    def write_track_positions(self, positions: Dict[int, List[dict]]) -> None:
        """Persist per-track (frame_idx, bbox) history so `demographics`
        mode can re-derive person crops from the original video without any
        images ever having been saved to disk.
        """
        serializable = {str(k): v for k, v in positions.items()}
        with open(self.track_positions_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
        logger.info("Wrote %s", self.track_positions_path)

    def load_track_positions(self) -> Dict[int, List[dict]]:
        with open(self.track_positions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    # -- debug crops (privacy: OFF by default) ------------------------------

    def maybe_save_debug_crop(self, track_id: int, frame_idx: int, crop: np.ndarray) -> None:
        if not self.config.output.save_debug_face_crops:
            return
        if self._debug_crop_count >= self.config.output.max_debug_crops:
            return
        out_path = self.run_dir / "debug_faces" / f"track{track_id}_frame{frame_idx}.jpg"
        cv2.imwrite(str(out_path), crop)
        self._debug_crop_count += 1


# -- frame annotation (pure function, no filesystem access) ------------------


def draw_annotations(
    frame: np.ndarray,
    detections: List,
    roi: PedestrianROI,
    total_count: int,
    timestamp: datetime,
    track_demographics: Dict[int, dict],
    show_demographics_min_quality: float,
    confident_count: int,
    riders_count: Optional[int] = None,
    uncertain_count: Optional[int] = None,
) -> np.ndarray:
    """Draw ROI polygon, person boxes + track IDs, HUD text, and (when
    quality clears the configured threshold) demographic labels.

    ``track_demographics`` holds the latest *usable* estimate per track,
    normalized to plain dicts ({"gender", "age_group", "quality", "usable"})
    so this function doesn't care whether a given track's label came from
    the face estimator (demographics.py) or the body/clothing estimator
    (body_attributes.py) -- see pipeline.py::_latest_estimate_dict.
    """
    h, w = frame.shape[:2]
    polygon = np.array(roi.scaled_polygon(w, h), dtype=np.int32)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    for det in detections:
        if det.track_id is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        label_lines = [f"ID {det.track_id}"]
        estimate = track_demographics.get(det.track_id)
        if estimate is not None and estimate.get("usable") and estimate.get("quality", 0.0) >= show_demographics_min_quality:
            if estimate.get("gender") != "unknown":
                label_lines.append(estimate["gender"])
            if estimate.get("age_group") != "unknown":
                label_lines.append(estimate["age_group"])

        for i, line in enumerate(label_lines):
            cv2.putText(
                frame,
                line,
                (x1, max(0, y1 - 10 - 18 * (len(label_lines) - 1 - i))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2,
            )

    # Bottom-right stats block on a translucent panel. Deliberately NOT
    # either top corner -- this dataset's source cameras burn their own
    # timestamp/site-label overlay into the top-left (all cameras) and, on
    # some (e.g. KLE Cam 1), a second large date/time stamp into the
    # top-right too, so any top-corner placement collides with at least one
    # camera's watermark. The translucent background additionally keeps text
    # legible over a bright sky or cluttered background regardless of
    # overlay collisions.
    #
    # riders_count/uncertain_count are None when rider_filter is disabled --
    # in that case there's nothing to break out, so only the total is shown
    # (matching total_count == confident_count in that case).
    lines = [timestamp.strftime("%Y-%m-%d %H:%M:%S")]
    lines.append(f"Total people: {total_count}")
    if riders_count is not None:
        lines.append(f"Riders: {riders_count}")
    if uncertain_count is not None:
        lines.append(f"Uncertain: {uncertain_count}")
    if riders_count is not None or uncertain_count is not None:
        lines.append(f"Confident pedestrians: {confident_count}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_height = 30
    margin = 15

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    block_w = max(size[0] for size in sizes) + 2 * margin
    block_h = len(lines) * line_height + margin

    box_x2, box_y2 = w - 5, h - 5
    box_x1, box_y1 = box_x2 - block_w, box_y2 - block_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = box_y1 + margin + 15
    for i, line in enumerate(lines):
        text_w = sizes[i][0]
        x = box_x2 - margin - text_w
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness)
        y += line_height

    return frame
