"""Layer 1 orchestration: wires MultiSourceDetector + Layer1Tracker +
taxonomy + classifier + direction + crops into one streaming frame loop,
then aggregates and writes every required output.

Deliberately streaming (one frame read -> processed -> discarded per
iteration, via scripts/video_utils.py::VideoStreamReader) so memory stays
flat regardless of video length, per the spec's performance requirements.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from roi import roi_from_config  # noqa: E402
from utils import ensure_dir, setup_logger, verify_core_dependencies  # noqa: E402
from video_utils import (  # noqa: E402
    VideoStreamReader,
    frame_idx_to_timestamp,
    parse_start_datetime,
    probe_video,
)
from movement_direction import movement_axis_from_config  # noqa: E402

import aggregator  # noqa: E402
import visualize  # noqa: E402
from config import Layer1Config  # noqa: E402
from crops import CropManager  # noqa: E402
from layer1_detector import MultiSourceDetector  # noqa: E402
from direction import resolve_direction  # noqa: E402
from layer1_output import Layer1OutputWriter  # noqa: E402
from taxonomy import load_class_mapping, tag_and_filter  # noqa: E402
from layer1_tracker import Layer1Tracker  # noqa: E402

logger = logging.getLogger("mvsa.traffic_layer1")


class Layer1Pipeline:
    def __init__(
        self,
        config: Layer1Config,
        run_name: str,
        movement_axis_entry: Optional[dict] = None,
        debug: Optional[bool] = None,
        save_crops: Optional[bool] = None,
    ):
        self.config = config
        self.run_name = run_name

        self.run_dir = ensure_dir(config.resolve(config.output.results_dir) / run_name)
        self.output = Layer1OutputWriter(self.run_dir, config)
        self.logger = setup_logger(self.output.log_path, level=config.logging.level, name="mvsa.traffic_layer1")
        verify_core_dependencies(self.logger)

        self.class_mapping = load_class_mapping(config.resolve(config.class_mapping_path))
        self.roi = roi_from_config(config.roi)
        self.movement_axis = movement_axis_from_config(movement_axis_entry) if movement_axis_entry else None

        self.debug = config.output.debug_mode if debug is None else debug
        crops_cfg = dataclasses.replace(config.crops, enabled=config.crops.enabled if save_crops is None else save_crops)
        self.crop_manager = CropManager(crops_cfg)

    def _resolve_video_path(self) -> Path:
        src = self.config.resolve(self.config.video.path)
        if not src.exists():
            raise FileNotFoundError(f"video file not found: {src}")
        return src

    def run(self) -> dict:
        cfg = self.config
        video_path = self._resolve_video_path()
        video_info = probe_video(video_path, cfg.resolve(cfg.tools.ffprobe))
        start_dt = parse_start_datetime(cfg.video.start_datetime)

        if video_info.frame_count == 0:
            self.logger.warning("Video %s reports 0 frames -- proceeding, expect an empty summary", video_path)

        if video_info.fps and abs(video_info.fps - cfg.video.fps) > 0.5:
            self.logger.warning(
                "layer1_config.yaml video.fps=%.2f but ffprobe measured %.2f for %s -- using config value "
                "for timestamp math.",
                cfg.video.fps,
                video_info.fps,
                video_path,
            )

        self.logger.info(
            "Starting Layer 1 run_name=%s video=%s (%dx%d, ~%d frames, %.1f fps)",
            self.run_name,
            video_path,
            video_info.width,
            video_info.height,
            video_info.frame_count,
            cfg.video.fps,
        )

        detector = MultiSourceDetector(cfg)
        tracker = Layer1Tracker(self.roi, video_info.width, video_info.height)

        writing_video = self.debug or cfg.output.save_annotated_video
        video_writer = (
            self.output.open_video_writer(video_info.width, video_info.height, cfg.video.fps)
            if writing_video
            else None
        )

        running_counts: Dict[str, int] = {}
        total_raw_detections = 0
        start_time = time.time()
        frame_idx = -1

        with VideoStreamReader(video_path) as reader:
            for frame_idx, frame in reader.frames(start_frame=0):
                timestamp = frame_idx_to_timestamp(frame_idx, cfg.video.fps, start_dt)

                raw_detections = detector.detect(frame)
                total_raw_detections += len(raw_detections)
                tagged = tag_and_filter(raw_detections, self.class_mapping, cfg.confidence)

                newly_counted = tracker.update(frame_idx, timestamp, tagged)

                for det in tagged:
                    state = tracker.tracks.get((det.source, det.raw_track_id))
                    if state is not None:
                        self.crop_manager.consider(state.track_id, det, frame_idx, frame)

                for state in newly_counted:
                    state.direction_label = resolve_direction(
                        list(state.center_history),
                        self.movement_axis,
                        video_info.width,
                        video_info.height,
                        cfg.direction.min_displacement_fraction_of_frame_diagonal,
                    )
                    running_counts[state.final_class] = running_counts.get(state.final_class, 0) + 1

                if video_writer is not None:
                    live_tracks = [
                        (det, tracker.tracks[(det.source, det.raw_track_id)])
                        for det in tagged
                        if (det.source, det.raw_track_id) in tracker.tracks
                    ]
                    annotated = visualize.draw_debug_frame(
                        frame.copy(), live_tracks, self.roi, running_counts, timestamp
                    )
                    video_writer.write(annotated)

                if frame_idx > 0 and frame_idx % cfg.logging.progress_interval_frames == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0.0
                    self.logger.info(
                        "Progress: frame %d/%d, tracks=%d, counted=%d, fps=%.1f",
                        frame_idx,
                        video_info.frame_count,
                        len(tracker.tracks),
                        sum(running_counts.values()),
                        fps_actual,
                    )

        if video_writer is not None:
            video_writer.release()

        elapsed_total = time.time() - start_time
        frames_processed = frame_idx + 1
        self.logger.info(
            "Finished: %d frames processed in %.1fs, counted=%s",
            frames_processed,
            elapsed_total,
            running_counts,
        )

        return self._finalize_and_write(
            tracker=tracker,
            video_path=video_path,
            video_info=video_info,
            frames_processed=frames_processed,
            total_raw_detections=total_raw_detections,
            elapsed_seconds=elapsed_total,
        )

    def _finalize_and_write(
        self,
        tracker: Layer1Tracker,
        video_path: Path,
        video_info,
        frames_processed: int,
        total_raw_detections: int,
        elapsed_seconds: float,
    ) -> dict:
        cfg = self.config
        all_states = list(tracker.tracks.values())
        counted_states = [s for s in all_states if s.counted]
        incomplete_states = [s for s in all_states if not s.counted]

        bus_written = self.crop_manager.write_all(self.run_dir, tracker.tracks)
        bus_crop_ids = set(bus_written.keys())

        class_counts = aggregator.build_class_counts(counted_states)
        direction_counts = aggregator.build_direction_counts(counted_states)
        direction_totals = aggregator.build_direction_totals(counted_states)
        time_buckets = aggregator.build_time_buckets(counted_states, cfg.aggregation.time_bucket_minutes)

        self.output.write_tracks_jsonl(counted_states, bus_crop_ids)
        self.output.write_counts_csv(class_counts)
        self.output.write_time_bucket_csv(time_buckets)
        self.output.write_direction_csv(direction_counts)
        self.output.write_config_json(dataclasses.asdict(cfg))

        duration_seconds = frames_processed / cfg.video.fps if cfg.video.fps else 0.0

        summary = {
            "run_name": self.run_name,
            "generated_at": datetime.now().isoformat(),
            "input": {
                "video": str(video_path),
                "duration": _format_duration(duration_seconds),
                "fps": cfg.video.fps,
                "frames_processed": frames_processed,
                "width": video_info.width,
                "height": video_info.height,
            },
            "detection": {
                "total_detections": total_raw_detections,
                "unique_tracks": len(all_states),
            },
            "class_counts": class_counts,
            "tracking": {
                "completed_tracks": len(counted_states),
                "incomplete_tracks": len(incomplete_states),
                "duplicate_count_prevented": True,
            },
            "direction_totals": direction_totals,
            "direction_by_class": direction_counts,
            "time_buckets": {
                "bucket_minutes": cfg.aggregation.time_bucket_minutes,
                "buckets": time_buckets,
            },
            "class_coverage": self.class_mapping.coverage_report(),
            "bus_crops": {"tracks_with_crops": len(bus_crop_ids)},
            "elapsed_seconds": elapsed_seconds,
        }
        self.output.write_summary(summary)
        print(aggregator.format_run_summary_text(summary))
        return summary


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
