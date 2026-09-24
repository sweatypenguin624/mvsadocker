"""Pipeline orchestration for all four CLI modes: detect, full,
demographics, demographics-test.

This module is the only place that wires together detector.py, tracker.py,
rider_filter.py, demographics.py, aggregation.py, checkpoint.py and
output.py into an actual frame loop. Everything else in scripts/ is a
narrow, independently-testable component; this file is intentionally the
"integration" layer.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from aggregation import TrackDemographicAggregator
from body_attributes import BodyAttributeAnalyzer, BodyAttributeEstimate
from checkpoint import CheckpointManager
from config_loader import Config
from demographics import DemographicAnalyzer
from detector import PersonDetector
from face_quality import FaceQualityAssessor
from models import BBox, Detection, FaceSample, SOURCE_BODY, TrackDemographicRecord
from output import OutputWriter, draw_annotations
from rider_filter import RiderFilter, aggregate_track_rider_status
from movement_direction import ALL_LABELS, DirectionResult, classify_direction, movement_axis_from_config
from roi import PedestrianROI, bottom_center, roi_from_config
from tracker import PedestrianTracker, TrackState
from vehicle_motion import VehicleMotionTracker
from utils import ensure_dir, setup_logger, verify_core_dependencies
from video_utils import (
    VideoInfo,
    VideoStreamReader,
    frame_idx_to_timestamp,
    hour_bucket,
    parse_start_datetime,
    probe_video,
    remux_dav_to_mp4,
)
import cv2

MODE_DETECT = "detect"
MODE_DEMOGRAPHICS = "demographics"
MODE_FULL = "full"
MODE_DEMOGRAPHICS_TEST = "demographics-test"

PROGRESS_LOG_INTERVAL_FRAMES = 2000


def _crop_person(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]
def _crop_demographic_region(
    frame: np.ndarray,
    bbox: BBox,
    scale: float = 2.0,
    upper_fraction: float = 0.65,
    pad_x: float = 0.15,
    pad_y: float = 0.10,
) -> np.ndarray:
    """
    Prepare a person crop specifically for face/demographic analysis.

    The demographic model does not need the full body. We retain the upper
    portion of the person bounding box, add a little horizontal/vertical
    padding, then upscale it for the face detector.
    """
    h, w = frame.shape[:2]

    bx1, by1, bx2, by2 = bbox

    person_w = max(1.0, bx2 - bx1)
    person_h = max(1.0, by2 - by1)

    # Focus on the upper body/head region.
    region_h = person_h * upper_fraction

    x1 = bx1 - person_w * pad_x
    x2 = bx2 + person_w * pad_x
    y1 = by1 - person_h * pad_y
    y2 = by1 + region_h

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=frame.dtype)

    crop = frame[y1:y2, x1:x2]

    if scale != 1.0:
        new_w = max(1, int(round(crop.shape[1] * scale)))
        new_h = max(1, int(round(crop.shape[0] * scale)))

        crop = cv2.resize(
            crop,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )

    return crop

class Pipeline:
    def __init__(self, config: Config, run_name: str, resume: bool = False):
        self.config = config
        self.run_name = run_name
        self.resume = resume

        self.run_dir = ensure_dir(config.resolve(config.output.results_dir) / run_name)
        self.output = OutputWriter(self.run_dir, config)
        self.logger = setup_logger(self.output.log_path, level=config.logging.level)
        verify_core_dependencies(self.logger)

        self.roi: PedestrianROI = roi_from_config(config.roi)

    # -- video resolution (handles .dav -> .mp4 remux transparently) --------

    def _resolve_video_path(self) -> Path:
        src = self.config.resolve(self.config.video.path)
        if src.suffix.lower() != ".dav":
            if not src.exists():
                raise FileNotFoundError(f"video file not found: {src}")
            return src

        mp4_path = src.with_suffix(".mp4")
        if not mp4_path.exists():
            self.logger.info("Remuxing DAV source %s -> %s (stream copy, no re-encode)", src, mp4_path)
            ffmpeg_bin = self.config.resolve(self.config.tools.ffmpeg)
            remux_dav_to_mp4(src, mp4_path, ffmpeg_bin)
        else:
            self.logger.info("Reusing existing remuxed MP4: %s", mp4_path)
        return mp4_path

    # -- shared setup ---------------------------------------------------------

    def _detector_classes(self) -> List[int]:
        classes = {self.config.model.person_class_id}
        if self.config.rider_filter.enabled:
            classes.update(self.config.rider_filter.vehicle_class_ids)
        return sorted(classes)

    def _resolve_tracker_config(self) -> str:
        """Resolve model.tracker against the project root if it points at a
        real file there (e.g. "config/tracking/pedestrian_bytetrack.yaml"),
        so it loads correctly regardless of the process's cwd. Bare
        Ultralytics-bundled names like "bytetrack.yaml"/"botsort.yaml" don't
        exist under the project root, so they fall through unresolved and
        Ultralytics' own lookup finds its packaged default -- unchanged
        behavior for configs that haven't opted into a custom tracker file.
        """
        raw = self.config.model.tracker
        resolved = self.config.resolve(raw)
        return str(resolved) if resolved.exists() else raw

    def _build_detector(self) -> PersonDetector:
        return PersonDetector(
            model_path=self.config.resolve(self.config.model.yolo_weights),
            device=self.config.model.device,
            confidence=self.config.model.confidence,
            imgsz=self.config.model.imgsz,
            classes=self._detector_classes(),
            tracker=self._resolve_tracker_config(),
        )

    def _build_demographic_analyzer(self) -> DemographicAnalyzer:
        quality_assessor = FaceQualityAssessor(self.config.face_quality)
        return DemographicAnalyzer(self.config.demographics, quality_assessor)

    def _build_body_analyzer(self) -> BodyAttributeAnalyzer:
        return BodyAttributeAnalyzer(self.config.body_attributes, self.config.project_root)

    @staticmethod
    def _latest_estimate_dict(gender: str, age_group: str, quality: float) -> dict:
        return {"gender": gender, "age_group": age_group, "quality": quality, "usable": True}

    def _load_checkpoint_if_resuming(
        self,
        checkpoint_mgr: CheckpointManager,
        tracker: PedestrianTracker,
        aggregator: Optional[TrackDemographicAggregator],
        vehicle_tracker: Optional[VehicleMotionTracker] = None,
    ) -> Tuple[int, Dict[int, List[dict]], int]:
        """Returns (resume_frame, track_positions, track_id_offset)."""
        if not self.resume or not checkpoint_mgr.exists():
            return 0, {}, 0

        state = checkpoint_mgr.load()
        tracks = {int(k): TrackState.from_dict(v) for k, v in state.get("track_states", {}).items()}
        counted_ids: Set[int] = set(state.get("counted_track_ids", []))
        tracker.restore(tracks, counted_ids)

        if vehicle_tracker is not None:
            vehicle_tracker.restore(state.get("vehicle_motion_states", {}))

        if aggregator is not None:
            for k, v in state.get("face_samples", {}).items():
                aggregator.samples[int(k)] = [FaceSample.from_dict(x) for x in v]

        track_positions = {int(k): v for k, v in state.get("track_positions", {}).items()}
        resume_frame = int(state["frame_idx"]) + 1
        id_offset = (max(tracks.keys()) + 1) if tracks else 0

        self.logger.warning(
            "Resuming from checkpoint at frame %d (offset new track IDs by %d to avoid "
            "colliding with pre-resume IDs; ByteTrack's internal counter restarts at 1 "
            "after a process restart). Annotated video writing is DISABLED for resumed "
            "runs because OpenCV's VideoWriter cannot append to an existing MP4.",
            resume_frame,
            id_offset,
        )
        return resume_frame, track_positions, id_offset

    def _checkpoint_state(
        self,
        frame_idx: int,
        mode: str,
        tracker: PedestrianTracker,
        aggregator: Optional[TrackDemographicAggregator],
        track_positions: Dict[int, List[dict]],
        vehicle_tracker: Optional[VehicleMotionTracker] = None,
    ) -> dict:
        return {
            "frame_idx": frame_idx,
            "mode": mode,
            "counted_track_ids": list(tracker.counted_ids),
            "track_states": {str(tid): st.to_dict() for tid, st in tracker.tracks.items()},
            "vehicle_motion_states": vehicle_tracker.to_checkpoint() if vehicle_tracker is not None else {},
            "face_samples": (
                {str(tid): [s.to_dict() for s in samples] for tid, samples in aggregator.samples.items()}
                if aggregator is not None
                else {}
            ),
            "track_positions": {str(tid): pos for tid, pos in track_positions.items()},
        }

    # -- core frame loop, shared by detect / full / demographics-test --------

    def _run_frame_loop(
        self,
        mode: str,
        run_demographics_inline: bool,
        max_tracks: Optional[int] = None,
    ) -> dict:
        cfg = self.config
        video_path = self._resolve_video_path()
        video_info = probe_video(video_path, cfg.resolve(cfg.tools.ffprobe))
        start_dt = parse_start_datetime(cfg.video.start_datetime)

        if video_info.fps and abs(video_info.fps - cfg.video.fps) > 0.5:
            self.logger.warning(
                "config.video.fps=%.2f but ffprobe measured %.2f for %s -- using config value "
                "for timestamp math; verify config.yaml if this is unexpected.",
                cfg.video.fps,
                video_info.fps,
                video_path,
            )

        self.logger.info(
            "Starting mode=%s run_name=%s video=%s (%dx%d, ~%d frames, %.1f fps)",
            mode,
            self.run_name,
            video_path,
            video_info.width,
            video_info.height,
            video_info.frame_count,
            cfg.video.fps,
        )

        detector = self._build_detector()
        tracker = PedestrianTracker(self.roi, video_info.width, video_info.height)
        rider_filter = RiderFilter(cfg.rider_filter) if cfg.rider_filter.enabled else None
        vehicle_tracker = (
            VehicleMotionTracker(
                parked_window_frames=max(1, round(cfg.rider_filter.parked_vehicle_stationary_seconds * cfg.video.fps)),
                max_drift_px=cfg.rider_filter.parked_vehicle_max_drift_px,
            )
            if rider_filter is not None and cfg.rider_filter.parked_vehicle_stationary_seconds > 0
            else None
        )
        aggregator = TrackDemographicAggregator(cfg.demographics) if run_demographics_inline else None
        demographic_analyzer = self._build_demographic_analyzer() if run_demographics_inline else None
        body_analyzer = (
            self._build_body_analyzer()
            if run_demographics_inline and cfg.body_attributes.enabled
            else None
        )

        checkpoint_mgr = CheckpointManager(self.run_dir / cfg.checkpoint.filename, cfg.checkpoint.interval_frames)
        resume_frame, track_positions, id_offset = self._load_checkpoint_if_resuming(
            checkpoint_mgr, tracker, aggregator, vehicle_tracker
        )
        writing_video = cfg.output.save_annotated_video and resume_frame == 0

        video_writer = None
        if writing_video:
            video_writer = self.output.open_video_writer(video_info.width, video_info.height, cfg.video.fps)

        latest_estimates: Dict[int, dict] = {}
        # Live, on-screen breakdown for the annotated video's HUD only -- NOT
        # used for any CSV/JSON output (those remain computed the existing
        # way, in _finalize_and_write below, via a majority vote over each
        # track's ENTIRE rider_statuses history -- which can only be known
        # once a track is finished being seen, i.e. never available while
        # still writing frames in real time). These live counters instead
        # classify a track exactly once, from the rider/vehicle overlap
        # classification at the single frame it first enters the ROI, and
        # never revisit that decision. They will occasionally disagree with
        # the final CSV numbers in edge cases (e.g. a rider dismounts moments
        # before crossing the ROI line) -- a deliberate, documented tradeoff
        # for a live preview, not a bug; the CSVs remain the source of truth.
        live_riders = 0
        live_uncertain = 0
        live_confident = 0
        reached_cap_frame: Optional[int] = None
        start_time = time.time()
        frame_idx = resume_frame

        with VideoStreamReader(video_path) as reader:
            for frame_idx, frame in reader.frames(start_frame=resume_frame):
                timestamp = frame_idx_to_timestamp(frame_idx, cfg.video.fps, start_dt)

                all_detections = detector.track_frame(frame)
                if id_offset:
                    for det in all_detections:
                        if det.track_id is not None:
                            det.track_id += id_offset

                person_dets = [d for d in all_detections if d.cls_id == cfg.model.person_class_id]

                newly_counted = tracker.update(frame_idx, timestamp, person_dets)

                if rider_filter is not None:
                    rider_filter_detections = all_detections
                    if vehicle_tracker is not None:
                        vehicle_dets = [d for d in all_detections if d.cls_id in cfg.rider_filter.vehicle_class_ids]
                        vehicle_tracker.update(frame_idx, vehicle_dets)
                        # A vehicle parked for the whole scene shouldn't read
                        # as rider evidence just because a pedestrian is
                        # standing or walking next to it -- drop parked
                        # vehicles from what classify() can match against.
                        rider_filter_detections = [
                            d
                            for d in all_detections
                            if d.cls_id not in cfg.rider_filter.vehicle_class_ids
                            or d.track_id is None
                            or not vehicle_tracker.is_parked(d.track_id, frame_idx)
                        ]
                    for det in person_dets:
                        if det.track_id is None:
                            continue
                        result = rider_filter.classify(det.bbox, rider_filter_detections)
                        tracker.tracks[det.track_id].rider_statuses.append(result.status)
                        tracker.tracks[det.track_id].rider_overlap_ratios.append(result.overlap_ratio)
                        if det.track_id in newly_counted:
                            if result.status == "rider":
                                live_riders += 1
                            elif result.status == "uncertain":
                                live_uncertain += 1
                            else:
                                live_confident += 1

                for det in person_dets:
                    if det.track_id is not None and det.track_id in tracker.counted_ids:
                        track_positions.setdefault(det.track_id, []).append(
                            {"frame_idx": frame_idx, "bbox": list(det.bbox)}
                        )

                if run_demographics_inline:
                    for det in person_dets:
                        tid = det.track_id
                        if tid is None or tid not in tracker.counted_ids:
                            continue
                        state = tracker.tracks[tid]
                        if state.frames_since_last_sample < cfg.demographics.sample_interval:
                            continue
                        state.frames_since_last_sample = 0
                        crop = _crop_demographic_region(
                            frame,
                            det.bbox,
                            scale=2.0,
                            upper_fraction=0.65,
                            pad_x=0.15,
                            pad_y=0.10,
                        )

                        estimate = demographic_analyzer.analyze(crop)
                        state.demographic_samples_taken += 1
                        sample = FaceSample(
                            frame_idx=frame_idx,
                            face_detected=estimate.face_detected,
                            usable=estimate.usable,
                            gender=estimate.gender if estimate.usable else None,
                            age=estimate.age if estimate.usable else None,
                            quality_score=estimate.face_quality,
                            det_score=estimate.det_score,
                            confidence_proxy=estimate.gender_confidence,
                        )
                        aggregator.add_sample(tid, sample)
                        self.output.maybe_save_debug_crop(tid, frame_idx, crop)

                        body_estimate: Optional[BodyAttributeEstimate] = None
                        if body_analyzer is not None:
                            body_crop = _crop_person(frame, det.bbox)
                            body_estimate = body_analyzer.analyze(body_crop)
                            body_sample = FaceSample(
                                frame_idx=frame_idx,
                                face_detected=body_estimate.usable,
                                usable=body_estimate.usable,
                                gender=body_estimate.gender if body_estimate.usable else None,
                                age=None,
                                quality_score=body_estimate.quality_score,
                                det_score=0.0,
                                confidence_proxy=body_estimate.confidence_proxy,
                                source=SOURCE_BODY,
                                age_group=body_estimate.age_group if body_estimate.usable else None,
                            )
                            aggregator.add_sample(tid, body_sample)

                        if estimate.usable:
                            latest_estimates[tid] = self._latest_estimate_dict(
                                estimate.gender, estimate.age_group, estimate.face_quality
                            )
                        elif body_estimate is not None and body_estimate.usable:
                            latest_estimates[tid] = self._latest_estimate_dict(
                                body_estimate.gender, body_estimate.age_group, body_estimate.quality_score
                            )

                if video_writer is not None:
                    annotated = draw_annotations(
                        frame.copy(),
                        person_dets,
                        self.roi,
                        len(tracker.counted_ids),
                        timestamp,
                        latest_estimates,
                        cfg.output.show_demographics_min_quality,
                        confident_count=live_confident,
                        riders_count=live_riders if rider_filter is not None else None,
                        uncertain_count=live_uncertain if rider_filter is not None else None,
                    )
                    video_writer.write(annotated)

                if checkpoint_mgr.should_save(frame_idx):
                    checkpoint_mgr.save(
                        self._checkpoint_state(frame_idx, mode, tracker, aggregator, track_positions, vehicle_tracker)
                    )

                if frame_idx > 0 and frame_idx % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(
                        "Progress: frame %d/%d, counted=%d, elapsed=%.1fs",
                        frame_idx,
                        video_info.frame_count,
                        len(tracker.counted_ids),
                        elapsed,
                    )

                if max_tracks is not None:
                    if reached_cap_frame is None and len(tracker.counted_ids) >= max_tracks:
                        reached_cap_frame = frame_idx
                        self.logger.info(
                            "demographics-test: reached %d counted tracks at frame %d, "
                            "processing %d more frames to let those tracks accumulate samples",
                            max_tracks,
                            frame_idx,
                            cfg.demographics_test.tail_frames,
                        )
                    if (
                        reached_cap_frame is not None
                        and frame_idx - reached_cap_frame >= cfg.demographics_test.tail_frames
                    ):
                        break

        if video_writer is not None:
            video_writer.release()

        checkpoint_mgr.save(
            self._checkpoint_state(frame_idx, mode, tracker, aggregator, track_positions, vehicle_tracker)
        )

        elapsed_total = time.time() - start_time
        self.logger.info(
            "Finished mode=%s: %d frames processed in %.1fs, %d pedestrians counted",
            mode,
            frame_idx - resume_frame + 1,
            elapsed_total,
            len(tracker.counted_ids),
        )

        return self._finalize_and_write(
            mode=mode,
            tracker=tracker,
            aggregator=aggregator,
            rider_filter_enabled=rider_filter is not None,
            track_positions=track_positions,
            video_path=video_path,
            video_info=video_info,
            elapsed_seconds=elapsed_total,
            start_dt=start_dt,
        )

    # -- finalization: rider filtering, demographics aggregation, all CSVs --

    def _finalize_and_write(
        self,
        mode: str,
        tracker: PedestrianTracker,
        aggregator: Optional[TrackDemographicAggregator],
        rider_filter_enabled: bool,
        track_positions: Dict[int, List[dict]],
        video_path: Path,
        video_info: VideoInfo,
        elapsed_seconds: float,
        start_dt: datetime,
    ) -> dict:
        cfg = self.config
        counted_states = [s for s in tracker.tracks.values() if s.counted]
        total_people = len(counted_states)

        rider_final: Dict[int, str] = {}
        if rider_filter_enabled:
            rider_final = {
                s.track_id: aggregate_track_rider_status(
                    s.rider_statuses,
                    s.rider_overlap_ratios,
                    min_frames_for_confident_pedestrian=cfg.rider_filter.min_frames_for_confident_pedestrian,
                    short_track_max_clean_ratio=cfg.rider_filter.short_track_max_clean_ratio,
                )
                for s in counted_states
            }
            excluded_statuses = set()
            if cfg.rider_filter.exclude_riders_from_count:
                excluded_statuses.add("rider")
            if cfg.rider_filter.require_confident_pedestrian:
                excluded_statuses.add("uncertain")
            if excluded_statuses:
                before = len(counted_states)
                counted_states = [s for s in counted_states if rider_final.get(s.track_id) not in excluded_statuses]
                self.logger.info(
                    "Rider filtering excluded %d of %d counted tracks from the pedestrian count (excluded statuses: %s)",
                    before - len(counted_states),
                    before,
                    sorted(excluded_statuses),
                )

        self.output.write_classification_summary(
            _build_classification_summary(
                run_name=self.run_name,
                total_people=total_people,
                rider_final=rider_final,
                rider_filter_enabled=rider_filter_enabled,
                exclude_riders_from_count=cfg.rider_filter.exclude_riders_from_count,
                require_confident_pedestrian=cfg.rider_filter.require_confident_pedestrian,
                final_pedestrian_count=len(counted_states),
            )
        )

        hourly_counts: Dict[str, int] = {}
        for s in counted_states:
            ts = frame_idx_to_timestamp(s.entered_roi_frame, cfg.video.fps, start_dt)
            hb = hour_bucket(ts)
            hourly_counts[hb] = hourly_counts.get(hb, 0) + 1
        self.output.write_hourly_counts_csv(hourly_counts)

        # Movement-direction classification -- opt-in per camera (only runs
        # when config/roi/roi_by_camera.yaml has a movement_axis entry for
        # this --camera-key). Scoped to counted_states, i.e. the FINAL
        # confident-pedestrian population (post rider-filtering) that also
        # drives hourly_pedestrian_counts.csv above: a rider's direction is
        # their vehicle's direction, not a pedestrian movement, so riders
        # and uncertain tracks are deliberately not classified here.
        movement_results: Dict[int, DirectionResult] = {}
        if cfg.movement_axis is not None:
            axis = movement_axis_from_config(cfg.movement_axis).scaled(video_info.width, video_info.height)
            min_displacement_px = cfg.movement.min_displacement_fraction_of_frame_diagonal * math.hypot(
                video_info.width, video_info.height
            )
            hourly_directions: Dict[str, Dict[str, int]] = {}
            for s in counted_states:
                points = [bottom_center(tuple(p["bbox"])) for p in track_positions.get(s.track_id, [])]
                result = classify_direction(points, axis, min_displacement_px=min_displacement_px)
                movement_results[s.track_id] = result

                ts = frame_idx_to_timestamp(s.entered_roi_frame, cfg.video.fps, start_dt)
                hb = hour_bucket(ts)
                bucket = hourly_directions.setdefault(hb, {})
                bucket[result.label] = bucket.get(result.label, 0) + 1
            self.output.write_hourly_direction_counts_csv(hourly_directions)
            self.logger.info(
                "Movement-direction breakdown (of %d confident pedestrians): %s",
                len(counted_states),
                {label: sum(1 for r in movement_results.values() if r.label == label) for label in ALL_LABELS},
            )

        tracks_rows = []
        for s in tracker.tracks.values():
            direction = movement_results.get(s.track_id)
            tracks_rows.append(
                {
                    "track_id": s.track_id,
                    "first_seen_frame": s.first_seen_frame,
                    "first_seen_time": s.first_seen_time.isoformat(sep=" "),
                    "last_seen_frame": s.last_seen_frame,
                    "last_seen_time": s.last_seen_time.isoformat(sep=" "),
                    "entered_roi": s.counted,
                    "entered_roi_frame": s.entered_roi_frame if s.entered_roi_frame is not None else "",
                    "rider_status": rider_final.get(s.track_id, ""),
                    "movement_direction": direction.label if direction is not None else "",
                    "movement_displacement_px": round(direction.displacement_px, 1) if direction is not None else "",
                }
            )
        self.output.write_tracks_csv(tracks_rows)
        self.output.write_track_positions(track_positions)

        demographic_records: List[TrackDemographicRecord] = []
        if aggregator is not None:
            for s in counted_states:
                record = aggregator.finalize_track(
                    s.track_id, s.first_seen_time.isoformat(sep=" "), s.last_seen_time.isoformat(sep=" ")
                )
                demographic_records.append(record)
            self.output.write_track_demographics_csv(demographic_records)

            hourly_demo = _compute_hourly_demographics(
                [(s.track_id, s.entered_roi_frame) for s in counted_states],
                demographic_records,
                cfg.video.fps,
                start_dt,
            )
            self.output.write_hourly_demographics_csv(hourly_demo)

        summary = {
            "run_name": self.run_name,
            "mode": mode,
            "video_path": str(video_path),
            "video_info": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": video_info.fps,
                "frame_count": video_info.frame_count,
            },
            "config_fps_used": cfg.video.fps,
            "start_datetime": cfg.video.start_datetime,
            "total_pedestrian_entries": len(counted_states),
            "total_tracks_observed": len(tracker.tracks),
            "rider_filtering_enabled": rider_filter_enabled,
            "demographics_enabled": aggregator is not None,
            "movement_direction_enabled": cfg.movement_axis is not None,
            "movement_direction_breakdown": (
                {label: sum(1 for r in movement_results.values() if r.label == label) for label in ALL_LABELS}
                if cfg.movement_axis is not None
                else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "generated_at": datetime.now().isoformat(),
        }
        self.output.write_run_summary(summary)
        return summary

    # -- public entry points --------------------------------------------------

    def run_detect(self) -> dict:
        return self._run_frame_loop(MODE_DETECT, run_demographics_inline=False)

    def run_full(self) -> dict:
        return self._run_frame_loop(
            MODE_FULL, run_demographics_inline=self.config.demographics.enabled
        )

    def run_demographics_test(self) -> dict:
        summary = self._run_frame_loop(
            MODE_DEMOGRAPHICS_TEST,
            run_demographics_inline=True,
            max_tracks=self.config.demographics_test.max_tracks,
        )
        self._write_demographic_test_report()
        return summary

    def run_demographics(self) -> dict:
        """Second-pass demographic analysis using bbox positions recorded by
        a prior `detect` or `full` run in the same run directory -- no
        re-running of YOLO, and no images were ever persisted to get here.
        """
        cfg = self.config
        if not self.output.track_positions_path.exists():
            raise FileNotFoundError(
                f"{self.output.track_positions_path} not found -- run `detect` or `full` "
                "for this run_name first."
            )
        if not self.output.tracks_path.exists():
            raise FileNotFoundError(
                f"{self.output.tracks_path} not found -- run `detect` or `full` for this run_name first."
            )

        positions = self.output.load_track_positions()
        tracks_meta = _read_tracks_csv(self.output.tracks_path)

        video_path = self._resolve_video_path()
        start_dt = parse_start_datetime(cfg.video.start_datetime)

        sample_targets: Dict[int, List[Tuple[int, BBox]]] = defaultdict(list)
        max_frame_idx = 0
        for tid, pos_list in positions.items():
            for i, p in enumerate(pos_list):
                if i % cfg.demographics.sample_interval == 0:
                    fidx = int(p["frame_idx"])
                    bbox = tuple(p["bbox"])
                    sample_targets[fidx].append((tid, bbox))
                    max_frame_idx = max(max_frame_idx, fidx)

        self.logger.info(
            "demographics mode: replaying %d sample frames for %d tracks from %s",
            len(sample_targets),
            len(positions),
            video_path,
        )

        aggregator = TrackDemographicAggregator(cfg.demographics)
        analyzer = self._build_demographic_analyzer()
        body_analyzer = self._build_body_analyzer() if cfg.body_attributes.enabled else None

        with VideoStreamReader(video_path) as reader:
            for frame_idx, frame in reader.frames(start_frame=0):
                targets = sample_targets.get(frame_idx)
                if targets:
                    for tid, bbox in targets:
                        crop = _crop_demographic_region(
                            frame,
                            bbox,
                            scale=2.0,
                            upper_fraction=0.65,
                            pad_x=0.15,
                            pad_y=0.10,
                        )
                        estimate = analyzer.analyze(crop)

                        sample = FaceSample(
                            frame_idx=frame_idx,
                            face_detected=estimate.face_detected,
                            usable=estimate.usable,
                            gender=estimate.gender if estimate.usable else None,
                            age=estimate.age if estimate.usable else None,
                            quality_score=estimate.face_quality,
                            det_score=estimate.det_score,
                            confidence_proxy=estimate.gender_confidence,
                        )
                        aggregator.add_sample(tid, sample)
                        self.output.maybe_save_debug_crop(tid, frame_idx, crop)

                        if body_analyzer is not None:
                            body_crop = _crop_person(frame, bbox)
                            body_estimate = body_analyzer.analyze(body_crop)
                            body_sample = FaceSample(
                                frame_idx=frame_idx,
                                face_detected=body_estimate.usable,
                                usable=body_estimate.usable,
                                gender=body_estimate.gender if body_estimate.usable else None,
                                age=None,
                                quality_score=body_estimate.quality_score,
                                det_score=0.0,
                                confidence_proxy=body_estimate.confidence_proxy,
                                source=SOURCE_BODY,
                                age_group=body_estimate.age_group if body_estimate.usable else None,
                            )
                            aggregator.add_sample(tid, body_sample)
                if frame_idx >= max_frame_idx:
                    break

        demographic_records = []
        entries = []
        for tid in positions.keys():
            meta = tracks_meta.get(tid)
            if meta is None:
                continue
            record = aggregator.finalize_track(tid, meta["first_seen_time"], meta["last_seen_time"])
            demographic_records.append(record)
            if meta["entered_roi_frame"] not in (None, ""):
                entries.append((tid, int(meta["entered_roi_frame"])))

        self.output.write_track_demographics_csv(demographic_records)
        hourly_demo = _compute_hourly_demographics(entries, demographic_records, cfg.video.fps, start_dt)
        self.output.write_hourly_demographics_csv(hourly_demo)

        summary = {
            "run_name": self.run_name,
            "mode": MODE_DEMOGRAPHICS,
            "video_path": str(video_path),
            "tracks_analyzed": len(demographic_records),
            "generated_at": datetime.now().isoformat(),
        }
        self.output.write_run_summary(summary)
        return summary

    def _write_demographic_test_report(self) -> None:
        """Detailed per-track debug report for demographics-test mode --
        the first thing to run after installing the demographic model, to
        judge whether this camera's footage supports age/gender estimation
        at all before scaling to full-video processing.
        """
        # Re-derive from the checkpoint we just saved, since it holds both
        # track_states and face_samples in one place. final_gender/final_age/
        # age_group/status are computed via the SAME TrackDemographicAggregator
        # used for the real track_demographics.csv (rather than a second,
        # hand-rolled vote here) so this debug report can never drift out of
        # sync with the authoritative output -- it only adds a raw,
        # per-source breakdown of what fed that vote.
        checkpoint_mgr = CheckpointManager(self.run_dir / self.config.checkpoint.filename, self.config.checkpoint.interval_frames)
        state = checkpoint_mgr.load()
        if state is None:
            self.logger.warning("No checkpoint found; cannot write demographic_test_report.csv")
            return

        rows = []
        header = [
            "track_id",
            "num_frames",
            "num_face_attempts",
            "num_face_usable",
            "face_gender_predictions",
            "face_age_predictions",
            "num_body_attempts",
            "num_body_usable",
            "body_gender_predictions",
            "body_age_group_predictions",
            "final_gender",
            "final_age",
            "final_age_group",
            "status",
            "quality_min",
            "quality_max",
            "quality_mean",
        ]

        track_states = state.get("track_states", {})
        raw_samples = state.get("face_samples", {})
        counted_ids = set(state.get("counted_track_ids", []))

        aggregator = TrackDemographicAggregator(self.config.demographics)
        for k, v in raw_samples.items():
            aggregator.samples[int(k)] = [FaceSample.from_dict(x) for x in v]

        for tid_str in sorted(counted_ids, key=lambda x: int(x) if isinstance(x, str) else x):
            tid = int(tid_str)
            samples = raw_samples.get(str(tid), [])
            ts = track_states.get(str(tid), {})

            face_samples = [s for s in samples if s.get("source", "face") == "face"]
            body_samples = [s for s in samples if s.get("source", "face") == "body"]
            face_usable = [s for s in face_samples if s["usable"]]
            body_usable = [s for s in body_samples if s["usable"]]
            qualities = [s["quality_score"] for s in samples]

            record = aggregator.finalize_track(tid, "", "")

            rows.append(
                [
                    tid,
                    ts.get("frames_seen", 0),
                    len(face_samples),
                    len(face_usable),
                    ";".join(s["gender"] for s in face_usable),
                    ";".join(str(s["age"]) for s in face_usable if s["age"] is not None),
                    len(body_samples),
                    len(body_usable),
                    ";".join(s["gender"] for s in body_usable),
                    ";".join(s.get("age_group") or "" for s in body_usable),
                    record.gender,
                    "" if record.age_estimate is None else round(record.age_estimate, 1),
                    record.age_group,
                    record.status,
                    round(min(qualities), 3) if qualities else "",
                    round(max(qualities), 3) if qualities else "",
                    round(sum(qualities) / len(qualities), 3) if qualities else "",
                ]
            )

        report_path = self.run_dir / "demographic_test_report.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        self.logger.info("Wrote %s (%d tracks)", report_path, len(rows))


def _build_classification_summary(
    run_name: str,
    total_people: int,
    rider_final: Dict[int, str],
    rider_filter_enabled: bool,
    exclude_riders_from_count: bool,
    require_confident_pedestrian: bool,
    final_pedestrian_count: int,
) -> dict:
    """Human-readable explanation of how total_people breaks down into
    riders/uncertain/confident-pedestrians -- the same breakdown shown live
    on the annotated video's HUD, but computed here from the FINAL,
    authoritative per-track classification (a majority vote over each
    track's entire rider_statuses history, from aggregate_track_rider_status)
    rather than the video's live, entry-frame-only approximation. This is
    the file to trust; the HUD is a preview.
    """

    def _pct(n: int) -> float:
        return round(100.0 * n / total_people, 1) if total_people else 0.0

    if not rider_filter_enabled:
        return {
            "run_name": run_name,
            "rider_filtering_enabled": False,
            "total_people": total_people,
            "final_pedestrian_count": final_pedestrian_count,
            "explanation": (
                "rider_filter.enabled was false for this run, so no rider/pedestrian "
                "classification was performed -- every track that entered the ROI counts "
                "as a pedestrian. final_pedestrian_count == total_people."
            ),
        }

    riders = sum(1 for v in rider_final.values() if v == "rider")
    uncertain = sum(1 for v in rider_final.values() if v == "uncertain")
    confident_pedestrians = sum(1 for v in rider_final.values() if v == "pedestrian")

    # Build the final_pedestrian_count formula/explanation from what's
    # actually excluded, rather than hard-coding one of a few fixed strings
    # -- exclude_riders_from_count and require_confident_pedestrian are
    # independent flags and all four combinations are valid.
    included_terms = ["confident_pedestrians"]
    excluded_terms = []
    if exclude_riders_from_count:
        excluded_terms.append("riders")
    else:
        included_terms.append("riders")
    if require_confident_pedestrian:
        excluded_terms.append("uncertain")
    else:
        included_terms.append("uncertain")
    formula = " + ".join(included_terms)
    explanation = f"{formula} -- matches total_pedestrian_entries in run_summary.json and the sum of hourly_pedestrian_counts.csv."
    if excluded_terms:
        explanation += f" ({' and '.join(excluded_terms)} excluded.)"

    return {
        "run_name": run_name,
        "rider_filtering_enabled": True,
        "exclude_riders_from_count": exclude_riders_from_count,
        "require_confident_pedestrian": require_confident_pedestrian,
        "total_people": total_people,
        "categories": {
            "riders": {
                "count": riders,
                "percent_of_total": _pct(riders),
                "description": (
                    "Bounding box overlapped a same-frame vehicle detection (bicycle/car/"
                    "motorcycle/bus/truck) above rider_filter.rider_overlap_ratio at the "
                    "track's final majority-vote classification. "
                    + ("Excluded from final_pedestrian_count." if exclude_riders_from_count
                       else "NOT excluded from final_pedestrian_count (exclude_riders_from_count is false).")
                ),
            },
            "uncertain": {
                "count": uncertain,
                "percent_of_total": _pct(uncertain),
                "description": (
                    "Some vehicle overlap detected, but below the confident rider_overlap_ratio "
                    "threshold -- not confidently a rider or a pedestrian. Always reported here "
                    "regardless of settings, but only excluded from final_pedestrian_count when "
                    "require_confident_pedestrian is true. "
                    + ("Excluded from final_pedestrian_count (require_confident_pedestrian is true)."
                       if require_confident_pedestrian
                       else "NOT excluded from final_pedestrian_count (require_confident_pedestrian is false).")
                ),
            },
            "confident_pedestrians": {
                "count": confident_pedestrians,
                "percent_of_total": _pct(confident_pedestrians),
                "description": "No significant same-frame vehicle overlap -- classified as a pedestrian. Always included in final_pedestrian_count.",
            },
        },
        "final_pedestrian_count": final_pedestrian_count,
        "final_pedestrian_count_explanation": explanation,
        "caveat": (
            "This is a geometric bbox-overlap heuristic (scripts/rider_filter.py), not a trained "
            "classifier -- spot-check tracks.csv's rider_status column against tracked.mp4 before "
            "trusting these numbers on a new camera."
        ),
    }


def _read_tracks_csv(path: Path) -> Dict[int, dict]:
    result: Dict[int, dict] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[int(row["track_id"])] = row
    return result


def _compute_hourly_demographics(
    entries: List[Tuple[int, int]],
    demographic_records: List[TrackDemographicRecord],
    fps: float,
    start_dt: datetime,
) -> Dict[str, dict]:
    records_by_id = {r.track_id: r for r in demographic_records}
    hourly: Dict[str, dict] = {}

    for track_id, entered_roi_frame in entries:
        ts = frame_idx_to_timestamp(entered_roi_frame, fps, start_dt)
        hb = hour_bucket(ts)
        bucket = hourly.setdefault(
            hb,
            {
                "pedestrian_entries": 0,
                "male": 0,
                "female": 0,
                "unknown_gender": 0,
                "age_0_18": 0,
                "age_18_60": 0,
                "age_60_plus": 0,
                "unknown_age": 0,
            },
        )
        bucket["pedestrian_entries"] += 1

        record = records_by_id.get(track_id)
        if record is None:
            bucket["unknown_gender"] += 1
            bucket["unknown_age"] += 1
            continue

        if record.gender == "male":
            bucket["male"] += 1
        elif record.gender == "female":
            bucket["female"] += 1
        else:
            bucket["unknown_gender"] += 1

        if record.age_group == "0-18":
            bucket["age_0_18"] += 1
        elif record.age_group == "18-60":
            bucket["age_18_60"] += 1
        elif record.age_group == "60+":
            bucket["age_60_plus"] += 1
        else:
            bucket["unknown_age"] += 1

    return hourly
