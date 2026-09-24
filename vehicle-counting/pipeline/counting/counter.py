import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from tracker import TrackStateEnum

logger = logging.getLogger("uvh_test.counter")


class VehicleCounter:
    """
    Robust line-crossing counter.

    Design goals:
    - Count a physical track at most once.
    - Use bottom-center of bbox for road vehicles.
    - Require a genuine side-of-line transition.
    - Require trajectory evidence before counting.
    - Avoid large fixed pixel-based duplicate suppression.
    - Keep enough diagnostics to understand failures.

    This class is intentionally conservative:
    a track should demonstrate that it approached and crossed the
    counting line before it is counted.
    """

    def __init__(
        self,
        line_points: List[List[float]],
        count_direction: str = "BOTH",
        min_track_frames: int = 3,
        min_crossing_frames: int = 2,
        trajectory_size: int = 30,
        crossing_cooldown_frames: int = 8,
    ):
        if not line_points or len(line_points) != 2:
            raise ValueError(
                "line_points must contain exactly two points"
            )

        self.line_p1 = np.array(line_points[0], dtype=float)
        self.line_p2 = np.array(line_points[1], dtype=float)

        if np.allclose(self.line_p1, self.line_p2):
            raise ValueError("Counting line points must be different")

        self.count_direction = count_direction.upper()

        self.min_track_frames = max(1, int(min_track_frames))
        self.min_crossing_frames = max(1, int(min_crossing_frames))
        self.trajectory_size = max(5, int(trajectory_size))
        self.crossing_cooldown_frames = max(0, int(crossing_cooldown_frames))

        # Track IDs that have already produced a valid count.
        self.counted_ids = set()
        self.recent_crossings = []  # stores (frame_idx, direction, class, point)

        # Track IDs that have been explicitly rejected after crossing.
        self.rejected_ids = set()

        # Previous side-of-line state for each track.
        self.previous_side: Dict[int, int] = {}

        # Consecutive observations on each side.
        self.side_history: Dict[int, deque] = {}

        # Recent bottom-center trajectory.
        self.trajectories: Dict[int, deque] = {}

        # Last frame where a track was seen.
        self.last_seen_frame: Dict[int, int] = {}

        # Last frame where a crossing was accepted.
        self.last_crossing_frame: Dict[int, int] = {}

        # Track → crossing information.
        self.crossing_info: Dict[int, dict] = {}

        # Track → latest state reference.
        self.track_states: Dict[int, object] = {}

        # Output records.
        self.records: List[dict] = []

        # Aggregate counts.
        self.counts: Dict[str, Dict[str, int]] = {}

        # Debug statistics.
        self.stats = {
            "tracks_seen": 0,
            "crossing_candidates": 0,
            "valid_crossings": 0,
            "counted": 0,
            "rejected_short_track": 0,
            "rejected_direction": 0,
            "rejected_duplicate": 0,
            "rejected_fragment": 0,
        }

        self.frame_count = 0
        self.pending_crossings = {}

        logger.info(
            "VehicleCounter initialized: line=%s -> %s, direction=%s",
            self.line_p1.tolist(),
            self.line_p2.tolist(),
            self.count_direction,
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def bottom_center(self, box) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, box)
        return (
            (x1 + x2) / 2.0,
            y2,
        )

    def side_of_line(self, point: Tuple[float, float]) -> int:
        """
        Returns:
            +1 = one side
            -1 = opposite side
             0 = exactly/on the line
        """
        p = np.array(point, dtype=float)

        direction = self.line_p2 - self.line_p1
        relative = p - self.line_p1

        cross = (
            direction[0] * relative[1]
            - direction[1] * relative[0]
        )

        eps = 1e-6

        if cross > eps:
            return 1

        if cross < -eps:
            return -1

        return 0

    def direction_from_sides(self, previous_side: int, current_side: int) -> str:
        """
        Converts a side transition into a stable direction label.

        The labels are intentionally generic:
        A_TO_B / B_TO_A

        This avoids assuming that image-left/right corresponds to
        physical traffic direction.
        """
        if previous_side < 0 and current_side > 0:
            return "A_TO_B"

        if previous_side > 0 and current_side < 0:
            return "B_TO_A"

        return "UNKNOWN"

    def direction_allowed(self, direction: str) -> bool:
        if self.count_direction == "BOTH":
            return True

        return direction.upper() == self.count_direction

    # ------------------------------------------------------------------
    # Track helpers
    # ------------------------------------------------------------------

    def _get_track_age(self, state, frame_idx: int) -> int:
        if state is None:
            return 0

        first_seen = getattr(state, "first_seen_frame", frame_idx)

        return frame_idx - first_seen + 1

    def _update_trajectory(
        self,
        tid: int,
        point: Tuple[float, float],
    ):
        if tid not in self.trajectories:
            self.trajectories[tid] = deque(
                maxlen=self.trajectory_size
            )

        self.trajectories[tid].append(point)

    def _update_side_history(
        self,
        tid: int,
        side: int,
    ):
        if tid not in self.side_history:
            self.side_history[tid] = deque(
                maxlen=self.min_crossing_frames + 2
            )

        if side != 0:
            self.side_history[tid].append(side)

    def _has_stable_previous_side(
        self,
        tid: int,
        previous_side: int,
    ) -> bool:
        history = self.side_history.get(tid)

        if not history:
            return False

        # We need enough evidence that the vehicle was actually
        # travelling on the previous side before crossing.
        # Account for the fact that current_side was already appended
        needed = min(
            self.min_crossing_frames,
            len(history) - 1,
        )

        if needed <= 0:
            return False

        recent = list(history)[-(needed + 1):-1]

        return all(side == previous_side for side in recent)

    def _has_post_crossing_evidence(
        self,
        tid: int,
        current_side: int,
    ) -> bool:
        history = self.side_history.get(tid)

        if not history:
            return False

        # Current observation is already in the history.
        needed = min(
            self.min_crossing_frames,
            len(history),
        )

        if needed <= 0:
            return False

        recent = list(history)[-needed:]

        return all(side == current_side for side in recent)

    def _trajectory_moving_towards_line(
        self,
        tid: int,
        previous_side: int,
        current_side: int,
    ) -> bool:
        """
        Basic trajectory sanity check.

        We do not require a sophisticated velocity model here.
        We only reject obviously unstable transitions.

        A side transition already provides the strongest evidence.
        """
        trajectory = self.trajectories.get(tid)

        if not trajectory or len(trajectory) < 2:
            return True

        p_prev = np.array(trajectory[-2], dtype=float)
        p_curr = np.array(trajectory[-1], dtype=float)

        movement = p_curr - p_prev

        if np.linalg.norm(movement) < 1e-6:
            return False

        return True

    # ------------------------------------------------------------------
    # Crossing
    # ------------------------------------------------------------------

    def _detect_crossing(
        self,
        tid: int,
        point: Tuple[float, float],
        frame_idx: int,
    ) -> Tuple[bool, Optional[str]]:
        current_side = self.side_of_line(point)

        # Exactly on the line: don't decide yet.
        if current_side == 0:
            return False, None

        previous_side = self.previous_side.get(tid)

        # First valid observation.
        if previous_side is None:
            self.previous_side[tid] = current_side
            self._update_side_history(tid, current_side)
            return False, None

        # Same side -> no crossing.
        if current_side == previous_side:
            self._update_side_history(tid, current_side)
            
            if tid in self.pending_crossings:
                cand_frame, old_side, new_side, cand_dir = self.pending_crossings[tid]
                if current_side == new_side:
                    if self._has_post_crossing_evidence(tid, current_side):
                        del self.pending_crossings[tid]
                        self.stats["valid_crossings"] += 1
                        logger.info(f"CROSSING_CONFIRMED: frame={frame_idx} tid={tid} dir={cand_dir}")
                        return True, cand_dir
                else:
                    del self.pending_crossings[tid]
                    logger.info(f"CROSSING_REJECTED: frame={frame_idx} tid={tid} reverted to previous side before evidence")
                    
            return False, None

        # We have a side transition.
        self.stats["crossing_candidates"] += 1

        direction = self.direction_from_sides(
            previous_side,
            current_side,
        )

        # Update state immediately so a noisy oscillation doesn't
        # repeatedly generate the same candidate.
        self.previous_side[tid] = current_side
        self._update_side_history(tid, current_side)

        # Need stable evidence before the transition.
        if not self._has_stable_previous_side(
            tid,
            previous_side,
        ):
            return False, None

        if not self._trajectory_moving_towards_line(
            tid,
            previous_side,
            current_side,
        ):
            return False, None

        # Direction restriction.
        if not self.direction_allowed(direction):
            self.stats["rejected_direction"] += 1
            return False, None

        self.pending_crossings[tid] = (frame_idx, previous_side, current_side, direction)
        logger.info(f"CROSSING_CANDIDATE: frame={frame_idx} tid={tid} dir={direction}")
        
        if self._has_post_crossing_evidence(tid, current_side):
            del self.pending_crossings[tid]
            self.stats["valid_crossings"] += 1
            logger.info(f"CROSSING_CONFIRMED: frame={frame_idx} tid={tid} dir={direction}")
            return True, direction

        return False, None

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        active_tracks: list,
        frame=None,
        bus_classifier=None,
        truck_classifier=None,
    ) -> List[dict]:
        """
        Process all active tracks for the current frame.

        active_tracks format:
            [
                (track_id, TrackState, bbox),
                ...
            ]
        """
        self.frame_count = frame_idx

        frame_records = []

        for tid, state, box in active_tracks:
            tid = int(tid)

            self.track_states[tid] = state
            self.last_seen_frame[tid] = frame_idx

            self.stats["tracks_seen"] += 1

            point = self.bottom_center(box)

            self._update_trajectory(tid, point)

            track_age = self._get_track_age(
                state,
                frame_idx,
            )

            # Establish initial side before considering crossing.
            side = self.side_of_line(point)

            if tid not in self.previous_side and side != 0:
                self.previous_side[tid] = side
                self._update_side_history(tid, side)

            crossed, direction = self._detect_crossing(
                tid,
                point,
                frame_idx,
            )

            if not crossed:
                continue

            # ----------------------------------------------------------
            # One count per track ID.
            # ----------------------------------------------------------
            if tid in self.counted_ids:
                self.stats["rejected_duplicate"] += 1
                sc = "Unknown"
                try: sc = state.get_stable_class(self._class_names)
                except: pass
                logger.info(f"REJECTED frame={frame_idx} tid={tid} class={sc} reason=duplicate")
                continue

            # Track must have existed long enough.
            if track_age < self.min_track_frames:
                self.stats["rejected_short_track"] += 1

                try:
                    state.not_counted_reason = (
                        "Crossed line before minimum track age"
                    )
                except Exception:
                    pass

                continue

            # Optional cooldown protection.
            previous_crossing = self.last_crossing_frame.get(tid)

            if (
                previous_crossing is not None
                and frame_idx - previous_crossing
                <= self.crossing_cooldown_frames
            ):
                self.stats["rejected_duplicate"] += 1
                sc = "Unknown"
                try: sc = state.get_stable_class(self._class_names)
                except: pass
                logger.info(f"REJECTED frame={frame_idx} tid={tid} class={sc} reason=duplicate")
                continue

            # ----------------------------------------------------------
            # Valid count.
            # ----------------------------------------------------------

            stable_class = "Unknown"

            try:
                stable_class = state.get_stable_class(
                    self._class_names
                )
            except Exception:
                pass

            if bus_classifier is not None and frame is not None and stable_class == "Bus":
                try:
                    x1, y1, x2, y2 = (int(round(v)) for v in box)
                    h, w = frame.shape[:2]
                    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                    if crop.size > 0:
                        bus_res = bus_classifier.classify(crop, stable_class)
                        stable_class = bus_res.subclass
                        state.override_class = stable_class
                except Exception as e:
                    import logging
                    logging.getLogger("uvh_test.counter").warning(f"Bus subclassification failed: {e}")

            if truck_classifier is not None and frame is not None and stable_class in ["Truck", "Three-wheeler"]:
                try:
                    best_crop = getattr(state, "best_crop", None)
                    if best_crop is None or best_crop.size == 0:
                        x1, y1, x2, y2 = (int(round(v)) for v in box)
                        h, w = frame.shape[:2]
                        best_crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                    if best_crop is not None and best_crop.size > 0:
                        truck_res = truck_classifier.classify(best_crop, stable_class)
                        stable_class = truck_res.subclass
                        state.override_class = stable_class
                except Exception as e:
                    import logging
                    logging.getLogger("uvh_test.counter").warning(f"Truck subclassification failed: {e}")

            
            # --- Anti-Fragmentation (Smart) ---
            is_fragment = False
            
            is_tw_curr = any(kw in stable_class.lower() for kw in ["wheel", "motor", "bike", "bicycle"])
            
            for rc in self.recent_crossings:
                rc_frame = rc[0]
                rc_dir = rc[1]
                rc_class = rc[2]
                rc_point = rc[3]
                rc_tid = rc[4] if len(rc) > 4 else None
                
                old_track_alive = (rc_tid is not None) and (self.last_seen_frame.get(rc_tid, -1) == frame_idx)
                if old_track_alive:
                    continue
                    
                is_tw_rc = any(kw in rc_class.lower() for kw in ["wheel", "motor", "bike", "bicycle"])
                class_match = (rc_class == stable_class) or (is_tw_curr and is_tw_rc)
                
                if class_match and rc_dir == direction:
                    time_diff = frame_idx - rc_frame
                    time_thresh = 30 if (is_tw_curr or is_tw_rc) else 15
                    
                    if time_diff < time_thresh:
                        dist = ((point[0] - rc_point[0])**2 + (point[1] - rc_point[1])**2)**0.5
                        bbox_size = max(box[2] - box[0], box[3] - box[1])
                        spatial_thresh = max(bbox_size, 150) if (is_tw_curr or is_tw_rc) else bbox_size
                        
                        if dist < spatial_thresh:
                            is_fragment = True
                            self.stats["rejected_fragment"] = self.stats.get("rejected_fragment", 0) + 1
                            logger.info(f"REJECTED frame={frame_idx} tid={tid} class={stable_class} reason=temporal_fragment")
                            break
            
            if is_fragment:
                # Add to counted_ids anyway so we don't keep checking it
                self.counted_ids.add(tid)
                continue
            
            # Record this valid crossing
            self.recent_crossings.append((frame_idx, direction, stable_class, point, tid))
            # Keep only recent crossings to avoid memory bloat
            self.recent_crossings = [rc for rc in self.recent_crossings if frame_idx - rc[0] < 300]
            
            self.counted_ids.add(tid)
            self.last_crossing_frame[tid] = frame_idx
            logger.info(f"COUNTED  frame={frame_idx} tid={tid} class={stable_class} direction={direction}")

            try:
                state.counted = True
                state.state = TrackStateEnum.COUNTED
                state.crossing_frame = frame_idx
                state.crossing_direction = direction
                state.counting_bbox = (
                    box.tolist()
                    if hasattr(box, "tolist")
                    else list(box)
                )
                state.not_counted_reason = ""
            except Exception:
                pass

            self.crossing_info[tid] = {
                "frame": frame_idx,
                "direction": direction,
                "class": stable_class,
                "point": [
                    float(point[0]),
                    float(point[1]),
                ],
            }

            if stable_class not in self.counts:
                self.counts[stable_class] = {
                    "total": 0,
                    "A_TO_B": 0,
                    "B_TO_A": 0,
                }

            self.counts[stable_class]["total"] += 1

            if direction in ("A_TO_B", "B_TO_A"):
                self.counts[stable_class][direction] += 1

            record = {
                "frame": frame_idx,
                "class": stable_class,
                "direction": direction,
                "track_id": tid,
            }

            self.records.append(record)
            frame_records.append(record)

            self.stats["counted"] += 1

            logger.info(
                "COUNT: frame=%d tid=%d class=%s direction=%s "
                "track_age=%d point=(%.1f, %.1f)",
                frame_idx,
                tid,
                stable_class,
                direction,
                track_age,
                point[0],
                point[1],
            )

        return frame_records

    # ------------------------------------------------------------------
    # Class names
    # ------------------------------------------------------------------

    def set_class_names(self, class_names):
        """
        Called by main.py after VehicleCounter construction.
        """
        self._class_names = class_names or {}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        return {
            "total_count": len(self.records),
            "counts": self.counts,
            "stats": self.stats,
            "unique_track_ids_counted": len(self.counted_ids),
        }

    def get_records(self) -> List[dict]:
        return list(self.records)

    def cleanup_track(self, tid: int):
        """
        Remove transient memory for an old track.

        Do NOT remove tid from counted_ids.
        counted_ids is intentionally permanent for the duration
        of this video.
        """
        tid = int(tid)

        self.previous_side.pop(tid, None)
        self.side_history.pop(tid, None)
        self.trajectories.pop(tid, None)
        self.last_seen_frame.pop(tid, None)
        self.track_states.pop(tid, None)

    def reset(self):
        self.counted_ids.clear()
        self.recent_crossings.clear()
        self.rejected_ids.clear()
        self.previous_side.clear()
        self.side_history.clear()
        self.trajectories.clear()
        self.last_seen_frame.clear()
        self.last_crossing_frame.clear()
        self.crossing_info.clear()
        self.track_states.clear()
        self.records.clear()
        self.counts.clear()

        self.stats = {
            "tracks_seen": 0,
            "crossing_candidates": 0,
            "valid_crossings": 0,
            "counted": 0,
            "rejected_short_track": 0,
            "rejected_direction": 0,
            "rejected_duplicate": 0,
            "rejected_fragment": 0,
        }

        self.frame_count = 0
        self.pending_crossings = {}


# ----------------------------------------------------------------------
# Compatibility helper
# ----------------------------------------------------------------------

def create_counter(
    line_points,
    count_direction="BOTH",
    class_names=None,
    **kwargs,
):
    """
    Convenience constructor.

    Example:
        counter = create_counter(
            line_points,
            count_direction="BOTH",
            class_names=detector.class_names,
        )
    """
    counter = VehicleCounter(
        line_points=line_points,
        count_direction=count_direction,
        **kwargs,
    )

    counter.set_class_names(class_names or {})

    return counter
