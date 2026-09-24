"""Debug/annotated-video rendering for Layer 1.

Extends scripts/vehicle_counting/vehicle_output.py::draw_annotations'
pattern (ROI polygon overlay + per-box label + running-count HUD) to also
show track id / class / confidence / direction per box, per spec section 19.
Deliberately shows only the broad "Bus" label, never a subtype -- that
belongs to Layer 2.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from roi import PedestrianROI  # noqa: E402

from classifier import vote_class  # noqa: E402
from layer1_models import SourcedDetection, TrackState  # noqa: E402


def draw_debug_frame(
    frame: np.ndarray,
    live_tracks: List[Tuple[SourcedDetection, TrackState]],
    roi: PedestrianROI,
    running_counts: Dict[str, int],
    timestamp: datetime,
) -> np.ndarray:
    h, w = frame.shape[:2]
    polygon = np.array(roi.scaled_polygon(w, h), dtype=np.int32)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    for det, state in live_tracks:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        live_class = state.final_class or vote_class(state.class_votes) or det.bucket
        color = (0, 200, 0) if state.counted else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        direction = state.direction_label or "-"
        label_lines = [
            f"ID:{state.track_id} {live_class}",
            f"conf:{det.conf:.2f} dir:{direction}",
        ]
        ty = max(0, y1 - 10)
        for i, text in enumerate(label_lines):
            cv2.putText(
                frame,
                text,
                (x1, ty - (len(label_lines) - 1 - i) * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    hud_text = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}  |  " + "  ".join(
        f"{cls}:{count}" for cls, count in running_counts.items() if count
    )
    cv2.putText(frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame
