"""Thin adapter around scripts/movement_direction.py for Layer 1 tracks.

Nothing here reimplements direction math -- classify_direction, MovementAxis
and the ALL_LABELS set (toward_junction / away_from_junction / left_to_right
/ right_to_left / indeterminate) are all reused verbatim from the pedestrian
pipeline's movement_direction.py, since Layer 1's counting is built on the
same ROI-polygon-entry paradigm (see layer1_tracker.py) and the spec's required
direction labels are exactly that module's ALL_LABELS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from movement_direction import (  # noqa: E402
    INDETERMINATE,
    MovementAxis,
    classify_direction,
    movement_axis_from_config,
)

Point = Tuple[float, float]


def resolve_direction(
    positions: Sequence[Point],
    axis: Optional[MovementAxis],
    frame_width: int,
    frame_height: int,
    min_displacement_fraction_of_frame_diagonal: float,
) -> str:
    """Classify a counted track's direction. Returns "indeterminate" when no
    movement_axis is calibrated for this camera -- direction classification
    is opt-in per camera, same as the pedestrian pipeline (see
    config/roi/roi_by_camera.yaml's movement_axis entries).
    """
    if axis is None:
        return INDETERMINATE

    diagonal = (frame_width**2 + frame_height**2) ** 0.5
    min_displacement_px = min_displacement_fraction_of_frame_diagonal * diagonal

    scaled_axis = axis.scaled(frame_width, frame_height)
    result = classify_direction(positions, scaled_axis, min_displacement_px=min_displacement_px)
    return result.label
