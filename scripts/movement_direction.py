"""Pedestrian movement-direction classification.

Classifies which way a counted pedestrian was moving, relative to two axes
calibrated once per camera (the same pattern as the ROI polygon in
roi_by_camera.yaml -- a human-verified, per-camera geometric fact, not
something inferred generically from image content):

  - road axis: "toward_junction" <-> "away_from_junction". "Junction" means
    whatever landmark the camera's two reference points were calibrated
    against (for a camera not actually facing a junction, calibrate the
    far/near points against whatever the along-the-road landmark is and
    treat the label loosely).
  - cross axis: perpendicular to the road axis -- "left_to_right" <->
    "right_to_left" as seen on screen. Because the cross axis is
    perpendicular to a (usually tilted, due to camera perspective) road
    axis rather than pinned to the image's horizontal, "left to right"
    is defined as whichever perpendicular direction has a positive x
    component -- i.e. the one that reads as moving rightward on screen.

A track's direction is read off its recorded ROI-window trajectory (the
same per-frame bbox history tracker_positions.json already stores), not a
single first-vs-last-frame difference: individual frame-to-frame bbox
positions carry a few px of detector/tracker jitter, so the direction is
instead fit as the dominant axis of the whole point cloud (2x2 covariance
/ PCA leading eigenvector), sign-corrected by net first->last displacement.
That is far less sensitive to one noisy endpoint than a naive
first-minus-last vector, especially on short tracks with only a handful of
recorded points.

Known simplification: a single fixed road-axis vector ignores perspective
(parallel lines actually converge toward a vanishing point, so the "true"
toward-junction direction drifts slightly depending on where in the frame
a pedestrian is). This was checked against real trajectories for
peds_18_kamaripet_police_station__cam_5 before accepting it: 149 confident
pedestrian tracks split into two tight opposite-ish clusters (concentration
r=0.95 each, circular-mean angles -131.5 deg / +24.7 deg) rather than a
smear, meaning the fixed-axis approximation already fits real foot traffic
well for this camera's (comparatively small, compact) ROI. Revisit with a
proper vanishing-point model if this axis is ever reused for a much larger
ROI spanning significant depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

TOWARD_JUNCTION = "toward_junction"
AWAY_FROM_JUNCTION = "away_from_junction"
LEFT_TO_RIGHT = "left_to_right"
RIGHT_TO_LEFT = "right_to_left"
INDETERMINATE = "indeterminate"

ALL_LABELS = (TOWARD_JUNCTION, AWAY_FROM_JUNCTION, LEFT_TO_RIGHT, RIGHT_TO_LEFT, INDETERMINATE)

Point = Tuple[float, float]


@dataclass
class MovementAxis:
    """A camera's calibrated road direction, as two reference points in the
    same reference_width x reference_height pixel space as that camera's
    ROI polygon (see roi_by_camera.yaml / roi.py's PedestrianROI).
    """

    near_point: Point  # a point along the road, away from the junction (nearer the camera)
    far_point: Point  # a point along the road, toward the junction
    reference_width: int = 1920
    reference_height: int = 1080

    def scaled(self, frame_width: int, frame_height: int) -> "MovementAxis":
        if frame_width == self.reference_width and frame_height == self.reference_height:
            return self
        sx = frame_width / float(self.reference_width)
        sy = frame_height / float(self.reference_height)
        return MovementAxis(
            near_point=(self.near_point[0] * sx, self.near_point[1] * sy),
            far_point=(self.far_point[0] * sx, self.far_point[1] * sy),
            reference_width=frame_width,
            reference_height=frame_height,
        )

    def unit_vectors(self) -> Tuple[Point, Point]:
        """Returns (toward_junction_unit, cross_unit). cross_unit is the
        perpendicular whose x component is >= 0, i.e. "left to right"."""
        dx = self.far_point[0] - self.near_point[0]
        dy = self.far_point[1] - self.near_point[1]
        norm = math.hypot(dx, dy)
        if norm == 0:
            raise ValueError("MovementAxis near_point and far_point must differ")
        toward = (dx / norm, dy / norm)
        perp = (-toward[1], toward[0])
        if perp[0] < 0:
            perp = (-perp[0], -perp[1])
        return toward, perp


@dataclass
class DirectionResult:
    label: str
    displacement_px: float
    along_axis_component: float
    cross_axis_component: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "displacement_px": round(self.displacement_px, 1),
            "along_axis_component": round(self.along_axis_component, 1),
            "cross_axis_component": round(self.cross_axis_component, 1),
        }


def _fit_direction(points: Sequence[Point]) -> Optional[Tuple[Point, float]]:
    """Fit the dominant direction of travel through a track's recorded
    (x, y) positions via PCA on the point cloud (its leading eigenvector).

    Sign (which way along that axis the track actually moved) and
    displacement (how far it moved along that axis) are both read off the
    *whole* sequence rather than the first/last point alone: every point is
    projected onto the fitted axis, then sign is the sign of the
    correlation between that projection and frame order, and displacement
    is the projection's range (max - min). A single corrupted frame (a
    momentary detector/tracker glitch) then only ever contributes 1/n of
    the sign signal instead of being able to flip it outright the way
    comparing just positions[0] to positions[-1] would.

    Degenerates cleanly to the plain first->last vector for a 2-point
    track, where PCA and the endpoint difference are the same thing anyway.
    Returns (unit_direction, displacement_along_axis_px), or None if there
    are fewer than 2 points or they're all coincident.
    """
    n = len(points)
    if n < 2:
        return None

    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n

    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    syy = sum((p[1] - mean_y) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)

    if sxx == 0.0 and syy == 0.0:
        return None

    # Closed-form leading eigenvector of the 2x2 symmetric covariance
    # matrix [[sxx, sxy], [sxy, syy]].
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, (trace / 2.0) ** 2 - det))
    lam1 = trace / 2.0 + disc

    if sxy != 0.0:
        vx, vy = lam1 - syy, sxy
    elif sxx >= syy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy)
    if norm == 0.0:
        return None
    vx, vy = vx / norm, vy / norm

    projections = [(p[0] - mean_x) * vx + (p[1] - mean_y) * vy for p in points]
    mean_t = (n - 1) / 2.0
    mean_p = sum(projections) / n
    covariance = sum((i - mean_t) * (p - mean_p) for i, p in enumerate(projections))
    if covariance < 0:
        vx, vy = -vx, -vy
        projections = [-p for p in projections]

    displacement = max(projections) - min(projections)
    return (vx, vy), displacement


def classify_direction(
    positions: Sequence[Point],
    axis: MovementAxis,
    min_displacement_px: float = 40.0,
) -> DirectionResult:
    """positions: a track's recorded (x, y) points in frame order -- must be
    the same point convention (bottom-center) the axis was calibrated
    against. Tracks whose motion along their own best-fit direction is
    under min_displacement_px are reported "indeterminate" rather than
    guessed: at that scale the direction is dominated by detector/tracker
    jitter, not real motion.
    """
    fit = _fit_direction(positions)
    if fit is None:
        return DirectionResult(INDETERMINATE, 0.0, 0.0, 0.0)
    direction, displacement = fit

    if displacement < min_displacement_px:
        return DirectionResult(INDETERMINATE, displacement, 0.0, 0.0)

    toward_unit, cross_unit = axis.unit_vectors()
    along = direction[0] * toward_unit[0] + direction[1] * toward_unit[1]
    cross = direction[0] * cross_unit[0] + direction[1] * cross_unit[1]

    if abs(along) >= abs(cross):
        label = TOWARD_JUNCTION if along > 0 else AWAY_FROM_JUNCTION
    else:
        label = LEFT_TO_RIGHT if cross > 0 else RIGHT_TO_LEFT

    return DirectionResult(label, displacement, along, cross)


def movement_axis_from_config(entry: dict) -> Optional[MovementAxis]:
    """Build a MovementAxis from a camera's "movement_axis" sub-dict in
    roi_by_camera.yaml, e.g.:

        "movement_axis": {
            "near_point": [1945, 1088],
            "far_point": [1463, 730]
        }

    Returns None if the camera has no movement_axis entry (direction
    classification is opt-in per camera, same as ROI itself).
    """
    if not entry:
        return None
    near = entry.get("near_point")
    far = entry.get("far_point")
    if near is None or far is None:
        return None
    return MovementAxis(
        near_point=(float(near[0]), float(near[1])),
        far_point=(float(far[0]), float(far[1])),
        reference_width=int(entry.get("reference_width", 1920)),
        reference_height=int(entry.get("reference_height", 1080)),
    )
