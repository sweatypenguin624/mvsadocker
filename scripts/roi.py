"""Central ROI (region of interest) definition and point-in-polygon logic.

This is the ONLY place in the codebase that should implement point-in-polygon
testing or bottom-center bbox math. Every other module (tracker.py, output.py,
pipeline.py) imports from here so the ROI is defined once and used
consistently everywhere.

No third-party geometry library (shapely, matplotlib) is used on purpose --
a plain ray-casting test avoids adding a dependency that could interact with
the pinned numpy/torch environment on the research server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from models import BBox

Point = Tuple[float, float]


@dataclass
class PedestrianROI:
    """A polygon ROI defined in the coordinate space of ``reference_width`` x
    ``reference_height`` (normally the native camera resolution, e.g.
    1920x1080). If a video is processed at a different resolution, points
    are scaled accordingly by ``scale_to``.
    """

    polygon: List[Point]
    reference_width: int = 1920
    reference_height: int = 1080

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError("ROI polygon must have at least 3 points")

    def scaled_polygon(self, frame_width: int, frame_height: int) -> List[Point]:
        """Return the polygon rescaled to a frame of the given size."""
        if frame_width == self.reference_width and frame_height == self.reference_height:
            return self.polygon
        sx = frame_width / float(self.reference_width)
        sy = frame_height / float(self.reference_height)
        return [(x * sx, y * sy) for x, y in self.polygon]

    def contains_point(
        self, point: Point, frame_width: int = None, frame_height: int = None
    ) -> bool:
        """Return True if ``point`` (in the same coordinate space as
        ``frame_width``/``frame_height``, or the reference space if omitted)
        lies inside the ROI polygon.
        """
        polygon = self.polygon
        if frame_width is not None and frame_height is not None:
            polygon = self.scaled_polygon(frame_width, frame_height)
        return point_in_polygon(point, polygon)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Standard ray-casting point-in-polygon test.

    Points exactly on an edge may be classified either way depending on
    floating point rounding -- this is an accepted limitation of the
    ray-casting approach and is not significant for pedestrian-scale ROIs.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def bottom_center(bbox: BBox) -> Point:
    """Return the bottom-center ("feet") point of a YOLO xyxy bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def roi_from_config(roi_config) -> PedestrianROI:
    """Build a PedestrianROI from the ``roi`` section of the loaded config."""
    polygon = [(float(p[0]), float(p[1])) for p in roi_config.polygon]
    return PedestrianROI(
        polygon=polygon,
        reference_width=roi_config.reference_width,
        reference_height=roi_config.reference_height,
    )
