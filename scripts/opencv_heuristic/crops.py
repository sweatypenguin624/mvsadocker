"""Bus bounding-box crop extraction."""

from __future__ import annotations

import numpy as np

from config import PipelineConfig


def crop_bus(frame, box, cfg: PipelineConfig):
    """Clip, pad, and validate a bus bounding box crop. Returns None if invalid."""
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        if any(v is None or np.isnan(v) for v in (x1, y1, x2, y2)):
            return None
        bw, bh = (x2 - x1), (y2 - y1)
        if bw <= 0 or bh <= 0:
            return None
        pad_x, pad_y = bw * cfg.CROP_PADDING, bh * cfg.CROP_PADDING
        x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        x2, y2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) < cfg.MIN_CROP_WIDTH or (y2 - y1) < cfg.MIN_CROP_HEIGHT:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop
    except Exception:
        return None
