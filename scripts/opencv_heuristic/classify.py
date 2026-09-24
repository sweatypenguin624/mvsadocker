"""OpenCV HSV heuristic bus-color classifier.

Masks out low-saturation/low-value pixels (shadow, glare, white/gray/black
bodywork), buckets remaining pixel hues into the configured color ranges,
and returns the majority color with a confidence equal to (winning color
pixel count / total valid colored pixels).
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from config import PipelineConfig


def classify_bus_crop(crop, cfg: PipelineConfig):
    try:
        if crop is None or crop.size == 0:
            return None, None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        valid_mask = (s >= cfg.HSV_MIN_SATURATION) & (v >= cfg.HSV_MIN_VALUE)
        total_valid = int(np.count_nonzero(valid_mask))
        if total_valid == 0:
            return cfg.FALLBACK_COLOR, 0.0

        color_pixel_counts = {}
        for color_name, ranges in cfg.HSV_COLOR_RANGES.items():
            min_sat = cfg.HSV_COLOR_MIN_SATURATION.get(color_name, cfg.HSV_MIN_SATURATION)
            color_valid_mask = valid_mask if min_sat <= cfg.HSV_MIN_SATURATION else (valid_mask & (s >= min_sat))
            color_mask = np.zeros_like(valid_mask)
            for lo, hi in ranges:
                color_mask |= (h >= lo) & (h <= hi)
            color_mask &= color_valid_mask
            color_pixel_counts[color_name] = int(np.count_nonzero(color_mask))

        matched_total = sum(color_pixel_counts.values())
        if matched_total == 0:
            return "rest", (total_valid - matched_total) / total_valid

        winning_color = max(color_pixel_counts, key=color_pixel_counts.get)
        winning_count = color_pixel_counts[winning_color]

        # Reject weak/ambiguous signal instead of forcing a pick: too little of
        # the crop matched any color at all, or no single color clearly won.
        match_coverage = matched_total / total_valid
        win_share = winning_count / matched_total
        if match_coverage < cfg.MIN_COLOR_MATCH_COVERAGE or win_share < cfg.MIN_COLOR_WIN_SHARE:
            return cfg.FALLBACK_COLOR, winning_count / total_valid

        confidence = winning_count / total_valid
        return winning_color, confidence
    except Exception as e:
        if cfg.DEBUG:
            print(f"  [WARN] classify_bus_crop failed: {e}")
        return None, None


class TrackColorAccumulator:
    """Per-track running evidence over multiple classify_bus_crop calls."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._history = defaultdict(lambda: {c: 0.0 for c in cfg.COLOR_CLASSES})
        self.prediction_count = defaultdict(int)

    def update(self, track_id, color, confidence):
        if color not in self.cfg.COLOR_CLASSES or confidence is None:
            return
        self._history[track_id][color] += float(confidence)
        self.prediction_count[track_id] += 1

    def final_color(self, track_id):
        evidence = self._history.get(track_id, {})
        total = sum(evidence.values())
        if total <= 0:
            return self.cfg.FALLBACK_COLOR, 0.0
        winning_color = max(evidence, key=evidence.get)
        return winning_color, (evidence[winning_color] / total)
