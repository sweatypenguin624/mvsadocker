"""Bus subclass classification -- runs only on frames Step 1 already
labeled "Bus" or "Mini-bus" (UVH-26 native classes).

5 subclasses (confirmed with user 2026-08-28):
    School Bus, BRT Bus, Mini Bus, Private Bus or City Bus, Other Bus

Decision logic, in priority order:

1. Mini Bus: taken directly from Step 1's native class ("Mini-bus"), not
   re-derived from color/shape. UVH-26 already distinguishes bus form
   factor; re-guessing it from a color heuristic would be strictly worse.

2. School Bus / BRT Bus: color-heuristic-only (per user decision -- no
   labeled livery dataset exists for a real classifier yet, see
   INPUT_REQUIREMENTS.md). Computed as the fraction of bbox pixels
   falling inside each livery's HSV range (config/bus_subclass_config.yaml).
   Whichever color clears its threshold AND dominates the other by
   `dominance_margin` wins.

3. Private Bus or City Bus: the default "confidently a bus, not
   yellow/mini, no BRT blue" bucket. Distinguishing an actual private
   operator from a municipal city-transport bus needs text/logo/route
   info recognition (OCR or a fine-tuned livery model) -- out of scope
   for a pure color heuristic, which is why the user's spec collapses
   these two into one class rather than leaving "City Bus" unreachable.

4. Other Bus: used whenever the color signal itself can't be trusted --
   crop too small, too dark to read color at all (night/IR/heavy
   underexposure -- judged by brightness, NOT saturation, since a plain
   white/silver bus is legitimately low-saturation and must still fall
   through to bucket 3, not here), or yellow vs. blue too close to call.
   This is the "doesn't confidently fit the above" bucket from the
   user's spec, applied to signal quality rather than forced into a
   guess.

Known limitation: color-only detection will misfire on a school bus shot
in low light/IR (loses yellow saturation -> falls to Other Bus, not a
wrong label but a missed one) and cannot split Private vs. City buses at
all yet. Both require real training data / OCR, tracked as future work,
not attempted here with fragile pixel heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

SCHOOL_BUS = "School Bus"
BRT_BUS = "BRT Bus"
MINI_BUS = "Mini Bus"
PRIVATE_OR_CITY_BUS = "Private Bus or City Bus"
OTHER_BUS = "Other Bus"


@dataclass
class BusSubclassResult:
    subclass: str
    confidence: float
    reason: str
    yellow_ratio: Optional[float] = None
    blue_ratio: Optional[float] = None


class BusSubclassClassifier:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def classify(self, crop_bgr: np.ndarray, native_class: str) -> BusSubclassResult:
        if native_class == "Mini-bus":
            return BusSubclassResult(
                subclass=MINI_BUS,
                confidence=self.cfg["confidence"]["mini_bus_from_native_class"],
                reason="native_class",
            )

        h, w = crop_bgr.shape[:2]
        if h * w < self.cfg["min_bbox_area_px"]:
            return BusSubclassResult(
                subclass=OTHER_BUS,
                confidence=self.cfg["confidence"]["other_bus_low_signal"],
                reason="bbox_too_small",
            )

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        total_px = h * w

        mean_brightness = float(np.mean(v_ch))
        if mean_brightness < self.cfg["min_mean_brightness"]:
            return BusSubclassResult(
                subclass=OTHER_BUS,
                confidence=self.cfg["confidence"]["other_bus_low_signal"],
                reason="crop_too_dark",
            )

        yellow_ratio = self._hue_ratio(h_ch, s_ch, v_ch, self.cfg["school_bus_yellow"], total_px)
        blue_ratio = self._hue_ratio(h_ch, s_ch, v_ch, self.cfg["brt_blue"], total_px)

        threshold = self.cfg["color_ratio_threshold"]
        margin = self.cfg["dominance_margin"]

        yellow_wins = yellow_ratio >= threshold and yellow_ratio >= blue_ratio * margin
        blue_wins = blue_ratio >= threshold and blue_ratio >= yellow_ratio * margin

        if yellow_wins and not blue_wins:
            return BusSubclassResult(
                subclass=SCHOOL_BUS,
                confidence=self.cfg["confidence"]["color_dominant"],
                reason="yellow_dominant",
                yellow_ratio=yellow_ratio,
                blue_ratio=blue_ratio,
            )
        if blue_wins and not yellow_wins:
            return BusSubclassResult(
                subclass=BRT_BUS,
                confidence=self.cfg["confidence"]["color_dominant"],
                reason="blue_dominant",
                yellow_ratio=yellow_ratio,
                blue_ratio=blue_ratio,
            )
        if yellow_wins and blue_wins:
            # Both clear threshold and each other's margin: contradictory
            # signal (e.g. a yellow school bus glimpsed next to a blue BRT
            # bus inside a loose crop) -- don't guess.
            return BusSubclassResult(
                subclass=OTHER_BUS,
                confidence=self.cfg["confidence"]["other_bus_ambiguous"],
                reason="yellow_and_blue_both_present",
                yellow_ratio=yellow_ratio,
                blue_ratio=blue_ratio,
            )

        # Neither color present in force -- the common case for ordinary
        # private/city bus liveries (white, silver, multicolor ads, etc.).
        return BusSubclassResult(
            subclass=PRIVATE_OR_CITY_BUS,
            confidence=self.cfg["confidence"]["color_default_bucket"],
            reason="no_dominant_livery_color",
            yellow_ratio=yellow_ratio,
            blue_ratio=blue_ratio,
        )

    @staticmethod
    def _hue_ratio(h_ch, s_ch, v_ch, hue_range: dict, total_px: int) -> float:
        mask = (
            (h_ch >= hue_range["hue_min"])
            & (h_ch <= hue_range["hue_max"])
            & (s_ch >= hue_range["sat_min"])
            & (v_ch >= hue_range["val_min"])
        )
        return float(np.count_nonzero(mask)) / total_px
