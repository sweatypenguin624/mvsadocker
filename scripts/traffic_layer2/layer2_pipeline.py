"""Glues the per-track cascade together (spec section 6 diagram):

    Goods Vehicle track
          |
    Broad Classifier -> 3W | LCV | Heavy Truck | Tractor
          |                  |          |            |
       (done)          Mini/LCV   Axle Config   Trolley relationship

One GoodsTrack in, one Layer2Result out. No video/detector access here --
everything comes from the crops + track metadata layer2_io.py already
loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from layer2_aggregation import aggregate_predictions
from layer2_classifiers import get_stage_classifier, load_crop_image
from layer2_config import Layer2Config
from layer2_models import GoodsTrack, Layer2Result, StagePrediction
from layer2_tractor_trolley import find_attached_trolley

logger = logging.getLogger("mvsa.traffic_layer2")

BROAD_LABELS = ["Goods 3 Wheeler", "LCV", "Heavy Truck", "Tractor"]
LCV_LABELS = ["Mini LCV", "LCV"]
HEAVY_LABELS = ["2 Axle Truck", "3 Axle Truck", "MAV"]
# Order matters -- must match train_stage.py's dataset label assignment
# (index 0 = negative, index 1 = positive_class).
GOODS_3W_DETECTOR_LABELS = ["Not Goods 3 Wheeler", "Goods 3 Wheeler"]


class GoodsClassifierCascade:
    def __init__(self, config: Layer2Config):
        self.config = config
        self.broad = get_stage_classifier("broad", BROAD_LABELS, config.stages["broad"].weights_path, config.project_root)
        self.lcv = get_stage_classifier("lcv", LCV_LABELS, config.stages["lcv"].weights_path, config.project_root)
        self.heavy = get_stage_classifier("heavy_truck", HEAVY_LABELS, config.stages["heavy_truck"].weights_path, config.project_root)

        # Optional trained short-circuit ahead of the broad heuristic --
        # see config/layer2_config.yaml's stages.goods_3_wheeler_detector
        # comment and scripts/traffic_layer2/train_stage.py. Only present
        # once weights_path resolves to a real checkpoint; a
        # HeuristicUniformClassifier fallback here would be actively
        # harmful (coin-flip "detector" overriding the broad classifier),
        # so this stage is skipped entirely rather than falling back.
        self.goods_3w_detector = None
        g3w_cfg = config.stages.get("goods_3_wheeler_detector")
        if g3w_cfg and g3w_cfg.weights_path:
            resolved = config.resolve(g3w_cfg.weights_path)
            if resolved.exists():
                from layer2_classifiers import TorchStageClassifier
                try:
                    self.goods_3w_detector = TorchStageClassifier(resolved, GOODS_3W_DETECTOR_LABELS)
                    self.goods_3w_min_confidence = g3w_cfg.min_confidence
                    logger.info("Loaded trained goods_3_wheeler_detector from %s", resolved)
                except Exception as e:
                    logger.error("Failed to load goods_3_wheeler_detector checkpoint %s: %s", resolved, e)

    def _run_stage(self, stage_classifier, track: GoodsTrack) -> Optional[StagePrediction]:
        n = min(self.config.representative_frames.max_frames, len(track.frames))
        if n < self.config.representative_frames.min_frames:
            return None

        frames = track.frames[:n]
        predictions: List[StagePrediction] = []
        weights: List[float] = []
        source = "heuristic" if stage_classifier.__class__.__name__ == "HeuristicAspectRatioClassifier" else "torch"

        for rf in frames:
            crop = load_crop_image(rf.path)
            if crop is None:
                continue
            probs = stage_classifier.predict_crop(crop, rf.bbox)
            label = max(probs.items(), key=lambda kv: kv[1])[0]
            predictions.append(StagePrediction(label=label, probs=probs, source=source))
            weights.append(max(rf.crop_quality_score, 1e-3))

        if not predictions:
            return None

        label, confidence, averaged = aggregate_predictions(predictions, weights)
        return StagePrediction(label=label, probs=averaged, source=source)

    def classify(
        self,
        track: GoodsTrack,
        candidate_trolley_tracks: List[GoodsTrack],
    ) -> Layer2Result:
        heavy_uncertain_label = self.config.taxonomy.uncertain.heavy_truck_label
        base_kwargs = dict(
            track_id=track.track_id,
            layer1_class=track.layer1_class,
            direction=track.direction,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            best_crop=str(track.best_frame.path) if track.best_frame else None,
        )

        if not track.frames:
            return Layer2Result(
                **base_kwargs,
                layer2_class=heavy_uncertain_label,
                confidence=0.0,
                branch_path=[],
                is_uncertain=True,
                per_stage={"reason": "no representative frames available"},
            )

        if self.goods_3w_detector is not None:
            g3w_pred = self._run_stage(self.goods_3w_detector, track)
            if g3w_pred is not None and g3w_pred.label == "Goods 3 Wheeler" and g3w_pred.confidence >= self.goods_3w_min_confidence:
                return Layer2Result(
                    **base_kwargs,
                    layer2_class="Goods 3 Wheeler",
                    confidence=g3w_pred.confidence,
                    branch_path=["Goods 3 Wheeler (trained detector)"],
                    per_stage={"goods_3_wheeler_detector": {
                        "label": g3w_pred.label, "confidence": round(g3w_pred.confidence, 4), "probs": g3w_pred.probs,
                    }},
                )

        broad_pred = self._run_stage(self.broad, track)
        if broad_pred is None:
            return Layer2Result(
                **base_kwargs,
                layer2_class=heavy_uncertain_label,
                confidence=0.0,
                branch_path=[],
                is_uncertain=True,
                per_stage={"reason": "broad stage produced no usable prediction"},
            )

        per_stage = {"broad": {"label": broad_pred.label, "confidence": round(broad_pred.confidence, 4), "probs": broad_pred.probs}}
        branch_path = [broad_pred.label]

        if broad_pred.label == "Goods 3 Wheeler":
            return Layer2Result(
                **base_kwargs,
                layer2_class="Goods 3 Wheeler",
                confidence=broad_pred.confidence,
                branch_path=branch_path,
                per_stage=per_stage,
            )

        if broad_pred.label == "LCV":
            lcv_pred = self._run_stage(self.lcv, track)
            # Below min_confidence (including the untrained heuristic
            # placeholder, which has no real Mini-LCV-vs-LCV signal --
            # see config/layer2_config.yaml's lcv stage comment), fall
            # back to the broad-level "LCV" label rather than trust a
            # low-confidence subtype split. "LCV" is already one of the 8
            # canonical taxonomy labels (spec section 2), so this is a
            # valid, honest answer -- not a new "uncertain" category.
            if lcv_pred is None or lcv_pred.confidence < self.config.stages["lcv"].min_confidence:
                if lcv_pred:
                    per_stage["lcv"] = {"label": lcv_pred.label, "confidence": round(lcv_pred.confidence, 4), "probs": lcv_pred.probs}
                return Layer2Result(
                    **base_kwargs,
                    layer2_class="LCV",
                    confidence=broad_pred.confidence * (lcv_pred.confidence if lcv_pred else 0.5),
                    branch_path=branch_path,
                    per_stage=per_stage,
                )
            per_stage["lcv"] = {"label": lcv_pred.label, "confidence": round(lcv_pred.confidence, 4), "probs": lcv_pred.probs}
            return Layer2Result(
                **base_kwargs,
                layer2_class=lcv_pred.label,
                confidence=broad_pred.confidence * lcv_pred.confidence,
                branch_path=branch_path + [lcv_pred.label],
                per_stage=per_stage,
            )

        if broad_pred.label == "Heavy Truck":
            usable_frames = min(self.config.representative_frames.max_frames, len(track.frames))
            side_view_frames = sum(
                1 for f in track.frames[:usable_frames]
                if f.aspect_ratio >= self.config.viewpoint.useful_min_aspect_ratio
            )
            viewpoint_ok = side_view_frames >= 1
            heavy_pred = self._run_stage(self.heavy, track)

            insufficient_evidence = (
                heavy_pred is None
                or usable_frames < self.config.representative_frames.min_frames_for_heavy_truck_confidence
                or not viewpoint_ok
                or heavy_pred.confidence < self.config.stages["heavy_truck"].min_confidence
            )
            if insufficient_evidence:
                conf = heavy_pred.confidence if heavy_pred else 0.0
                if heavy_pred:
                    per_stage["heavy_truck"] = {"label": heavy_pred.label, "confidence": round(conf, 4), "probs": heavy_pred.probs}
                per_stage["viewpoint_ok"] = viewpoint_ok
                per_stage["usable_frames"] = usable_frames
                return Layer2Result(
                    **base_kwargs,
                    layer2_class=heavy_uncertain_label,
                    confidence=broad_pred.confidence * conf,
                    branch_path=branch_path,
                    is_uncertain=True,
                    per_stage=per_stage,
                )

            per_stage["heavy_truck"] = {"label": heavy_pred.label, "confidence": round(heavy_pred.confidence, 4), "probs": heavy_pred.probs}
            return Layer2Result(
                **base_kwargs,
                layer2_class=heavy_pred.label,
                confidence=broad_pred.confidence * heavy_pred.confidence,
                branch_path=branch_path + [heavy_pred.label],
                per_stage=per_stage,
            )

        # broad_pred.label == "Tractor"
        trolley_id = find_attached_trolley(track, candidate_trolley_tracks, self.config.tractor_trolley)
        if trolley_id is not None:
            return Layer2Result(
                **base_kwargs,
                layer2_class="Tractor Trolley",
                confidence=broad_pred.confidence,
                branch_path=branch_path + ["Tractor Trolley"],
                trolley_track_id=trolley_id,
                per_stage=per_stage,
            )
        return Layer2Result(
            **base_kwargs,
            layer2_class="Tractor",
            confidence=broad_pred.confidence,
            branch_path=branch_path,  # broad already predicted "Tractor" -- no trolley found, no further split
            per_stage=per_stage,
        )


class BusClassifierCascade:
    """BRTC Bus / City-Private Bus / School Bus -- single-stage cascade
    (no further branching), analogous to GoodsClassifierCascade but for
    Layer 1's "Bus" bucket. See config/layer2_bus_taxonomy.yaml.
    """

    def __init__(self, config: Layer2Config):
        self.config = config
        if config.bus_taxonomy is None:
            raise ValueError("BusClassifierCascade requires config.bus_taxonomy_path to be set")
        self.labels = config.bus_taxonomy.subclasses
        self.uncertain_label = config.bus_taxonomy.uncertain_label
        self.classifier = get_stage_classifier("bus", self.labels, config.stages["bus"].weights_path, config.project_root)

    def _run_stage(self, track: GoodsTrack) -> Optional[StagePrediction]:
        n = min(self.config.representative_frames.max_frames, len(track.frames))
        if n < self.config.representative_frames.min_frames:
            return None
        frames = track.frames[:n]
        predictions: List[StagePrediction] = []
        weights: List[float] = []
        source = "heuristic" if self.classifier.__class__.__name__ in ("HeuristicAspectRatioClassifier", "HeuristicUniformClassifier") else "torch"
        for rf in frames:
            crop = load_crop_image(rf.path)
            if crop is None:
                continue
            probs = self.classifier.predict_crop(crop, rf.bbox)
            label = max(probs.items(), key=lambda kv: kv[1])[0]
            predictions.append(StagePrediction(label=label, probs=probs, source=source))
            weights.append(max(rf.crop_quality_score, 1e-3))
        if not predictions:
            return None
        label, confidence, averaged = aggregate_predictions(predictions, weights)
        return StagePrediction(label=label, probs=averaged, source=source)

    def classify(self, track: GoodsTrack) -> Layer2Result:
        base_kwargs = dict(
            track_id=track.track_id,
            layer1_class=track.layer1_class,
            direction=track.direction,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            best_crop=str(track.best_frame.path) if track.best_frame else None,
        )
        pred = self._run_stage(track)
        min_conf = self.config.stages["bus"].min_confidence
        if pred is None or pred.confidence < min_conf:
            return Layer2Result(
                **base_kwargs,
                layer2_class=self.uncertain_label,
                confidence=pred.confidence if pred else 0.0,
                branch_path=["Bus"],
                is_uncertain=True,
                per_stage={"bus": {"label": pred.label, "confidence": round(pred.confidence, 4), "probs": pred.probs}} if pred else {"reason": "no usable frames"},
            )
        return Layer2Result(
            **base_kwargs,
            layer2_class=pred.label,
            confidence=pred.confidence,
            branch_path=["Bus", pred.label],
            per_stage={"bus": {"label": pred.label, "confidence": round(pred.confidence, 4), "probs": pred.probs}},
        )
