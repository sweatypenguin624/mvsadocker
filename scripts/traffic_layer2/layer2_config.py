"""Loads config/layer2_config.yaml + config/layer2_goods_taxonomy.yaml into
typed dataclasses. Same one-config-per-pipeline convention as
scripts/traffic_layer1/config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class Layer2ConfigError(ValueError):
    pass


@dataclass
class StageConfig:
    weights_path: Optional[str]
    min_confidence: float


@dataclass
class RepresentativeFramesConfig:
    min_frames: int
    max_frames: int
    min_frames_for_heavy_truck_confidence: int


@dataclass
class ViewpointConfig:
    side_view_min_aspect_ratio: float
    useful_min_aspect_ratio: float


@dataclass
class TractorTrolleyConfig:
    max_link_distance_px: float
    min_persistent_frames: int
    candidate_trolley_buckets: List[str]


@dataclass
class DatasetConfig:
    output_dir: str
    train_fraction: float
    val_fraction: float

    @property
    def test_fraction(self) -> float:
        return max(0.0, 1.0 - self.train_fraction - self.val_fraction)


@dataclass
class ReviewConfig:
    output_dir: str
    max_top2_margin: float


@dataclass
class MavDefinition:
    min_axle_count: int
    min_valid_axle_count: int


@dataclass
class UncertainPolicy:
    heavy_truck_label: str
    exclude_uncertain_from_total: bool


@dataclass
class Layer2Taxonomy:
    subclasses: List[str]
    mav_definition: MavDefinition
    uncertain: UncertainPolicy


@dataclass
class BusTaxonomy:
    subclasses: List[str]
    uncertain_label: str
    exclude_uncertain_from_total: bool


def load_bus_taxonomy(path: Path) -> BusTaxonomy:
    path = Path(path)
    if not path.exists():
        raise Layer2ConfigError(f"bus taxonomy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    subclasses = list(raw.get("subclasses", []))
    if not subclasses:
        raise Layer2ConfigError(f"{path} has no 'subclasses' list")
    unc_raw = raw.get("uncertain", {})
    return BusTaxonomy(
        subclasses=subclasses,
        uncertain_label=str(unc_raw.get("label", "Uncertain Bus Type")),
        exclude_uncertain_from_total=bool(unc_raw.get("exclude_uncertain_from_total", True)),
    )


@dataclass
class Layer2Config:
    project_root: Path
    taxonomy: Layer2Taxonomy
    bus_taxonomy: Optional[BusTaxonomy]
    stages: Dict[str, StageConfig]
    representative_frames: RepresentativeFramesConfig
    viewpoint: ViewpointConfig
    tractor_trolley: TractorTrolleyConfig
    dataset: DatasetConfig
    review: ReviewConfig
    results_dirname: str

    def resolve(self, relative_path: str) -> Path:
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def _load_stage(raw: dict, section: str) -> StageConfig:
    if section not in raw:
        raise Layer2ConfigError(f"layer2_config.yaml missing stages.{section}")
    s = raw[section]
    return StageConfig(
        weights_path=s.get("weights_path"),
        min_confidence=float(s.get("min_confidence", 0.55)),
    )


def load_layer2_taxonomy(path: Path) -> Layer2Taxonomy:
    path = Path(path)
    if not path.exists():
        raise Layer2ConfigError(f"taxonomy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    subclasses = list(raw.get("subclasses", []))
    if not subclasses:
        raise Layer2ConfigError(f"{path} has no 'subclasses' list")
    mav_raw = raw.get("mav_definition", {})
    mav = MavDefinition(
        min_axle_count=int(mav_raw.get("min_axle_count", 4)),
        min_valid_axle_count=int(mav_raw.get("min_valid_axle_count", 2)),
    )
    unc_raw = raw.get("uncertain", {})
    uncertain = UncertainPolicy(
        heavy_truck_label=str(unc_raw.get("heavy_truck_label", "Uncertain Heavy Truck")),
        exclude_uncertain_from_total=bool(unc_raw.get("exclude_uncertain_from_total", True)),
    )
    return Layer2Taxonomy(subclasses=subclasses, mav_definition=mav, uncertain=uncertain)


def load_layer2_config(config_path: Path) -> Layer2Config:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise Layer2ConfigError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise Layer2ConfigError("layer2_config.yaml did not parse into a mapping")

    project_root = config_path.parent.parent

    taxonomy_path = project_root / raw.get("taxonomy_path", "config/layer2_goods_taxonomy.yaml")
    taxonomy = load_layer2_taxonomy(taxonomy_path)

    bus_taxonomy = None
    if raw.get("bus_taxonomy_path"):
        bus_taxonomy = load_bus_taxonomy(project_root / raw["bus_taxonomy_path"])

    stages_raw = raw.get("stages", {})
    stages = {
        "broad": _load_stage(stages_raw, "broad"),
        "lcv": _load_stage(stages_raw, "lcv"),
        "heavy_truck": _load_stage(stages_raw, "heavy_truck"),
    }
    if "goods_3_wheeler_detector" in stages_raw:
        stages["goods_3_wheeler_detector"] = _load_stage(stages_raw, "goods_3_wheeler_detector")
    if "bus" in stages_raw:
        stages["bus"] = _load_stage(stages_raw, "bus")

    rf_raw = raw.get("representative_frames", {})
    representative_frames = RepresentativeFramesConfig(
        min_frames=int(rf_raw.get("min_frames", 1)),
        max_frames=int(rf_raw.get("max_frames", 5)),
        min_frames_for_heavy_truck_confidence=int(rf_raw.get("min_frames_for_heavy_truck_confidence", 2)),
    )

    vp_raw = raw.get("viewpoint", {})
    viewpoint = ViewpointConfig(
        side_view_min_aspect_ratio=float(vp_raw.get("side_view_min_aspect_ratio", 1.6)),
        useful_min_aspect_ratio=float(vp_raw.get("useful_min_aspect_ratio", 1.15)),
    )

    tt_raw = raw.get("tractor_trolley", {})
    tractor_trolley = TractorTrolleyConfig(
        max_link_distance_px=float(tt_raw.get("max_link_distance_px", 220)),
        min_persistent_frames=int(tt_raw.get("min_persistent_frames", 2)),
        candidate_trolley_buckets=list(tt_raw.get("candidate_trolley_buckets", ["Others"])),
    )

    ds_raw = raw.get("dataset", {})
    dataset = DatasetConfig(
        output_dir=str(ds_raw.get("output_dir", "goods_dataset")),
        train_fraction=float(ds_raw.get("train_fraction", 0.7)),
        val_fraction=float(ds_raw.get("val_fraction", 0.15)),
    )

    rv_raw = raw.get("review", {})
    review = ReviewConfig(
        output_dir=str(rv_raw.get("output_dir", "review")),
        max_top2_margin=float(rv_raw.get("max_top2_margin", 0.15)),
    )

    out_raw = raw.get("output", {})
    results_dirname = str(out_raw.get("results_dirname", "layer2"))

    return Layer2Config(
        project_root=project_root,
        taxonomy=taxonomy,
        bus_taxonomy=bus_taxonomy,
        stages=stages,
        representative_frames=representative_frames,
        viewpoint=viewpoint,
        tractor_trolley=tractor_trolley,
        dataset=dataset,
        review=review,
        results_dirname=results_dirname,
    )
