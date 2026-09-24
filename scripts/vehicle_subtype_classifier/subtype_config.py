"""Config loader for the vehicle subtype classifier (Step 1).

Prefixed ``subtype_`` (not ``config.py``) to avoid the flat-import basename
collision documented in the Layer 1 pipeline memory -- scripts/config.py
and scripts/traffic_layer1/config.py already exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelConfig:
    path: Path
    device: str = "cuda:0"
    confidence: float = 0.25
    imgsz: int = 640
    iou: float = 0.45
    agnostic_nms: bool = True


@dataclass
class OutputConfig:
    results_filename: str = "subtype_results.jsonl"
    summary_filename: str = "subtype_summary.json"


@dataclass
class SubtypeConfig:
    model: ModelConfig
    expected_classes: List[str] = field(default_factory=list)
    no_detection_label: str = "Unclassified"
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Path) -> SubtypeConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    model_raw = raw["model"]
    model_path = Path(model_raw["path"])
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    return SubtypeConfig(
        model=ModelConfig(
            path=model_path,
            device=model_raw.get("device", "cuda:0"),
            confidence=float(model_raw.get("confidence", 0.25)),
            imgsz=int(model_raw.get("imgsz", 640)),
            iou=float(model_raw.get("iou", 0.45)),
            agnostic_nms=bool(model_raw.get("agnostic_nms", True)),
        ),
        expected_classes=list(raw.get("expected_classes", [])),
        no_detection_label=raw.get("no_detection_label", "Unclassified"),
        output=OutputConfig(**raw.get("output", {})),
    )
