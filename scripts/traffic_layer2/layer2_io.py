"""Reads one track-producing pipeline's run directory and produces the
GoodsTrack objects Layer 2 operates on. Never re-opens the source video or
re-runs detection (spec section 23: "do not rerun detection unnecessarily").

Two source pipelines are supported, auto-detected from tracks.jsonl's own
record shape (each pipeline's crop directory name and field names differ):

  - Layer 1 (scripts/traffic_layer1/) -- broad-class taxonomy, one
    "Goods Vehicle" bucket, crops under bus_tracks/track_NNNNNN/,
    first_seen/last_seen as ISO datetimes.
  - This repo's own ByteTrack-based vehicle-counting engine
    (vehicle-counting/pipeline/counting/) -- fine-grained UVH-26 classes
    directly (Truck/LCV/tempo-traveller = the Goods-Vehicle-shaped ones),
    crops under goods_tracks/track_NNNNNN/, first_frame/last_frame as
    integer frame indices (no wall-clock datetime available without a
    --start-datetime the pipeline doesn't currently accept).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from layer2_models import GoodsTrack, RepresentativeFrame

logger = logging.getLogger("mvsa.traffic_layer2")

LAYER1_CROPS_DIRNAME = "bus_tracks"  # historical name, see traffic_layer1/crops.py
VCLASSIFICATION_CROPS_DIRNAME = "goods_tracks"  # see vehicle-counting/pipeline/counting/goods_crops.py

# UVH-26 native classes the vehicle-counting engine can emit that are
# Goods-Vehicle-shaped -- kept in sync by hand with
# config/layer1_class_mapping.yaml's uvh26 section and
# vehicle-counting/pipeline/counting/goods_crops.py::GOODS_VEHICLE_NATIVE_CLASSES.
VCLASSIFICATION_GOODS_CLASSES = {"Truck", "LCV", "tempo-traveller"}


def _detect_source_format(run_dir: Path) -> str:
    """Returns "layer1" or "vclassification" by peeking at tracks.jsonl's
    first record shape, or by which crop directory exists as a fallback.
    """
    tracks_path = run_dir / "tracks.jsonl"
    with open(tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "first_frame" in record:
                return "vclassification"
            if "first_seen" in record:
                return "layer1"
            break
    if (run_dir / VCLASSIFICATION_CROPS_DIRNAME).exists():
        return "vclassification"
    return "layer1"


def _load_track_crops(run_dir: Path, track_id: int, crops_dirname: str) -> List[RepresentativeFrame]:
    track_dir = run_dir / crops_dirname / f"track_{track_id:06d}"
    metadata_path = track_dir / "metadata.json"
    if not metadata_path.exists():
        return []

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    frames = []
    for entry in metadata.get("crops", []):
        crop_path = track_dir / entry["file"]
        if not crop_path.exists():
            continue
        frames.append(
            RepresentativeFrame(
                file=entry["file"],
                path=crop_path,
                frame_idx=int(entry.get("frame_idx", -1)),
                crop_quality_score=float(entry.get("score", 0.0)),
                bbox=[float(v) for v in entry.get("bbox", [0.0, 0.0, 0.0, 0.0])],
                detector_conf=float(entry.get("conf", 0.0)),
            )
        )
    # Both pipelines' crop writers already write these best-first.
    frames.sort(key=lambda f: f.crop_quality_score, reverse=True)
    return frames


def load_goods_tracks(run_dir: Path, target_class: str = "Goods Vehicle") -> List[GoodsTrack]:
    """Load every counted Goods-Vehicle-shaped track from run_dir, joined
    with its saved representative crops. Tracks with no saved crops are
    still returned (so callers can report "0 usable frames" honestly) but
    Layer 2 classification cannot proceed on them.
    """
    run_dir = Path(run_dir)
    tracks_path = run_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"{tracks_path} not found -- run Layer 1 (scripts/traffic_layer1/run.py) or the "
            f"vehicle-counting engine (vehicle-counting/pipeline/counting/main.py) first"
        )

    source_format = _detect_source_format(run_dir)
    crops_dirname = LAYER1_CROPS_DIRNAME if source_format == "layer1" else VCLASSIFICATION_CROPS_DIRNAME
    logger.info("Detected source format=%s (crops under %s/)", source_format, crops_dirname)

    goods_tracks: List[GoodsTrack] = []
    with open(tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if source_format == "layer1":
                if record.get("class") != target_class:
                    continue
                first_seen, last_seen = record.get("first_seen", ""), record.get("last_seen", "")
            else:
                if record.get("class") not in VCLASSIFICATION_GOODS_CLASSES:
                    continue
                first_seen = f"frame:{record.get('first_frame', -1)}"
                last_seen = f"frame:{record.get('last_frame', -1)}"

            frames = _load_track_crops(run_dir, record["track_id"], crops_dirname)

            goods_tracks.append(
                GoodsTrack(
                    track_id=record["track_id"],
                    layer1_class=record.get("class", target_class),
                    layer1_confidence=float(record.get("confidence", 0.0)),
                    direction=record.get("direction", "indeterminate"),
                    first_seen=first_seen,
                    last_seen=last_seen,
                    frames_seen=int(record.get("frames_seen", 0)),
                    best_frame=frames[0] if frames else None,
                    frames=frames,
                )
            )

    logger.info("Loaded %d Goods Vehicle track(s) from %s", len(goods_tracks), tracks_path)
    return goods_tracks


def load_all_vehicle_tracks(run_dir: Path) -> List[GoodsTrack]:
    """Every counted track, any class -- for a run whose crop-saving was
    widened to all vehicle classes (not just Goods-Vehicle-shaped), used by
    classify_vehicles.py to route Bus/Goods/passthrough per track. Returned
    as GoodsTrack objects (the name predates this generalization but the
    shape -- track_id/class/crops -- is class-agnostic).
    """
    run_dir = Path(run_dir)
    tracks_path = run_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"{tracks_path} not found")

    source_format = _detect_source_format(run_dir)
    crops_dirname = LAYER1_CROPS_DIRNAME if source_format == "layer1" else VCLASSIFICATION_CROPS_DIRNAME
    logger.info("Detected source format=%s (crops under %s/)", source_format, crops_dirname)

    tracks: List[GoodsTrack] = []
    with open(tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if source_format == "layer1":
                first_seen, last_seen = record.get("first_seen", ""), record.get("last_seen", "")
            else:
                first_seen = f"frame:{record.get('first_frame', -1)}"
                last_seen = f"frame:{record.get('last_frame', -1)}"

            frames = _load_track_crops(run_dir, record["track_id"], crops_dirname)
            tracks.append(
                GoodsTrack(
                    track_id=record["track_id"],
                    layer1_class=record.get("class", "Unknown"),
                    layer1_confidence=float(record.get("confidence", 0.0)),
                    direction=record.get("direction", "indeterminate"),
                    first_seen=first_seen,
                    last_seen=last_seen,
                    frames_seen=int(record.get("frames_seen", 0)),
                    best_frame=frames[0] if frames else None,
                    frames=frames,
                )
            )

    logger.info("Loaded %d track(s) (all classes) from %s", len(tracks), tracks_path)
    return tracks


def load_all_tracks_by_bucket(run_dir: Path) -> Dict[str, List[dict]]:
    """Every counted track (any class), grouped by broad bucket -- used by
    layer2_tractor_trolley.py to find trolley candidate tracks. Only
    meaningful for Layer 1 runs (broad "Others" bucket exists); a
    vclassification-source run returns its native fine-grained classes
    as-is, which candidate_trolley_buckets in config/layer2_config.yaml
    should be adjusted for if used against that source.
    """
    run_dir = Path(run_dir)
    tracks_path = run_dir / "tracks.jsonl"
    by_bucket: Dict[str, List[dict]] = {}
    with open(tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_bucket.setdefault(record.get("class", "Others"), []).append(record)
    return by_bucket
