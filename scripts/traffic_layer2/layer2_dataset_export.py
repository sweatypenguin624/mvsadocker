"""Track-based dataset export for manual labeling (spec sections 16-17).

Layer 2 has no ground-truth labels yet -- nothing in this repo has been
manually annotated into goods_dataset/ classes. This module does NOT invent
labels. It exports each Goods Vehicle track's crops into a per-split
staging area (split decided by a deterministic hash of track_id, so a
track's crops never straddle train/val/test -- spec section 16) plus a CSV
manifest a human fills in with the true label. A second pass then sorts
staged crops into the real class folders once labeled.

Usage:
    python scripts/traffic_layer2/classify_goods.py --input <run> --output <run>/layer2 --export-dataset
    # -> edit <dataset_dir>/labeling_manifest.csv, filling in true_label
    python -c "from layer2_dataset_export import apply_labels_from_manifest; \\
        apply_labels_from_manifest(Path('goods_dataset'))"
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import List

from layer2_best_frame import select_best_confidence_frame
from layer2_config import DatasetConfig
from layer2_models import GoodsTrack, Layer2Result

logger = logging.getLogger("mvsa.traffic_layer2")

# Which broad-classifier branch each dataset stage's crops are drawn from.
STAGE_BRANCH_LABEL = {
    "broad": None,  # every Goods Vehicle track
    "lcv": "LCV",
    "heavy": "Heavy Truck",
}


def _slugify_label(label: str) -> str:
    """A class-folder-safe name for any label, including the non-goods
    reclassify labels (tools/goods_review_server/server.py's
    NON_GOODS_RECLASSIFY_LABELS), which contain "/" and ":" -- both of
    which are path separators or otherwise unsafe in a bare directory
    component and must never reach Path() unescaped.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "unlabeled"


def _split_for_track(track_id: int, config: DatasetConfig) -> str:
    """Deterministic (not random) split: hash track_id into [0, 1), same
    result every run so re-exporting is reproducible and idempotent.
    """
    digest = hashlib.sha256(str(track_id).encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    if frac < config.train_fraction:
        return "train"
    if frac < config.train_fraction + config.val_fraction:
        return "val"
    return "test"


MANIFEST_FIELDS = [
    "stage", "track_id", "split", "predicted_label", "crop_path",
    "frame_confidence", "total_frames_available", "true_label",
]


def export_dataset(
    tracks: List[GoodsTrack],
    results: List[Layer2Result],
    output_root: Path,
    config: DatasetConfig,
) -> Path:
    """Exports exactly ONE image per track (unique vehicle): the frame with
    the highest raw detector confidence among everything ByteTrack saw for
    that track (see layer2_best_frame.py). A track is one physical
    vehicle -- ByteTrack already guarantees that identity -- so its other
    frames are near-duplicate crops of the same object, not additional
    training diversity; keeping only the sharpest/most-confident one keeps
    the labeling set small and each label point high quality, per the
    project's explicit "one best frame per vehicle" data policy.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results_by_id = {r.track_id: r for r in results}
    manifest_rows = []
    skipped_no_frame = 0

    for stage_name, required_branch in STAGE_BRANCH_LABEL.items():
        stage_dir = output_root / stage_name
        for track in tracks:
            result = results_by_id.get(track.track_id)
            if result is None:
                continue
            if required_branch is not None and required_branch not in result.branch_path:
                continue

            best = select_best_confidence_frame(track)
            if best is None:
                skipped_no_frame += 1
                continue

            split = _split_for_track(track.track_id, config)
            split_dir = stage_dir / split / "_unlabeled"
            split_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"track_{track.track_id:06d}.jpg"
            shutil.copy2(best.path, split_dir / dest_name)

            manifest_rows.append({
                "stage": stage_name,
                "track_id": track.track_id,
                "split": split,
                "predicted_label": result.layer2_class,
                "crop_path": str((split_dir / dest_name).relative_to(output_root)),
                "frame_confidence": round(best.detector_conf, 4),
                "total_frames_available": len(track.frames),
                "true_label": "",  # for the human labeler to fill in
            })

    manifest_path = output_root / "labeling_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        "Exported %d unique vehicle(s) (1 frame each, highest-confidence) for labeling -> %s (%d skipped: no usable frame)",
        len(manifest_rows), manifest_path, skipped_no_frame,
    )
    return manifest_path


def apply_labels_from_manifest(output_root: Path) -> int:
    """After a human fills in true_label in labeling_manifest.csv, move each
    track's single labeled image from <stage>/<split>/_unlabeled/track_X.jpg
    into <stage>/<split>/<true_label>/track_X.jpg -- the final structure
    spec section 17 describes. Rows with an empty true_label are skipped
    (not yet labeled). Returns how many images were moved.
    """
    output_root = Path(output_root)
    manifest_path = output_root / "labeling_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    moved = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        true_label = (row.get("true_label") or "").strip()
        if not true_label:
            continue
        src = output_root / row["crop_path"]
        if not src.exists():
            continue
        dest = src.parent.parent / _slugify_label(true_label) / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src != dest:
            shutil.move(str(src), str(dest))
            row["crop_path"] = str(dest.relative_to(output_root))
            moved += 1

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Moved %d labeled image(s) into class folders", moved)
    return moved
