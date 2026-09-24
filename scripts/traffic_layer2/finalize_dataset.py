#!/usr/bin/env python3
"""Finalize a full annotation pass: every track gets a ground-truth label
-- the human correction if one exists in corrections.jsonl, otherwise the
model's own Layer 2 prediction (explicit user policy: "the ones who are not
marked are by default correct marked by the model"). Exports one image per
track (its highest-confidence frame, see layer2_best_frame.py) into a
flat, track-based train/val/test split.

Usage:
    python scripts/traffic_layer2/finalize_dataset.py \\
        --run-dir results/vc_all_vehicles_full \\
        --output vehicle_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from layer2_best_frame import select_best_confidence_frame  # noqa: E402
from layer2_dataset_export import _split_for_track, _slugify_label  # noqa: E402
from layer2_io import load_all_vehicle_tracks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finalize_dataset")


class _SplitCfg:
    """Matches layer2_dataset_export.DatasetConfig's shape without
    requiring a full Layer2Config load."""
    def __init__(self, train_fraction=0.7, val_fraction=0.15):
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction


def load_predictions(layer2_dir: Path) -> dict:
    predictions = {}
    with open(layer2_dir / "tracks_layer2.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            predictions[rec["track_id"]] = rec["layer_2_class"]
    return predictions


def load_corrections(layer2_dir: Path) -> dict:
    latest = {}
    corrections_path = layer2_dir / "corrections.jsonl"
    if not corrections_path.exists():
        return latest
    with open(corrections_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[rec["track_id"]] = rec["corrected_label"]
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. results/vc_all_vehicles_full")
    parser.add_argument("--layer2-dir", type=Path, default=None, help="default: <run-dir>/layer2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    layer2_dir = (args.layer2_dir or (run_dir / "layer2")).resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(layer2_dir)
    corrections = load_corrections(layer2_dir)
    tracks = load_all_vehicle_tracks(run_dir)
    split_cfg = _SplitCfg()

    manifest_rows = []
    skipped_no_frame = 0
    n_corrected = 0
    n_model_default = 0

    for track in tracks:
        predicted = predictions.get(track.track_id)
        if predicted is None:
            continue
        corrected = corrections.get(track.track_id)
        final_label = corrected if corrected is not None else predicted
        source = "human_corrected" if corrected is not None else "model_default"
        if corrected is not None:
            n_corrected += 1
        else:
            n_model_default += 1

        best = select_best_confidence_frame(track)
        if best is None:
            skipped_no_frame += 1
            continue

        split = _split_for_track(track.track_id, split_cfg)
        class_dir = output_dir / split / _slugify_label(final_label)
        class_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"track_{track.track_id:06d}.jpg"
        shutil.copy2(best.path, class_dir / dest_name)

        manifest_rows.append({
            "track_id": track.track_id,
            "native_class": track.layer1_class,
            "predicted_label": predicted,
            "final_label": final_label,
            "label_source": source,
            "split": split,
            "frame_confidence": round(best.detector_conf, 4),
            "image_path": str((class_dir / dest_name).relative_to(output_dir)),
        })

    manifest_path = output_dir / "final_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "native_class", "predicted_label", "final_label",
            "label_source", "split", "frame_confidence", "image_path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info("Wrote %s", manifest_path)
    logger.info(
        "%d tracks: %d human-corrected, %d model-default, %d skipped (no usable frame)",
        len(manifest_rows), n_corrected, n_model_default, skipped_no_frame,
    )

    # Group by the same slug used for the on-disk folder name, not the raw
    # label string -- otherwise a case-only difference (e.g. the corrected
    # "Two-Wheeler" vs. the native passthrough "Two-wheeler") looks like
    # two separate classes here even though they already share one folder.
    class_counts = {}
    for row in manifest_rows:
        slug = _slugify_label(row["final_label"])
        entry = class_counts.setdefault(slug, {"label": row["final_label"], "count": 0})
        entry["count"] += 1
    print("\nFinal label distribution (human-corrected + model-default combined):")
    print("-" * 50)
    for entry in sorted(class_counts.values(), key=lambda e: -e["count"]):
        print(f"{entry['label']:<32} {entry['count']:>5}")
    print("-" * 50)
    print(f"{'TOTAL':<32} {len(manifest_rows):>5}")
    print(f"\n  human-corrected: {n_corrected}   model-default: {n_model_default}")


if __name__ == "__main__":
    main()
