#!/usr/bin/env python3
"""Layer 2 CLI, Mode A (offline): classify every Goods Vehicle track from a
saved Layer 1 run into the 8-class goods sub-taxonomy.

Usage:
    python scripts/traffic_layer2/classify_goods.py \\
        --input results/layer1_run1 \\
        --output results/layer1_run1/layer2 \\
        [--config config/layer2_config.yaml] \\
        [--export-dataset] [--debug]

Requires the Layer 1 run to have been produced with crops enabled and
"Goods Vehicle" in crops.target_classes (config/layer1_config.yaml), e.g.:
    python scripts/traffic_layer1/run.py ... --save-crops
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from layer2_best_frame import build_unique_vehicle_manifest  # noqa: E402
from layer2_config import Layer2ConfigError, load_layer2_config  # noqa: E402
from layer2_dataset_export import export_dataset  # noqa: E402
from layer2_io import load_goods_tracks, load_all_tracks_by_bucket  # noqa: E402
from layer2_models import GoodsTrack  # noqa: E402
from layer2_output import Layer2OutputWriter  # noqa: E402
from layer2_pipeline import GoodsClassifierCascade  # noqa: E402
from layer2_visualize import write_debug_report  # noqa: E402


def _build_candidate_trolley_tracks(run_dir: Path, config) -> list:
    """GoodsTrack-shaped wrappers for non-Goods-Vehicle tracks in the
    configured candidate buckets (spec section 12-13) -- reuses layer2_io's
    crop loader so trolley candidates get the same bbox/frame data.
    """
    from layer2_io import _detect_source_format, _load_track_crops, LAYER1_CROPS_DIRNAME, VCLASSIFICATION_CROPS_DIRNAME

    source_format = _detect_source_format(run_dir)
    crops_dirname = LAYER1_CROPS_DIRNAME if source_format == "layer1" else VCLASSIFICATION_CROPS_DIRNAME

    by_bucket = load_all_tracks_by_bucket(run_dir)
    candidates = []
    for bucket in config.tractor_trolley.candidate_trolley_buckets:
        for record in by_bucket.get(bucket, []):
            frames = _load_track_crops(run_dir, record["track_id"], crops_dirname)
            if not frames:
                continue
            candidates.append(
                GoodsTrack(
                    track_id=record["track_id"],
                    layer1_class=record["class"],
                    layer1_confidence=float(record.get("confidence", 0.0)),
                    direction=record.get("direction", "indeterminate"),
                    first_seen=record.get("first_seen", ""),
                    last_seen=record.get("last_seen", ""),
                    frames_seen=int(record.get("frames_seen", 0)),
                    best_frame=frames[0],
                    frames=frames,
                )
            )
    return candidates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Layer 1 run directory (contains tracks.jsonl)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: <input>/layer2)")
    parser.add_argument("--config", type=Path, default=Path("config/layer2_config.yaml"))
    parser.add_argument("--export-dataset", action="store_true", help="Also export track crops for manual labeling (spec section 16)")
    parser.add_argument("--debug", action="store_true", help="Write a human-readable debug_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        config = load_layer2_config(args.config)
    except Layer2ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    run_dir = args.input.resolve()
    output_dir = (args.output or (run_dir / config.results_dirname)).resolve()

    try:
        tracks = load_goods_tracks(run_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not tracks:
        print("No Goods Vehicle tracks found in this Layer 1 run -- nothing to classify.")
        return 0

    candidate_trolleys = _build_candidate_trolley_tracks(run_dir, config)

    cascade = GoodsClassifierCascade(config)
    writer = Layer2OutputWriter(output_dir, config)

    results = []
    for track in tracks:
        result = cascade.classify(track, candidate_trolleys)
        results.append(result)
        writer.write_track_result(result, track)

        stage_pred = result.per_stage.get("heavy_truck") or result.per_stage.get("broad")
        if stage_pred and "probs" in stage_pred:
            ranked = sorted(stage_pred["probs"].values(), reverse=True)
            margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
            if margin <= config.review.max_top2_margin:
                writer.write_review_case(result, track, stage_name="broad" if "heavy_truck" not in result.per_stage else "heavy_truck")

    writer.write_tracks_jsonl(results)
    report = writer.write_aggregate_report(results)
    writer.print_summary(report)

    # One row per unique vehicle (ByteTrack track_id) with the single
    # highest-confidence frame chosen for it -- makes the raw-crop-volume
    # vs. unique-vehicle-count distinction explicit (a "save all frames"
    # Layer 1/vehicle-counting run can produce thousands of near-duplicate
    # crops per track; this is the deduplicated view).
    import json as _json
    unique_manifest = build_unique_vehicle_manifest(tracks)
    with open(output_dir / "unique_vehicles.json", "w", encoding="utf-8") as f:
        _json.dump(unique_manifest, f, indent=2)
    total_raw_frames = sum(len(t.frames) for t in tracks)
    print(f"Unique vehicles: {len(tracks)} (from {total_raw_frames} raw saved crop(s) -- see unique_vehicles.json)")

    if args.debug:
        write_debug_report(results, output_dir / "debug_report.txt")
        print(f"Debug report: {output_dir / 'debug_report.txt'}")

    if args.export_dataset:
        dataset_dir = config.resolve(config.dataset.output_dir)
        manifest = export_dataset(tracks, results, dataset_dir, config.dataset)
        print(f"Dataset staged for labeling: {dataset_dir} (manifest: {manifest})")

    print(f"\nLayer 2 results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
