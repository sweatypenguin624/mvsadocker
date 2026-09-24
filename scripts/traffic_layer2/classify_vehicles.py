#!/usr/bin/env python3
"""Layer 2 CLI, all-vehicle mode: classify EVERY track from a saved
vehicle-counting/Layer 1 run, not just Goods Vehicle ones.

Routing per track's native class:
  - Truck / LCV / tempo-traveller  -> GoodsClassifierCascade (existing
    Goods 3 Wheeler / LCV / Heavy Truck / Tractor cascade, now with the
    trained goods_3_wheeler_detector short-circuit if configured)
  - Bus / Mini-bus                 -> BusClassifierCascade (BRTC / City-
    Private / School Bus)
  - everything else (Two-wheeler, Three-wheeler [passenger], Car-shaped
    classes, bicycle, ...) -> passthrough: UVH-26 already classifies these
    finely enough: layer2_class = the native class, unchanged.

Requires the source run's crop-saving to have been widened to all classes
(vehicle-counting/config/vehicle_count_config.yaml's goods_crops.all_classes:
true), otherwise non-Goods/non-Bus tracks pass through with 0 crops (still
reported, just with no image to review).

Usage:
    python scripts/traffic_layer2/classify_vehicles.py \\
        --input results/<run> --output results/<run>/layer2 [--debug]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from layer2_config import Layer2ConfigError, load_layer2_config  # noqa: E402
from layer2_io import load_all_vehicle_tracks  # noqa: E402
from layer2_models import Layer2Result  # noqa: E402
from layer2_output import Layer2OutputWriter  # noqa: E402
from layer2_pipeline import BusClassifierCascade, GoodsClassifierCascade  # noqa: E402
from layer2_visualize import write_debug_report  # noqa: E402
from classify_goods import _build_candidate_trolley_tracks  # noqa: E402

logger = logging.getLogger("mvsa.traffic_layer2")

GOODS_NATIVE_CLASSES = {"Truck", "LCV", "tempo-traveller"}
BUS_NATIVE_CLASSES = {"Bus", "Mini-bus"}


def _passthrough_result(track) -> Layer2Result:
    return Layer2Result(
        track_id=track.track_id,
        layer1_class=track.layer1_class,
        layer2_class=track.layer1_class,
        confidence=track.layer1_confidence,
        direction=track.direction,
        first_seen=track.first_seen,
        last_seen=track.last_seen,
        best_crop=str(track.best_frame.path) if track.best_frame else None,
        branch_path=[track.layer1_class],
        per_stage={"reason": "passthrough -- UVH-26 native class used as-is, no Layer 2 subclassification defined for this class"},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/layer2_config.yaml"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        config = load_layer2_config(args.config)
    except Layer2ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    run_dir = args.input.resolve()
    output_dir = (args.output or (run_dir / config.results_dirname)).resolve()

    all_tracks = load_all_vehicle_tracks(run_dir)
    if not all_tracks:
        print("No tracks found in this run.")
        return 0

    goods_cascade = GoodsClassifierCascade(config)
    bus_cascade = BusClassifierCascade(config) if config.bus_taxonomy else None
    candidate_trolleys = _build_candidate_trolley_tracks(run_dir, config)

    writer = Layer2OutputWriter(output_dir, config)
    results = []
    counts_by_native_class = {}

    for track in all_tracks:
        native = track.layer1_class
        counts_by_native_class[native] = counts_by_native_class.get(native, 0) + 1

        if native in GOODS_NATIVE_CLASSES:
            result = goods_cascade.classify(track, candidate_trolleys)
        elif native in BUS_NATIVE_CLASSES and bus_cascade is not None:
            result = bus_cascade.classify(track)
        else:
            result = _passthrough_result(track)

        results.append(result)
        writer.write_track_result(result, track)

    writer.write_tracks_jsonl(results)

    # Aggregate report across everything: goods subclasses + bus subclasses
    # + passthrough native-class counts, all in one place.
    # write_aggregate_report only knows the 8 Goods Vehicle subclasses --
    # feeding it Bus/passthrough results would count every non-Goods track
    # as "uncertain" (they don't match any Goods subclass label). Filter
    # to Goods-native-class results only; Bus and passthrough get their
    # own breakdowns printed below instead.
    goods_results = [r for r in results if r.layer1_class in GOODS_NATIVE_CLASSES]
    goods_report = writer.write_aggregate_report(goods_results)
    print("\nGoods Vehicle Subclassification")
    print("-" * 34)
    for cls, count in goods_report["counts"].items():
        print(f"{cls:<28} {count:>4}")
    print(f"{goods_report['uncertain_label']:<28} {goods_report['uncertain_count']:>4}")

    bus_counts = {}
    if config.bus_taxonomy:
        bus_counts = {c: 0 for c in config.bus_taxonomy.subclasses}
        bus_counts[config.bus_taxonomy.uncertain_label] = 0
    passthrough_counts = {}
    for r in results:
        if r.layer1_class in GOODS_NATIVE_CLASSES:
            continue
        if r.layer1_class in BUS_NATIVE_CLASSES and config.bus_taxonomy:
            bus_counts[r.layer2_class] = bus_counts.get(r.layer2_class, 0) + 1
        elif r.layer1_class not in BUS_NATIVE_CLASSES:
            passthrough_counts[r.layer2_class] = passthrough_counts.get(r.layer2_class, 0) + 1

    print("\nBus Subclassification")
    print("-" * 34)
    for cls, count in bus_counts.items():
        print(f"{cls:<28} {count:>4}")

    print("\nOther Vehicle Classes (passthrough, native UVH-26 class)")
    print("-" * 34)
    for cls, count in sorted(passthrough_counts.items(), key=lambda kv: -kv[1]):
        print(f"{cls:<28} {count:>4}")

    print(f"\nTotal tracks: {len(results)}  (native class breakdown: {counts_by_native_class})")

    if args.debug:
        write_debug_report(results, output_dir / "debug_report.txt")
        print(f"Debug report: {output_dir / 'debug_report.txt'}")

    print(f"\nLayer 2 (all-vehicle) results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
