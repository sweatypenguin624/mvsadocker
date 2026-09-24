"""Writes Layer 2 results (spec sections 21-22): per-track JSON,
tracks_layer2.jsonl, aggregate goods-count report with sum verification,
and hard-example review crops.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

from layer2_config import Layer2Config
from layer2_models import GoodsTrack, Layer2Result

logger = logging.getLogger("mvsa.traffic_layer2")


class Layer2OutputWriter:
    def __init__(self, output_dir: Path, config: Layer2Config):
        self.output_dir = Path(output_dir)
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "goods_tracks").mkdir(parents=True, exist_ok=True)

    def write_track_result(self, result: Layer2Result, track: GoodsTrack) -> None:
        track_dir = self.output_dir / "goods_tracks" / f"{result.track_id:06d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        with open(track_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        if track.best_frame is not None:
            try:
                shutil.copy2(track.best_frame.path, track_dir / "best.jpg")
            except OSError as e:
                logger.warning("track_id=%s: could not copy best crop: %s", result.track_id, e)

    def write_tracks_jsonl(self, results: List[Layer2Result]) -> Path:
        path = self.output_dir / "tracks_layer2.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
        logger.info("Wrote %s", path)
        return path

    def write_review_case(self, result: Layer2Result, track: GoodsTrack, stage_name: str) -> None:
        review_dir = self.config.resolve(self.config.review.output_dir) / stage_name
        review_dir.mkdir(parents=True, exist_ok=True)
        if track.best_frame is not None:
            try:
                shutil.copy2(track.best_frame.path, review_dir / f"track_{result.track_id:06d}.jpg")
            except OSError as e:
                logger.warning("track_id=%s: could not copy review crop: %s", result.track_id, e)
        meta_path = review_dir / f"track_{result.track_id:06d}.json"
        meta_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    def write_aggregate_report(self, results: List[Layer2Result]) -> dict:
        subclass_counts: Dict[str, int] = {c: 0 for c in self.config.taxonomy.subclasses}
        uncertain_label = self.config.taxonomy.uncertain.heavy_truck_label
        uncertain_count = 0

        for r in results:
            if r.is_uncertain:
                uncertain_count += 1
                continue
            if r.layer2_class in subclass_counts:
                subclass_counts[r.layer2_class] += 1
            else:
                logger.warning("track_id=%s produced unexpected label %r, counting as uncertain", r.track_id, r.layer2_class)
                uncertain_count += 1

        subclass_total = sum(subclass_counts.values())
        exclude_uncertain = self.config.taxonomy.uncertain.exclude_uncertain_from_total
        total_goods_vehicles = subclass_total if exclude_uncertain else subclass_total + uncertain_count

        report = {
            "counts": subclass_counts,
            "uncertain_count": uncertain_count,
            "uncertain_label": uncertain_label,
            "uncertain_excluded_from_total": exclude_uncertain,
            "total_goods_vehicles": total_goods_vehicles,
            "total_tracks_processed": len(results),
            "sum_check_ok": (subclass_total == total_goods_vehicles) if exclude_uncertain else (subclass_total + uncertain_count == total_goods_vehicles),
        }

        path = self.output_dir / "aggregate_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote %s", path)
        return report

    def print_summary(self, report: dict) -> None:
        print("\nGoods Vehicle Subclassification")
        print("-" * 34)
        for cls, count in report["counts"].items():
            print(f"{cls:<20} {count:>6}")
        print(f"{report['uncertain_label']:<20} {report['uncertain_count']:>6}")
        print("-" * 34)
        print(f"{'Total Goods Vehicles':<20} {report['total_goods_vehicles']:>6}")
        if not report["sum_check_ok"]:
            print("WARNING: subclass sum does not reconcile with total_goods_vehicles -- see aggregate_report.json")
