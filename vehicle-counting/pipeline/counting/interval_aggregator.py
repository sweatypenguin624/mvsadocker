#!/usr/bin/env python3
"""
Dynamic Interval Aggregator for Vehicle Counting.

Aggregates counting records into fixed-duration intervals (e.g., 15 minutes)
without hardcoding any vehicle classes or applying any class grouping.
All classes detected by the model are preserved dynamically.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def get_bucket_label(bucket_idx: int, start_dt: datetime, interval_minutes: int) -> str:
    start_time = start_dt + timedelta(minutes=interval_minutes * bucket_idx)
    end_time = start_time + timedelta(minutes=interval_minutes)
    return f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"


def aggregate_intervals(
    results_dir: Path,
    fps: float,
    start_time_str: str = "09:00:00",
    interval_minutes: int = 15,
    known_classes: Optional[Sequence[str]] = None,
    total_frames: Optional[int] = None,
    records: Optional[List[dict]] = None,
) -> Path:
    """
    Aggregates counting events into time intervals.
    Prioritizes in-memory records, then records.csv, then tracks.jsonl.
    """
    results_dir = Path(results_dir)
    start_dt = datetime.strptime(start_time_str, "%H:%M:%S")

    # Class set preservation
    class_set = list(known_classes) if known_classes else []
    seen_classes = set(class_set)

    # Raw counts: bucket_idx -> class -> count
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    max_frame = 0

    if records is not None and len(records) > 0:
        # Use provided in-memory records
        for r in records:
            cls = r.get("class", "Unknown")
            frame = int(r.get("frame", 0))
            if frame > max_frame:
                max_frame = frame
            if cls not in seen_classes:
                class_set.append(cls)
                seen_classes.add(cls)
            t_sec = frame / fps
            bucket_idx = int(t_sec // (interval_minutes * 60))
            counts[bucket_idx][cls] += 1
    elif (results_dir / "records.csv").exists():
        # Read from records.csv (exact line crossing records)
        with open(results_dir / "records.csv", mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row["class"]
                frame = int(row["timestamp_or_frame"])
                if frame > max_frame:
                    max_frame = frame
                if cls not in seen_classes:
                    class_set.append(cls)
                    seen_classes.add(cls)
                t_sec = frame / fps
                bucket_idx = int(t_sec // (interval_minutes * 60))
                counts[bucket_idx][cls] += int(row.get("count", 1))
    elif (results_dir / "tracks.jsonl").exists():
        # Fallback to tracks.jsonl
        with open(results_dir / "tracks.jsonl", mode="r") as f:
            for line in f:
                d = json.loads(line)
                is_counted = d.get("counted", d.get("crossed", False))
                if not is_counted or d.get("not_counted_reason"):
                    continue
                cls = d.get("class", "Unknown")
                if cls not in seen_classes:
                    class_set.append(cls)
                    seen_classes.add(cls)
                cross_frame = d.get("crossing_frame")
                frame = cross_frame if (cross_frame is not None and cross_frame > 0) else d.get("first_frame", 0)
                if frame > max_frame:
                    max_frame = frame
                t_sec = frame / fps
                bucket_idx = int(t_sec // (interval_minutes * 60))
                counts[bucket_idx][cls] += 1
    else:
        raise FileNotFoundError(f"No records or records.csv or tracks.jsonl found in {results_dir}")

    # Determine total buckets to display continuously
    ref_frames = total_frames if (total_frames is not None and total_frames > 0) else max_frame
    total_duration_sec = (ref_frames / fps) if fps > 0 else 0
    num_buckets = max(1, math.ceil(total_duration_sec / (interval_minutes * 60)))
    if counts:
        num_buckets = max(num_buckets, max(counts.keys()) + 1)

    # Ensure consistent column ordering: known_classes first, then any newly seen
    ordered_classes = list(class_set)

    out_csv_path = results_dir / "interval_counts.csv"
    with open(out_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["Interval"] + ordered_classes + ["Total"]
        writer.writerow(header)

        for b_idx in range(num_buckets):
            row_label = get_bucket_label(b_idx, start_dt, interval_minutes)
            row_counts = counts[b_idx]
            total_interval = sum(row_counts.get(cls, 0) for cls in ordered_classes)
            row = [row_label] + [row_counts.get(cls, 0) for cls in ordered_classes] + [total_interval]
            writer.writerow(row)

    print(f"\n[INTERVAL SUMMARY] Wrote 15-min interval counts to: {out_csv_path}")
    header_str = f"{'Interval':<25} | " + " | ".join(f"{cls[:10]:>10}" for cls in ordered_classes) + f" | {'Total':>8}"
    print(header_str)
    print("-" * len(header_str))
    for b_idx in range(num_buckets):
        row_label = get_bucket_label(b_idx, start_dt, interval_minutes)
        row_counts = counts[b_idx]
        total_interval = sum(row_counts.get(cls, 0) for cls in ordered_classes)
        cells = " | ".join(f"{row_counts.get(cls, 0):>10}" for cls in ordered_classes)
        print(f"{row_label:<25} | {cells} | {total_interval:>8}")
    print()

    return out_csv_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate counting results into time intervals without class grouping.")
    parser.add_argument("results_dir", type=str, help="Directory containing records.csv or tracks.jsonl")
    parser.add_argument("--fps", type=float, required=True, help="Video native FPS")
    parser.add_argument("--start", type=str, default="09:00:00", help="Video start time in HH:MM:SS format")
    parser.add_argument("--interval-min", type=int, default=15, help="Interval duration in minutes (default: 15)")
    args = parser.parse_args()

    aggregate_intervals(
        results_dir=Path(args.results_dir),
        fps=args.fps,
        start_time_str=args.start,
        interval_minutes=args.interval_min,
    )


if __name__ == "__main__":
    main()
