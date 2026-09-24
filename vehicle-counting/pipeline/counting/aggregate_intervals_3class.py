#!/usr/bin/env python3
"""Aggregate a run's tracks.jsonl into 15-min two-wheeler/three-wheeler/car counts.

Usage: aggregate_intervals_3class.py <results_dir> [--fps FPS] [--start HH:MM:SS] [--interval-min N]
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

CAR_CLASSES = {"Hatchback", "Sedan", "SUV", "MUV", "Van"}
TWO_WHEELER_CLASSES = {"Two-wheeler"}
THREE_WHEELER_CLASSES = {"Three-wheeler"}


def bucket_label(bucket_idx, start_dt, interval_min):
    s = start_dt + timedelta(minutes=interval_min * bucket_idx)
    e = s + timedelta(minutes=interval_min)
    return f"{s.strftime('%H:%M:%S')} - {e.strftime('%H:%M:%S')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--fps", type=float, required=True, help="native video fps (frame indices in tracks.jsonl are native frame numbers)")
    ap.add_argument("--start", type=str, required=True, help="video start time HH:MM:SS")
    ap.add_argument("--interval-min", type=int, default=15)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    tracks_path = results_dir / "tracks.jsonl"
    start_dt = datetime.strptime(args.start, "%H:%M:%S")

    counts = defaultdict(lambda: {"Two-wheeler": 0, "Three-wheeler": 0, "Car": 0})

    with open(tracks_path) as f:
        for line in f:
            d = json.loads(line)
            counted = d.get("counted", d.get("crossed", False))
            if not counted:
                continue
            if d.get("not_counted_reason"):
                continue
            cls = d["class"]
            if cls in TWO_WHEELER_CLASSES:
                cat = "Two-wheeler"
            elif cls in THREE_WHEELER_CLASSES:
                cat = "Three-wheeler"
            elif cls in CAR_CLASSES:
                cat = "Car"
            else:
                continue
            cross_frame = d.get("crossing_frame")
            frame = cross_frame if cross_frame and cross_frame > 0 else d.get("first_frame", d.get("first_seen_frame", 0))
            t_sec = frame / args.fps
            bucket_idx = int(t_sec // (args.interval_min * 60))
            counts[bucket_idx][cat] += 1

    out_path = results_dir / "interval_counts_3class.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Interval", "Two-wheeler", "Three-wheeler", "Car", "Total"])
        for idx in sorted(counts.keys()):
            row = counts[idx]
            total = row["Two-wheeler"] + row["Three-wheeler"] + row["Car"]
            w.writerow([bucket_label(idx, start_dt, args.interval_min), row["Two-wheeler"], row["Three-wheeler"], row["Car"], total])

    print(f"Wrote {out_path}")
    for idx in sorted(counts.keys()):
        row = counts[idx]
        print(bucket_label(idx, start_dt, args.interval_min), dict(row))


if __name__ == "__main__":
    main()
