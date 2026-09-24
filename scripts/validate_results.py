#!/usr/bin/env python3
"""Validate the batch's results: for every 'done' row in queue/manifest.csv,
check that the engine's expected output files exist and are internally
consistent, and that the Drive-source traceability sidecar
(results/<run_name>/source_manifest.json) still agrees with the manifest.

Writes results/validation/validation_report.csv and .json. Exits non-zero
if any row fails, so this can gate "is the validation batch trustworthy"
decisions rather than requiring a manual read of the CSV.

Usage:
    python scripts/validate_results.py [--manifest queue/manifest.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from batch_utils import PROJECT_ROOT, load_pipeline_config, read_manifest, resolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_results")

REQUIRED_FILES = ["tracks.csv", "hourly_pedestrian_counts.csv", "run_summary.json", "track_positions.json"]


def validate_row(row: dict, results_dir: Path, save_annotated_video: bool) -> dict:
    run_name = row["run_name"]
    run_dir = results_dir / run_name
    checks = {}
    problems = []

    for fname in REQUIRED_FILES:
        p = run_dir / fname
        ok = p.exists() and p.stat().st_size > 0
        checks[fname] = ok
        if not ok:
            problems.append(f"missing/empty {fname}")

    if save_annotated_video:
        p = run_dir / "tracked.mp4"
        ok = p.exists() and p.stat().st_size > 0
        checks["tracked.mp4"] = ok
        if not ok:
            problems.append("missing/empty tracked.mp4")

    summary = None
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"run_summary.json unreadable: {e}")

    hourly_sum = None
    counts_path = run_dir / "hourly_pedestrian_counts.csv"
    if counts_path.exists():
        try:
            with open(counts_path, "r", newline="", encoding="utf-8") as f:
                hourly_sum = sum(int(r["pedestrian_entries"]) for r in csv.DictReader(f))
        except (ValueError, KeyError, OSError) as e:
            problems.append(f"hourly_pedestrian_counts.csv unreadable: {e}")

    if summary is not None and hourly_sum is not None:
        expected = summary.get("total_pedestrian_entries")
        if expected is not None and expected != hourly_sum:
            problems.append(f"hourly_pedestrian_counts.csv sums to {hourly_sum}, run_summary.json says {expected}")

    source_manifest_path = run_dir / "source_manifest.json"
    if not source_manifest_path.exists():
        problems.append("missing source_manifest.json (Drive-source traceability sidecar)")
    else:
        try:
            with open(source_manifest_path, "r", encoding="utf-8") as f:
                src = json.load(f)
            if src.get("drive_path") != row["drive_path"]:
                problems.append(
                    f"source_manifest.json drive_path ({src.get('drive_path')!r}) "
                    f"disagrees with manifest.csv ({row['drive_path']!r})"
                )
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"source_manifest.json unreadable: {e}")

    return {
        "run_name": run_name,
        "drive_path": row["drive_path"],
        "manifest_status": row.get("status", ""),
        "pedestrian_entries": (summary or {}).get("total_pedestrian_entries", ""),
        "passed": len(problems) == 0,
        "problems": "; ".join(problems),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "queue" / "manifest.csv")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.pipeline_config)
    results_dir = resolve(cfg["paths"]["results_dir"])
    save_annotated_video = cfg.get("processing", {}).get("save_annotated_video", True)
    validation_dir = results_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    done_rows = [r for r in rows if r.get("status") == "done"]
    if not done_rows:
        logger.warning("No rows with status=done in %s -- nothing to validate yet", args.manifest)

    reports = [validate_row(r, results_dir, save_annotated_video) for r in done_rows]

    csv_path = validation_dir / "validation_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "drive_path", "manifest_status", "pedestrian_entries", "passed", "problems"])
        writer.writeheader()
        writer.writerows(reports)

    json_path = validation_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    n_pass = sum(1 for r in reports if r["passed"])
    n_fail = len(reports) - n_pass
    logger.info("Validated %d run(s): %d passed, %d failed -> %s", len(reports), n_pass, n_fail, csv_path)
    for r in reports:
        if not r["passed"]:
            logger.warning("FAIL %s: %s", r["run_name"], r["problems"])

    not_done = [r for r in rows if r.get("status") != "done"]
    if not_done:
        logger.info("%d manifest row(s) not yet status=done, excluded from validation: %s", len(not_done), [r["run_name"] for r in not_done])

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
