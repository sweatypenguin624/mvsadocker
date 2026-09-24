#!/usr/bin/env python3
"""Batch orchestrator for the VEHICLE-counting pilot: drives
queue/vehicle/manifest.csv through download -> convert ->
detect+track+count (scripts/vehicle_counting/vehicle_counter.py) ->
populate results/{vehicle_counts,vehicle_tracks,vehicle_annotated,
vehicle_summaries}/ -- same shape as scripts/batch_pipeline.py (the
pedestrian batch orchestrator), reusing all its shared helpers, but wired to
the vehicle-counting engine instead.

Two differences from the pedestrian batch pipeline, both because this is a
pilot on a much larger source dataset (CVC's 593GB vs. Peds' 155GB) with no
resume support in the underlying engine:
  - No --resume / --camera-key passthrough (vehicle_counter.py has neither).
  - After a video is successfully converted/processed, its downloaded .dav
    and converted .mp4 are deleted (config: processing.cleanup_after_processing,
    default true) so disk usage stays bounded to roughly one video's worth
    at a time rather than accumulating the whole batch. Results (CSVs/JSON/
    annotated mp4) are unaffected -- they live under results/, not work/.

Never downloads anything not listed in queue/vehicle/manifest.csv (capped at
10-20 videos by create_manifest.py) -- this script cannot accidentally pull
the full CVC dataset.

Usage:
    python scripts/vehicle_batch_pipeline.py run [--manifest queue/vehicle/manifest.csv] [--limit N]
    python scripts/vehicle_batch_pipeline.py run --retry-failed
    python scripts/vehicle_batch_pipeline.py status [--manifest queue/vehicle/manifest.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from batch_utils import (
    FAILED_PREFIX,
    PROJECT_ROOT,
    PerVideoLog,
    link_or_copy,
    load_pipeline_config,
    now_iso,
    read_manifest,
    resolve,
    run_subprocess,
    write_manifest,
    write_source_manifest,
)
from convert_video import convert_one
from download_video import DownloadError, download_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vehicle_batch_pipeline")

ENV_PYTHON = PROJECT_ROOT / "env" / "bin" / "python"
VEHICLE_COUNTER = PROJECT_ROOT / "scripts" / "vehicle_counting" / "vehicle_counter.py"


def _pipeline_log_path(cfg: dict) -> Path:
    return resolve(cfg["paths"]["logs_dir"]) / "vehicle_pipeline.log"


def _batch_log(cfg: dict, message: str) -> None:
    path = _pipeline_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")
    logger.info(message)


def _cleanup(path: Path, label: str, plog) -> None:
    try:
        if path.exists():
            path.unlink()
            plog.write(f"cleanup: removed {label} {path}")
    except OSError as e:
        # Never let a cleanup failure fail the run -- the video is already
        # processed and its results already written; a leftover source file
        # just means disk usage is a bit higher than intended, not data loss.
        plog.write(f"cleanup: failed to remove {label} {path}: {e}")


def process_one(row: dict, cfg: dict) -> dict:
    """Advance one manifest row through download -> convert -> process.
    Mutates and returns the row; never raises -- failures are captured into
    row['status']/row['error'] so the caller can persist progress and move
    on to the next video.
    """
    run_name = row["run_name"]
    logs_dir = resolve(cfg["paths"]["logs_dir"]) / "per_video"
    downloads_dir = resolve(cfg["paths"]["work_downloads"])
    converted_dir = resolve(cfg["paths"]["work_converted"])
    engine_config = resolve(cfg["engine_config"])
    results_dir = resolve(cfg["paths"]["results_dir"])
    cleanup = bool(cfg.get("processing", {}).get("cleanup_after_processing", True))

    dav_path = downloads_dir / f"{run_name}.dav"
    mp4_path = converted_dir / f"{run_name}.mp4"
    run_results_dir = results_dir / run_name

    with PerVideoLog(logs_dir, run_name) as plog:
        plog.write(f"=== processing {run_name} (drive_path={row['drive_path']}) ===")
        try:
            # -- download -----------------------------------------------------
            row["status"] = "downloading"
            plog.write("stage=download start")
            download_one(row["drive_path"], dav_path, cfg, log_write=plog.write)
            row["download_path"] = str(dav_path)
            row["downloaded_at"] = now_iso()
            row["status"] = "downloaded"
            plog.write("stage=download ok")

            # -- convert --------------------------------------------------------
            row["status"] = "converting"
            plog.write("stage=convert start")
            ffmpeg_bin = resolve(cfg["tools"]["ffmpeg"])
            convert_one(dav_path, mp4_path, ffmpeg_bin, log_write=plog.write)
            row["converted_path"] = str(mp4_path)
            row["converted_at"] = now_iso()
            row["status"] = "converted"
            plog.write("stage=convert ok")

            if cleanup:
                # The .dav is never read again once the .mp4 exists (no
                # resume support downstream needs it either).
                _cleanup(dav_path, "source .dav", plog)

            # -- detect + track + count (vehicle-counting engine, own process) --
            row["status"] = "processing"
            plog.write("stage=process start")
            write_source_manifest(run_results_dir, row)

            start_dt = f"{row['date']} {row['hour_start'] or '00:00:00'}"
            cmd = [
                str(ENV_PYTHON),
                str(VEHICLE_COUNTER),
                "--config",
                str(engine_config),
                "--run-name",
                run_name,
                "--video",
                str(mp4_path),
                "--start-datetime",
                start_dt,
            ]

            rc = run_subprocess(cmd, plog.file, cwd=PROJECT_ROOT, timeout=cfg["rclone"].get("timeout_seconds", 3600) * 4)
            if rc != 0:
                raise RuntimeError(f"vehicle_counter.py exited {rc} for {run_name} -- see {plog.path}")

            row["results_dir"] = str(run_results_dir)
            row["processed_at"] = now_iso()
            row["status"] = "done"
            row["error"] = ""
            plog.write("stage=process ok")

            # -- populate the requested results/ taxonomy (hardlinked views
            #    of the engine's canonical results/<run_name>/ output) -------
            _populate_results_taxonomy(results_dir, run_name, run_results_dir, cfg)
            plog.write("results taxonomy populated")

            if cleanup:
                # Safe to drop now: results/<run_name>/ (and its taxonomy
                # hardlinks) hold everything downstream consumers need; the
                # converted source video itself is not a deliverable.
                _cleanup(mp4_path, "converted .mp4", plog)

        except Exception as e:  # noqa: BLE001 -- one bad video must never abort the batch
            stage = row["status"]
            row["status"] = f"{FAILED_PREFIX}{stage}"
            row["error"] = str(e)
            row["retries"] = str(int(row.get("retries") or 0) + 1)
            plog.write(f"FAILED at stage={stage}: {type(e).__name__}: {e}")
            logger.error("run_name=%s failed at %s: %s", run_name, stage, e)

    return row


def _populate_results_taxonomy(results_dir: Path, run_name: str, run_results_dir: Path, cfg: dict) -> None:
    mapping = {
        "vehicle_counts": [run_results_dir / "hourly_vehicle_counts.csv"],
        "vehicle_tracks": [run_results_dir / "vehicle_tracks.csv"],
        "vehicle_summaries": [run_results_dir / "run_summary.json"],
    }
    if cfg.get("processing", {}).get("save_annotated_video", True):
        mapping["vehicle_annotated"] = [run_results_dir / "tracked.mp4"]

    for category, sources in mapping.items():
        for src in sources:
            if not src.exists():
                logger.warning("Expected output missing, skipping taxonomy link: %s", src)
                continue
            dst = results_dir / category / f"{run_name}__{src.stem}{src.suffix}"
            link_or_copy(src, dst)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_pipeline_config(args.pipeline_config)
    rows = read_manifest(args.manifest)
    if not rows:
        logger.error("%s is empty or missing -- run discover_videos.py + create_manifest.py first", args.manifest)
        return 1

    eligible = []
    for row in rows:
        status = row.get("status", "")
        if status == "done":
            continue
        if status.startswith(FAILED_PREFIX) and not args.retry_failed:
            continue
        eligible.append(row)

    if args.limit:
        eligible = eligible[: args.limit]

    if not eligible:
        logger.info("Nothing to do (all rows are 'done', or failures exist but --retry-failed was not passed)")
        return 0

    _batch_log(cfg, f"vehicle batch run starting: {len(eligible)} video(s) eligible (retry_failed={args.retry_failed})")

    by_run_name = {r["run_name"]: r for r in rows}
    n_done = n_failed = 0
    for row in eligible:
        _batch_log(cfg, f"-> {row['run_name']} ({row['drive_path']})")
        updated = process_one(row, cfg)
        by_run_name[updated["run_name"]] = updated
        write_manifest(args.manifest, list(by_run_name.values()))  # persist after every video
        if updated["status"] == "done":
            n_done += 1
            _batch_log(cfg, f"<- {row['run_name']} done")
        else:
            n_failed += 1
            _batch_log(cfg, f"<- {row['run_name']} {updated['status']}: {updated['error']}")

    _batch_log(cfg, f"vehicle batch run finished: {n_done} done, {n_failed} failed this run")
    logger.info("Done: %d succeeded, %d failed this run. See %s for full status.", n_done, n_failed, args.manifest)
    return 0 if n_failed == 0 else 2


def cmd_status(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    if not rows:
        logger.error("%s is empty or missing", args.manifest)
        return 1
    counts: dict = {}
    for row in rows:
        counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
    for status, n in sorted(counts.items()):
        print(f"{status:20s} {n}")
    print(f"{'TOTAL':20s} {len(rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "config" / "vehicle_pipeline.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "queue" / "vehicle" / "manifest.csv")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Process eligible manifest rows")
    p_run.add_argument("--limit", type=int, default=None, help="Process at most N videos this invocation")
    p_run.add_argument("--retry-failed", action="store_true", help="Also (re)attempt rows with a failed_* status")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="Print a status breakdown of the manifest")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
