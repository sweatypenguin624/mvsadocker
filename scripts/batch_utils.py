"""Shared helpers for the multi-video batch/orchestration layer.

Nothing in scripts/detector.py, tracker.py, pipeline.py, or config_loader.py
(the existing single-video engine) is imported or modified here -- this
module only supports the batch scripts (discover_videos.py, create_manifest.py,
download_video.py, convert_video.py, pedestrian_counter.py, batch_pipeline.py,
validate_results.py) that select, fetch, and orchestrate runs of that engine.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATES_FIELDS = [
    "drive_path",
    "site",
    "camera",
    "date",
    "hour_start",
    "hour_end",
    "filename",
    "size_bytes",
    "discovered_at",
]

MANIFEST_FIELDS = [
    "run_name",
    "drive_path",
    "site",
    "camera",
    "date",
    "hour_start",
    "hour_end",
    "size_bytes",
    "status",
    "retries",
    "download_path",
    "converted_path",
    "results_dir",
    "error",
    "added_at",
    "downloaded_at",
    "converted_at",
    "processed_at",
]

# Terminal "good" state for a manifest row.
STATUS_DONE = "done"
STATUS_PENDING = "pending"
# Failure states are prefixed "failed_" so validate_results.py / operators
# can tell at a glance which stage broke.
FAILED_PREFIX = "failed_"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(rel_or_abs: str) -> Path:
    """Resolve a path from config/pipeline.yaml against the project root."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def load_pipeline_config(path: Optional[Path] = None) -> dict:
    path = path or (PROJECT_ROOT / "config" / "pipeline.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not parse into a mapping")
    return cfg


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def derive_run_name(site: str, camera: str, date: str, hour_start: str) -> str:
    """Deterministic, human-readable, filesystem-safe run identifier.

    Derived only from metadata (never from the possibly-duplicated bare
    filename) so that two videos sharing the same filename in different
    Drive folders never collide -- see the project's "never identify a
    video only by its filename" requirement.
    """
    hh = hour_start.replace(":", "")[:4] if hour_start else "unknown"
    date_compact = date.replace("-", "") if date else "unknowndate"
    return f"{slugify(site)}_{slugify(camera)}_{date_compact}_{hh}"


# -- CSV I/O -------------------------------------------------------------


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fields: List[str], rows: List[Dict[str, str]]) -> None:
    """Atomic write: write to a temp file in the same directory, then
    os.replace, so a crash mid-write never leaves a truncated/corrupt CSV
    (manifest.csv in particular is read-modify-written after every video).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    os.replace(tmp_path, path)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(path)


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    write_csv_rows(path, MANIFEST_FIELDS, rows)


def read_candidates(path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(path)


def write_candidates(path: Path, rows: List[Dict[str, str]]) -> None:
    write_csv_rows(path, CANDIDATES_FIELDS, rows)


# -- filesystem helpers ----------------------------------------------------


def link_or_copy(src: Path, dst: Path) -> None:
    """Populate the results/{detections,tracks,counts,annotated,summaries}
    taxonomy from the engine's canonical results/<run_name>/ output without
    doubling disk usage for large files (tracked.mp4 can be multiple GB).
    Falls back to a real copy if hardlinking isn't possible (e.g. dst on a
    different filesystem).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run_subprocess(cmd: List[str], log_file, cwd: Optional[Path] = None, timeout: Optional[int] = None) -> int:
    """Run a subprocess, streaming combined stdout/stderr into an
    already-open file handle. Returns the exit code.
    """
    log_file.write(f"\n$ {' '.join(cmd)}\n")
    log_file.flush()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode


class PerVideoLog:
    """Append-only per-video orchestration log at logs/per_video/<run_name>.log.

    Distinct from the engine's own results/<run_name>/logs/run.log (which
    records frame-by-frame detection/tracking detail); this one records
    batch-level events -- download/convert/process start & end, retries,
    and errors -- plus captures the pedestrian_counter.py subprocess's raw
    stdout/stderr.
    """

    def __init__(self, logs_dir: Path, run_name: str):
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"{run_name}.log"
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self._fh.write(f"{now_iso()} {message}\n")
        self._fh.flush()

    @property
    def file(self):
        return self._fh

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "PerVideoLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def write_source_manifest(results_run_dir: Path, row: Dict[str, str]) -> None:
    """Sidecar written into results/<run_name>/ itself (not just
    queue/manifest.csv) so the mapping from a result back to its exact
    Google Drive source path survives even if manifest.csv is ever lost --
    see the project requirement to never lose track of which Drive source
    generated which result.
    """
    results_run_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: row.get(k, "") for k in MANIFEST_FIELDS if k in ("run_name", "drive_path", "site", "camera", "date", "hour_start", "hour_end", "size_bytes")}
    with open(results_run_dir / "source_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
