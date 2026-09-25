#!/usr/bin/env python3
"""End-to-end, resumable batch pipeline for one folder of camera videos.

    python historical-processor/pipeline.py "TVC 5" [--dry-run] [--map "Cam 3=tvc5_cam1_eb"] ...

SOURCE may be a local directory, a Google Drive folder ID, a Drive path, or
just a folder name to search for on Drive (e.g. "tvc-5"). Every camera
subfolder is detected and matched to a vehicle-counting config; videos are
downloaded a few at a time, remuxed, counted, and cleaned up. Progress lives
in a SQLite manifest, so a crash or restart resumes where it left off.

Layout:
    pipeline_runs/<run>/   manifest.db, plan.json, status.txt, work/ (temp videos)
    results/<run>/<camera>/<date>/<video>/   per-video outputs (+ stdout.log)
    results/<run>/all_interval_counts*.csv   merged across every video
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HIST_DIR = Path(__file__).resolve().parent
ROOT = HIST_DIR.parent
sys.path.insert(0, str(HIST_DIR))

from ingest.manifest import IngestManifest  # noqa: E402
from ingest.downloader import Downloader  # noqa: E402
from orchestrator import finish_run, RunTimer, RUNS_LOG_DIR, safe_name  # noqa: E402

RCLONE = "tools/rclone-v1.75.0-linux-amd64/rclone"
RCLONE_CONF = "config/rclone.conf"
FFMPEG = "tools/ffmpeg_bin/ffmpeg"
CONFIG_DIR = Path("vehicle-counting/config")
CONFIG_PREFIX = "vehicle_count_config_"
VIDEO_EXTS = {".dav", ".mp4", ".mkv", ".avi", ".mov"}
IN_FLIGHT = ("QUEUED", "DOWNLOADING", "DOWNLOADED", "PROCESSING")

DRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{25,}$")
DATE_DIR_RE = re.compile(r"^(\d{4}[-_.]?\d{2}[-_.]?\d{2}|\d{2}[-_.]\d{2}[-_.]\d{4})$")
DAV_TIME_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{2})-")
STAMP_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
CAM_TOKEN_RE = re.compile(r"^cam\d+$")
DIRECTION_TOKENS = {"nb", "sb", "eb", "wb", "ir"}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{threading.current_thread().name}] {msg}", flush=True)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "main"


def tokens(s: str) -> set[str]:
    """'TVC 5 - Cam_1 EB' -> {'tvc5', 'cam1', 'eb'}. Letters join a number only when
    adjacent or split by one space/_/- ('TVC - 7 Days' stays 'tvc', '7', 'days')."""
    out = set()
    for m in re.finditer(r"([a-z]+)[ _-]?(\d+)|([a-z]+)|(\d+)", s.lower()):
        if m.group(1):
            out.add(m.group(1) + str(int(m.group(2))))
        elif m.group(3):
            out.add(m.group(3))
        else:
            out.add(str(int(m.group(4))))
    return out


def rclone(*args, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run([RCLONE, "--config", RCLONE_CONF, *args],
                          capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- source

def resolve_source(src: str, remote: str, shared: bool) -> dict:
    if os.path.isdir(src):
        root = os.path.abspath(src)
        return {"kind": "local", "root": root, "name": os.path.basename(root.rstrip("/"))}
    base = remote + (",shared_with_me" if shared else "")
    if "/" not in src and DRIVE_ID_RE.match(src):
        return {"kind": "drive", "remote": f"{base},root_folder_id={src}", "path": "", "name": src}

    path = src.strip("/")
    if rclone("lsf", "--dirs-only", "--max-depth", "1", f"{base}:{path}", timeout=600).returncode == 0:
        return {"kind": "drive", "remote": base, "path": path, "name": path.split("/")[-1]}

    log(f"'{src}' is not an exact Drive path; searching Drive folders by name (can take a few minutes)...")
    r = rclone("lsf", "-R", "--dirs-only", "--fast-list", "--max-depth", "6", f"{base}:", timeout=3600)
    if r.returncode != 0:
        sys.exit(f"Drive folder search failed:\n{r.stderr[-2000:]}")
    want = tokens(src)
    hits = [d.rstrip("/") for d in r.stdout.splitlines() if want and want <= tokens(d.rstrip("/").split("/")[-1])]
    # A hit nested inside another hit is a camera/date subfolder of it; keep the top one.
    top = [h for h in hits if not any(h != o and h.startswith(o + "/") for o in hits)]
    if len(top) != 1:
        listing = "\n  ".join(top) or "(none)"
        sys.exit(f"Could not pick a single Drive folder for '{src}'. Matches:\n  {listing}\n"
                 f"Re-run with the exact path in quotes.")
    log(f"Resolved '{src}' -> {top[0]}")
    return {"kind": "drive", "remote": base, "path": top[0], "name": top[0].split("/")[-1]}


def parse_video(rel: str) -> dict:
    parts = rel.split("/")
    dirs, name = parts[:-1], parts[-1]
    date = None
    while dirs and DATE_DIR_RE.match(dirs[-1]):
        date = date or dirs[-1]
        dirs.pop()
    stem = Path(name).stem
    start = None
    if m := DAV_TIME_RE.match(stem):
        start = ":".join(m.groups())
    elif m := STAMP_RE.search(stem):
        y, mo, d, h, mi, s = m.groups()
        date = date or f"{y}-{mo}-{d}"
        start = f"{h}:{mi}:{s}"
    return {"camera": "/".join(dirs) or ".", "date": date or "unknown", "start_time": start or ""}


def scan(source: dict, manifest: IngestManifest) -> int:
    found = []
    if source["kind"] == "local":
        for dirpath, _, files in os.walk(source["root"]):
            for f in files:
                full = os.path.join(dirpath, f)
                found.append((full, os.path.relpath(full, source["root"]).replace(os.sep, "/")))
    else:
        target = f"{source['remote']}:{source['path']}"
        log(f"Listing {target} ...")
        r = rclone("lsjson", "-R", "--files-only", "--fast-list", "--tpslimit", "4", "--tpslimit-burst", "4",
                   target, timeout=3600)
        if r.returncode != 0:
            sys.exit(f"rclone listing failed:\n{r.stderr[-2000:]}")
        for f in json.loads(r.stdout):
            found.append((f.get("ID") or f["Path"], f["Path"]))

    # An already-remuxed copy (X.mp4 / X.dav.mp4 next to X.dav) would be counted twice.
    davs = {rel[:-4].lower() for _, rel in found if rel.lower().endswith(".dav")}
    added = skipped_dupes = 0
    for file_id, rel in found:
        if Path(rel).suffix.lower() not in VIDEO_EXTS or "backup" in rel.lower():
            continue
        low = rel.lower()
        if low.endswith(".mp4") and (low[:-4] in davs or low[:-8] in davs and low.endswith(".dav.mp4")):
            skipped_dupes += 1
            continue
        info = parse_video(rel)
        manifest.add_or_update(file_id=file_id, gdrive_path=rel, filename=rel.split("/")[-1],
                               date=info["date"], start_time=info["start_time"], camera=info["camera"])
        added += 1
    log(f"Scan found {added} videos ({len(found)} files listed, {skipped_dupes} remuxed duplicates of .dav skipped).")
    return added


# ---------------------------------------------------------------- config plan

def config_path(value: str) -> str:
    p = Path(value)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return str(p)
    cand = CONFIG_DIR / f"{CONFIG_PREFIX}{value}.yaml"
    if cand.exists():
        return str(cand)
    sys.exit(f"Config not found: {value} (tried {p} and {cand})")


def auto_config(run_name: str, camera: str, configs: dict) -> tuple:
    have = tokens(f"{run_name} {'' if camera == '.' else camera}")
    per_camera = {p: t for p, t in configs.items() if any(CAM_TOKEN_RE.match(x) for x in t)}
    for strip_direction in (False, True):
        cands = [(p, t) for p, t in per_camera.items()
                 if ((t - DIRECTION_TOKENS) if strip_direction else t) <= have]
        if cands:
            best = max(len(t) for _, t in cands)
            top = sorted(p for p, t in cands if len(t) == best)
            if len(top) == 1:
                return top[0], "auto"
            return None, f"ambiguous: {', '.join(top)}"
    return None, "no matching config"


def build_plan(run_name: str, manifest: IngestManifest, maps: dict, default_cfg: str | None) -> dict:
    configs = {str(p): tokens(p.stem[len(CONFIG_PREFIX):]) for p in sorted(CONFIG_DIR.glob(f"{CONFIG_PREFIX}*.yaml"))}
    rows = manifest.get_all()
    cams: dict[str, dict] = {}
    for r in rows:
        c = cams.setdefault(r["camera"] or ".", {"videos": 0, "dates": set(), "no_start_time": 0})
        c["videos"] += 1
        c["dates"].add(r["date"])
        c["no_start_time"] += not r["start_time"]
    plan = {}
    for cam, c in sorted(cams.items()):
        mapped = maps.get(cam) or maps.get(cam.split("/")[-1])
        if mapped:
            cfg, how = config_path(mapped), "--map"
        elif default_cfg:
            cfg, how = config_path(default_cfg), "--config"
        else:
            cfg, how = auto_config(run_name, cam, configs)
        dates = sorted(c["dates"])
        plan[cam] = {"config": cfg, "how": how, "slug": slug(cam), "videos": c["videos"],
                     "dates": f"{dates[0]} .. {dates[-1]} ({len(dates)} days)",
                     "no_start_time": c["no_start_time"]}
    return plan


def print_plan(plan: dict) -> bool:
    ok = True
    print("\n================ PLAN ================")
    for cam, p in plan.items():
        print(f"camera '{cam}': {p['videos']} videos, {p['dates']}")
        if p["config"]:
            print(f"    config: {p['config']}  ({p['how']})")
        else:
            ok = False
            print(f"    config: MISSING -- {p['how']}")
        if p["no_start_time"]:
            print(f"    WARNING: {p['no_start_time']} videos have no start time in their filename; "
                  f"their intervals will use the config's video_start_time")
    if not ok:
        print("\nMap the missing cameras, e.g.:\n"
              "    --map \"Cam 3=tvc5_cam1_eb\" --map \"Cam 4=tvc5_cam2_wb\"\n"
              "or apply one config to all cameras with --config <name|path>.")
    print("======================================\n")
    return ok


# ---------------------------------------------------------------- processing

def run_logged(cmd: list, log_path: Path, timeout_s: float, env=None) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} $ {' '.join(map(str, cmd))}\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_s / 60:.0f} min"
    return r.returncode == 0, f"exit code {r.returncode}"


class Pipeline:
    def __init__(self, args, source: dict, plan: dict, run_name: str, run_dir: Path, manifest: IngestManifest):
        self.args, self.source, self.plan, self.manifest = args, source, plan, manifest
        self.run_name = run_name
        self.work_dir = run_dir / "work"
        self.results_dir = ROOT / "results" / run_name
        self.status_path = run_dir / "status.txt"
        self.session_dir = RUNS_LOG_DIR / safe_name(run_name) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.dl_q: queue.Queue = queue.Queue()
        self.inf_q: queue.Queue = queue.Queue(maxsize=max(1, args.ready_files))
        self.downloader = Downloader(remote=source["remote"]) if source["kind"] == "drive" else None
        self.completed_this_session = 0
        self.started = time.time()

    # -- helpers
    def out_dir(self, item) -> Path:
        return self.results_dir / self.plan[item["camera"]]["slug"] / slug(item["date"]) / safe_name(Path(item["filename"]).stem)

    def fail(self, item, error: str) -> None:
        log(f"FAILED {item['camera']}/{item['filename']}: {error}")
        self.manifest.update_status(item["file_id"], "FAILED", error=error)
        self.cleanup(item)
        if "timer" in item:
            finish_run(item["timer"], self.session_dir, self.run_name, "FAILED", error)

    def cleanup(self, item) -> None:
        for p in item.get("temp_files", []):
            try:
                os.remove(p)
            except OSError:
                pass

    def wait_for_disk(self) -> None:
        warned = 0.0
        while shutil.disk_usage(self.work_dir).free / 1e9 < self.args.min_free_gb:
            if time.time() - warned > 300:
                log(f"Low disk space (<{self.args.min_free_gb} GB free on {self.work_dir}); waiting...")
                warned = time.time()
            time.sleep(30)

    # -- download + remux worker
    def download_worker(self) -> None:
        while (item := self.dl_q.get()) is not None:
            try:
                self.prepare(item)
            except Exception as e:
                self.fail(item, f"unexpected error in download/remux: {e!r}")
            finally:
                self.dl_q.task_done()

    def prepare(self, item) -> None:
        a = self.args
        rel, name = item["gdrive_path"], item["filename"]
        timer = RunTimer()
        timer.meta.update(camera=item["camera"], file=name, date=item["date"], source_path=rel)
        item["timer"], item["temp_files"] = timer, []
        local_dir = self.work_dir / self.plan[item["camera"]]["slug"] / slug(item["date"])
        local_dir.mkdir(parents=True, exist_ok=True)

        if self.source["kind"] == "local":
            video = os.path.join(self.source["root"], rel)
        else:
            self.wait_for_disk()
            self.manifest.update_status(item["file_id"], "DOWNLOADING")
            video = str(local_dir / name)
            remote_path = f"{self.source['path']}/{rel}" if self.source["path"] else rel
            with timer.stage("download"):
                ok = self.downloader.download_file(remote_path, video, timeout=a.download_timeout_min * 60)
            if not ok:
                return self.fail(item, "download failed")
            item["temp_files"].append(video)
            mb = os.path.getsize(video) / 1e6
            timer.meta["dav_size_mb"] = round(mb, 2)
            timer.meta["download_speed_mb_s"] = round(mb / max(timer.seconds("download"), 1e-6), 2)

        if Path(name).suffix.lower() == ".dav":
            mp4 = str(local_dir / (name + ".mp4"))
            item["temp_files"].append(mp4)
            with timer.stage("remux"):
                ok, why = run_logged([FFMPEG, "-y", "-i", video, "-c", "copy", mp4],
                                     self.work_dir / "remux.log", a.remux_timeout_min * 60)
            if not ok:
                return self.fail(item, f"remux failed ({why})")
            if video in item["temp_files"]:  # the downloaded .dav is no longer needed
                self.cleanup({"temp_files": [video]})
                item["temp_files"].remove(video)
            timer.meta["mp4_size_mb"] = round(os.path.getsize(mp4) / 1e6, 2)
            video = mp4

        item["video"] = video
        self.manifest.update_status(item["file_id"], "DOWNLOADED")
        item["handoff_at"] = time.perf_counter()
        self.inf_q.put(item)  # blocks while enough videos are already waiting: bounded disk use

    # -- inference worker
    def inference_worker(self) -> None:
        while (item := self.inf_q.get()) is not None:
            try:
                self.infer(item)
            except Exception as e:
                self.fail(item, f"unexpected error in inference: {e!r}")
            finally:
                self.inf_q.task_done()

    def infer(self, item) -> None:
        timer = item["timer"]
        timer.add("queue_wait_for_inference", time.perf_counter() - item["handoff_at"])
        self.manifest.update_status(item["file_id"], "PROCESSING")
        out = self.out_dir(item)
        out.mkdir(parents=True, exist_ok=True)
        timing_path = out / "timing.json"
        timing_path.unlink(missing_ok=True)

        cmd = [sys.executable, "vehicle-counting/pipeline/counting/main.py",
               "--video", item["video"], "--output_dir", str(out), "--no_annotation",
               "--config", self.plan[item["camera"]]["config"]]
        if item["start_time"]:
            cmd += ["--start_time", item["start_time"]]
        log(f"inference start {item['camera']}/{item['date']}/{item['filename']}")
        with timer.stage("inference"):
            ok, why = run_logged(cmd, out / "stdout.log", self.args.inference_timeout_min * 60,
                                 env={**os.environ, "MVSA_ORCHESTRATED": "1"})
        if timing_path.exists():
            try:
                timer.children["inference"] = json.loads(timing_path.read_text())
            except (OSError, ValueError):
                pass
        with timer.stage("cleanup"):
            self.cleanup(item)
        if ok:
            self.manifest.update_status(item["file_id"], "COMPLETED")
            self.completed_this_session += 1
            log(f"DONE {item['camera']}/{item['date']}/{item['filename']} in {timer.seconds('inference'):.0f}s")
            finish_run(timer, self.session_dir, self.run_name, "OK")
        else:
            self.fail(item, f"inference failed ({why}); see {out / 'stdout.log'}")

    # -- main loop
    def heartbeat(self) -> None:
        s = self.manifest.get_stats()
        done, total = s.get("COMPLETED", 0), s.get("TOTAL", 0)
        elapsed = time.time() - self.started
        rate = self.completed_this_session / elapsed if elapsed > 0 else 0
        remaining = self.manifest.count_pending(self.args.retries) + sum(s.get(k, 0) for k in IN_FLIGHT)
        eta = f"{remaining / rate / 3600:.1f} h" if rate > 0 else "n/a"
        line = (f"{done}/{total} completed | failed {s.get('FAILED', 0)} | in flight "
                f"{sum(s.get(k, 0) for k in IN_FLIGHT)} | {rate * 3600:.1f} videos/h | ETA {eta}")
        log(line)
        self.status_path.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n{json.dumps(s, indent=2)}\n")

    def run(self) -> None:
        a = self.args
        threads = [threading.Thread(target=self.download_worker, name=f"DL-{i}", daemon=True)
                   for i in range(a.dl_threads)]
        threads += [threading.Thread(target=self.inference_worker, name=f"INF-{i}", daemon=True)
                    for i in range(a.inf_threads)]
        for t in threads:
            t.start()
        log(f"Started {a.dl_threads} download and {a.inf_threads} inference workers; "
            f"results -> {self.results_dir}; timing logs -> {self.session_dir}")

        last_beat = 0.0
        while True:
            if self.dl_q.qsize() < a.dl_threads:
                for item in self.manifest.get_pending_batch(a.dl_threads, a.retries):
                    self.manifest.update_status(item["file_id"], "QUEUED")
                    self.dl_q.put(item)
            if time.time() - last_beat > 300:
                self.heartbeat()
                last_beat = time.time()
            s = self.manifest.get_stats()
            if self.manifest.count_pending(a.retries) == 0 and not any(s.get(k, 0) for k in IN_FLIGHT):
                break
            time.sleep(2)

        for _ in range(a.dl_threads):
            self.dl_q.put(None)
        for _ in range(a.inf_threads):
            self.inf_q.put(None)
        for t in threads:
            t.join(timeout=60)
        self.heartbeat()


# ---------------------------------------------------------------- results

def merge_results(run_name: str, plan: dict, manifest: IngestManifest) -> None:
    results = ROOT / "results" / run_name
    merged: dict[str, dict[str, list]] = {}
    summary = []
    for r in manifest.get_all():
        cam = r["camera"] or "."
        p = plan.get(cam, {"slug": slug(cam)})
        out = results / p["slug"] / slug(r["date"]) / safe_name(Path(r["filename"]).stem)
        counted = ""
        if (out / "timing.json").exists():
            try:
                counted = json.loads((out / "timing.json").read_text())["meta"].get("total_counted", "")
            except (OSError, ValueError, KeyError):
                pass
        summary.append({"camera": cam, "date": r["date"], "video": r["filename"], "start_time": r["start_time"],
                        "status": r["status"], "total_counted": counted,
                        "error": "" if r["status"] == "COMPLETED" else (r["error"] or ""),
                        "output_dir": str(out.relative_to(ROOT))})
        if r["status"] != "COMPLETED":
            continue
        for fname in ("interval_counts.csv", "interval_counts_3class.csv"):
            f = out / fname
            if f.exists():
                with open(f, newline="") as fh:
                    for row in csv.DictReader(fh):
                        merged.setdefault(fname, {}).setdefault(cam, []).append(
                            {"camera": cam, "date": r["date"], "video": r["filename"], **row})

    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "run_summary.csv", summary)
    for fname, by_cam in merged.items():
        everything = []
        for cam, rows in by_cam.items():
            _write_csv(results / plan.get(cam, {"slug": slug(cam)})["slug"] / f"all_{fname}", rows)
            everything += rows
        _write_csv(results / f"all_{fname}", everything)
    log(f"Merged results written under {results}")


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    header = []
    for r in rows:
        header += [k for k in r if k not in header]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Local dir, Drive folder ID, Drive path, or folder name to search (e.g. 'tvc-5')")
    ap.add_argument("--name", help="Run name (default: derived from the folder name)")
    ap.add_argument("--map", action="append", default=[], metavar="CAMERA=CONFIG",
                    help="Config for a camera subfolder; CONFIG is a path or short name like tvc5_cam1_eb")
    ap.add_argument("--config", help="Use this config for every camera without a --map")
    ap.add_argument("--dry-run", action="store_true", help="Scan and print the plan, then exit")
    ap.add_argument("--rescan", action="store_true", help="Re-list the source to pick up new videos")
    ap.add_argument("--retry-failed", action="store_true", help="Give permanently failed videos another try")
    ap.add_argument("--merge-only", action="store_true", help="Only rebuild merged result CSVs")
    ap.add_argument("--remote", default="Gdrive-yogesh", help="rclone remote name")
    ap.add_argument("--shared-with-me", action="store_true", help="Search/list Drive 'Shared with me'")
    ap.add_argument("--dl-threads", type=int, default=2)
    ap.add_argument("--inf-threads", type=int, default=2)
    ap.add_argument("--ready-files", type=int, default=2,
                    help="Max prepared videos waiting for inference (bounds disk use)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--min-free-gb", type=float, default=30)
    ap.add_argument("--download-timeout-min", type=float, default=120)
    ap.add_argument("--remux-timeout-min", type=float, default=30)
    ap.add_argument("--inference-timeout-min", type=float, default=240)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    threading.current_thread().name = "MAIN"

    # Resolution is cached per source so restarts don't repeat the Drive search.
    cache = ROOT / "pipeline_runs" / "_sources.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{args.remote}|{args.shared_with_me}|{args.source}"
    source = known.get(key) or resolve_source(args.source, args.remote, args.shared_with_me)
    known[key] = source
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, indent=2))

    run_name = slug(args.name or source["name"])
    run_dir = ROOT / "pipeline_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = IngestManifest(db_path=str(run_dir / "manifest.db"))
    log(f"Run '{run_name}' | source: {source}")

    if args.rescan or manifest.get_stats().get("TOTAL", 0) == 0:
        if scan(source, manifest) == 0:
            log("No videos found.")
            return 2

    maps = {}
    for m in args.map:
        cam, sep, cfg = m.partition("=")
        if not sep:
            sys.exit(f"--map must be CAMERA=CONFIG, got: {m}")
        maps[cam.strip()] = cfg.strip()
    plan = build_plan(source["name"], manifest, maps, args.config)
    ok = print_plan(plan)
    (run_dir / "plan.json").write_text(json.dumps({"source": source, "cameras": plan}, indent=2))
    if args.merge_only:
        merge_results(run_name, plan, manifest)
        return 0
    if not ok:
        return 2
    if args.dry_run:
        return 0

    if args.retry_failed:
        manifest.reset_retries()
    for status in IN_FLIGHT:  # left over from a crash/restart
        for item in manifest.get_by_status(status):
            manifest.update_status(item["file_id"], "DISCOVERED")
    shutil.rmtree(run_dir / "work", ignore_errors=True)
    (run_dir / "work").mkdir()

    Pipeline(args, source, plan, run_name, run_dir, manifest).run()
    merge_results(run_name, plan, manifest)
    s = manifest.get_stats()
    log(f"ALL DONE: {s.get('COMPLETED', 0)}/{s.get('TOTAL', 0)} completed, {s.get('FAILED', 0)} failed "
        f"(see results/{run_name}/run_summary.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
