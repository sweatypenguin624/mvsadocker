import sys
import os
import json
from datetime import datetime
from pathlib import Path
import subprocess
import time
import queue
import threading
import argparse

# Ensure historical-processor is on sys.path
HIST_DIR = Path(__file__).resolve().parent
if str(HIST_DIR) not in sys.path:
    sys.path.insert(0, str(HIST_DIR))
# Appended (not inserted) so scripts/ never shadows historical-processor modules.
SCRIPTS_DIR = HIST_DIR.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from run_timer import RunTimer, write_report, append_summary_csv, safe_name, RUNS_LOG_DIR

# Always use project virtual environment python if it exists
ENV_PYTHON = "/home/users/oauser/mvsa/env/bin/python"
if not os.path.exists(ENV_PYTHON):
    ENV_PYTHON = sys.executable

from ingest.manifest import IngestManifest
from ingest.downloader import Downloader
from ingest.drive_scanner import DriveScanner

SUMMARY_STAGES = {
    "download_s": "download",
    "remux_s": "remux",
    "queue_wait_s": "queue_wait_for_inference",
    "inference_s": "inference",
    "cleanup_s": "cleanup",
}
SUMMARY_INFERENCE_STAGES = {
    "inf_imports_s": "startup.imports",
    "inf_model_load_s": "setup.model_load",
    "inf_video_probe_s": "setup.video_probe",
    "inf_frame_loop_s": "frame_loop",
    "inf_decode_s": "frame_loop.decode",
    "inf_detect_track_s": "frame_loop.detect_track",
    "inf_yolo_inference_s": "frame_loop.detect_track.yolo_inference",
    "inf_track_update_s": "frame_loop.track_update",
    "inf_count_classify_s": "frame_loop.count_and_classify",
    "inf_annotate_s": "frame_loop.annotate",
    "inf_intervals_s": "output.interval_counts",
}
SUMMARY_META = ["dav_size_mb", "download_speed_mb_s", "download_skipped_existing", "mp4_size_mb"]
SUMMARY_INFERENCE_META = ["video_duration_s", "frames_decoded", "frames_processed", "decode_fps",
                          "loop_processed_fps", "total_counted"]


def finish_run(timer, session_dir, camera_name, status, error=None):
    try:
        _finish_run(timer, session_dir, camera_name, status, error)
    except Exception as e:
        print(f"[{threading.current_thread().name}] WARNING: failed to write timing log: {e}")


def _finish_run(timer, session_dir, camera_name, status, error=None):
    """Write the per-file timing log and append one row to the session summary.csv."""
    if error:
        timer.meta["error"] = error
    data = timer.to_dict(status)
    camera = timer.meta.get("camera", camera_name)
    name = (f"{safe_name(camera)}__{timer.meta.get('date', 'nodate')}__"
            f"{safe_name(Path(timer.meta.get('file', 'unknown')).stem)}")
    write_report(session_dir / name, data, f"{camera} | {timer.meta.get('file')}")

    inf = data["children"].get("inference", {})
    row = {
        "camera": camera,
        "started_at": data["started_at"],
        "finished_at": data["finished_at"],
        "date": timer.meta.get("date"),
        "file": timer.meta.get("file"),
        "status": status,
        "total_s": data["total_seconds"],
    }
    for col, stage in SUMMARY_STAGES.items():
        row[col] = round(timer.seconds(stage), 3)
    for col, stage in SUMMARY_INFERENCE_STAGES.items():
        row[col] = inf.get("stages", {}).get(stage, {}).get("seconds", "")
    for key in SUMMARY_META:
        row[key] = timer.meta.get(key, "")
    for key in SUMMARY_INFERENCE_META:
        row[key] = inf.get("meta", {}).get(key, "")
    row["error"] = error or ""
    append_summary_csv(session_dir / "summary.csv", row)


def run_command(cmd, desc, env=None):
    print(f"\n[{threading.current_thread().name}] --- {desc} ---")
    print(f"[{threading.current_thread().name}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[{threading.current_thread().name}] ERROR in {desc}:\n{result.stderr}")
    return result.returncode == 0

def download_and_remux_worker(download_queue, inference_queue, manifest, folder_id, base_dir,
                              session_dir, camera_name):
    downloader = Downloader(root_folder_id=folder_id)

    while True:
        item = download_queue.get()
        if item is None:
            break

        file_id = item['file_id']
        gdrive_path = item['gdrive_path']
        date = item['date']
        filename = item['filename']
        timestamp = filename.replace('.dav', '')

        local_dir = os.path.join(base_dir, date)
        local_dav = os.path.join(local_dir, filename)
        local_mp4 = local_dav + ".mp4"

        timer = RunTimer()
        timer.meta.update(camera=camera_name, file=filename, date=date, gdrive_path=gdrive_path,
                          dl_worker=threading.current_thread().name)

        # 1. Download
        manifest.update_status(file_id, "DOWNLOADING")
        skipped = os.path.exists(local_dav) and os.path.getsize(local_dav) > 0
        timer.meta["download_skipped_existing"] = skipped
        with timer.stage("download"):
            ok = downloader.download_file(gdrive_path, local_dav)
        if not ok:
            manifest.update_status(file_id, "FAILED", error="Download failed")
            finish_run(timer, session_dir, camera_name, "FAILED", "Download failed")
            download_queue.task_done()
            continue
        dav_mb = os.path.getsize(local_dav) / 1e6
        timer.meta["dav_size_mb"] = round(dav_mb, 2)
        if not skipped:
            timer.meta["download_speed_mb_s"] = round(dav_mb / max(timer.seconds("download"), 1e-6), 2)

        # 2. Remux
        manifest.update_status(file_id, "DOWNLOADED", local_path=local_dav)
        remux_cmd = ['tools/ffmpeg_bin/ffmpeg', '-y', '-i', local_dav, '-c', 'copy', local_mp4]
        with timer.stage("remux"):
            ok = run_command(remux_cmd, f"Remuxing {filename}")
        if not ok:
            manifest.update_status(file_id, "FAILED", error="Remux failed")
            finish_run(timer, session_dir, camera_name, "FAILED", "Remux failed")
            download_queue.task_done()
            continue
        timer.meta["mp4_size_mb"] = round(os.path.getsize(local_mp4) / 1e6, 2)

        # 3. Pass to inference (queue wait includes time blocked on a full inference queue)
        item['local_dav'] = local_dav
        item['local_mp4'] = local_mp4
        item['timestamp'] = timestamp
        item['timer'] = timer
        item['handoff_at'] = time.perf_counter()
        inference_queue.put(item)
        download_queue.task_done()

def inference_worker(inference_queue, manifest, base_output_dir, session_dir, camera_name, config_file=None):
    while True:
        item = inference_queue.get()
        if item is None:
            break
            
        file_id = item['file_id']
        date = item['date']
        filename = item['filename']
        timestamp = item['timestamp']
        local_dav = item['local_dav']
        local_mp4 = item['local_mp4']
        timer = item['timer']
        timer.add("queue_wait_for_inference", time.perf_counter() - item['handoff_at'])
        timer.meta["inf_worker"] = threading.current_thread().name

        start_time_part = timestamp.split('-')[0]
        start_time_formatted = start_time_part.replace('.', ':')
        
        output_dir = os.path.join(base_output_dir, date, timestamp)
        os.makedirs(output_dir, exist_ok=True)
            
        # Inference using verified env python
        inference_cmd = [
            ENV_PYTHON, 'vehicle-counting/pipeline/counting/main.py',
            '--video', local_mp4,
            '--output_dir', output_dir,
            '--no_annotation',
            '--start_time', start_time_formatted
        ]
        if config_file:
            inference_cmd.extend(['--config', config_file])

        # main.py writes its own step breakdown here; drop any stale copy from a previous attempt.
        timing_path = os.path.join(output_dir, "timing.json")
        if os.path.exists(timing_path):
            os.remove(timing_path)
        env = {**os.environ, "MVSA_ORCHESTRATED": "1"}

        with timer.stage("inference"):
            ok = run_command(inference_cmd, f"Inference {filename}", env=env)
        if os.path.exists(timing_path):
            try:
                with open(timing_path) as f:
                    timer.children["inference"] = json.load(f)
            except (OSError, ValueError):
                pass
        if not ok:
            manifest.update_status(file_id, "FAILED", error="Inference failed")
        else:
            manifest.update_status(file_id, "COMPLETED")
            print(f"\n[{threading.current_thread().name}] SUCCESS: Completed {filename}")

        # Cleanup
        with timer.stage("cleanup"):
            if os.path.exists(local_dav):
                try:
                    os.remove(local_dav)
                except OSError:
                    pass
            if os.path.exists(local_mp4):
                try:
                    os.remove(local_mp4)
                except OSError:
                    pass

        finish_run(timer, session_dir, camera_name, "OK" if ok else "FAILED",
                   None if ok else "Inference failed")

        inference_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded Orchestrator for Historical Video Processing")
    parser.add_argument('--folder-id', type=str, required=True, help="Google Drive folder ID")
    parser.add_argument('--camera-name', type=str, default=None, help="Camera identifier (e.g. tvc5_wb)")
    parser.add_argument('--db', type=str, default=None, help="Manifest database path")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--config', type=str, default=None, help="Vehicle count config YAML path")
    parser.add_argument('--dl-threads', type=int, default=4, help="Number of download/remux threads")
    parser.add_argument('--inf-threads', type=int, default=2, help="Number of inference threads")
    parser.add_argument('--batch-size', type=int, default=5, help="Batch size for fetching pending videos")
    parser.add_argument('--scan', action='store_true', help="Scan Google Drive before starting")
    args = parser.parse_args()

    camera_name = args.camera_name or args.folder_id
    db_path = args.db or f"historical-processor/data/manifest_{camera_name}.db"
    output_dir = args.output_dir or f"historical-processor/output/{camera_name}"
    video_base_dir = f"historical-processor/data/videos/{camera_name}"
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_base_dir, exist_ok=True)

    session_dir = RUNS_LOG_DIR / safe_name(camera_name) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(json.dumps({
        "camera_name": camera_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }, indent=2))

    manifest = IngestManifest(db_path=db_path)

    stats = manifest.get_stats()
    if args.scan or stats.get('TOTAL', 0) == 0:
        print(f"Scanning Google Drive folder ID: {args.folder_id} ...")
        scan_t0 = time.perf_counter()
        scanner = DriveScanner(root_folder_id=args.folder_id)
        scanner.scan(manifest)
        stats = manifest.get_stats()
        scan_secs = time.perf_counter() - scan_t0
        print(f"Scan complete. Total files in manifest: {stats.get('TOTAL', 0)} ({scan_secs:.1f}s)")
        (session_dir / "drive_scan.log").write_text(
            f"drive scan took {scan_secs:.2f}s, manifest total={stats.get('TOTAL', 0)}\n")
        
    download_queue = queue.Queue(maxsize=20)
    inference_queue = queue.Queue(maxsize=10)
    
    # Start thread pools
    dl_threads = []
    for i in range(args.dl_threads):
        t = threading.Thread(
            target=download_and_remux_worker,
            args=(download_queue, inference_queue, manifest, args.folder_id, video_base_dir,
                  session_dir, camera_name),
            name=f"DL-Worker-{i}"
        )
        t.daemon = True
        t.start()
        dl_threads.append(t)
        
    inf_threads = []
    for i in range(args.inf_threads):
        t = threading.Thread(
            target=inference_worker,
            args=(inference_queue, manifest, output_dir, session_dir, camera_name, args.config),
            name=f"INF-Worker-{i}"
        )
        t.daemon = True
        t.start()
        inf_threads.append(t)
        
    print(f"\n=======================================================")
    print(f"Starting Orchestrator for: {camera_name}")
    print(f"Python Binary:{ENV_PYTHON}")
    print(f"Folder ID:    {args.folder_id}")
    print(f"Manifest DB:  {db_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Config:       {args.config or 'default'}")
    print(f"Workers:      {args.dl_threads} DL, {args.inf_threads} INF")
    print(f"Timing logs:  {session_dir}")
    print(f"=======================================================\n")
    
    # Reset any previously stuck or failed files on fresh launch
    for status in ["DOWNLOADING", "DOWNLOADED", "QUEUED", "FAILED"]:
        for item in manifest.get_by_status(status):
            manifest.update_status(item['file_id'], "DISCOVERED")
        
    while True:
        if download_queue.qsize() < 10:
            batch = manifest.get_pending_batch(batch_size=args.batch_size)
            if not batch:
                # download_queue/inference_queue.qsize() only reflects items
                # NOT yet picked up by a worker thread -- it hits 0 the moment
                # a worker calls .get(), well before that item's download,
                # remux, or inference actually finishes. Checking qsize()==0
                # here races with in-flight work and exits (killing the daemon
                # worker threads) while the last item(s) in a batch are still
                # being processed, silently losing them. The manifest DB is
                # the source of truth for whether work is actually done.
                stats = manifest.get_stats()
                in_flight = (stats.get('DISCOVERED', 0) + stats.get('QUEUED', 0)
                             + stats.get('DOWNLOADING', 0) + stats.get('DOWNLOADED', 0))
                if in_flight == 0:
                    print(f"\nAll files processed for {camera_name}! Pipeline complete.")
                    break
                else:
                    time.sleep(5)
                    continue

            for item in batch:
                manifest.update_status(item['file_id'], "QUEUED")
                download_queue.put(item)

        time.sleep(1)

if __name__ == '__main__':
    main()
