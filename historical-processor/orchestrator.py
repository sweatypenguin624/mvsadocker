import sys
import os
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

# Always use project virtual environment python if it exists
ENV_PYTHON = "/home/users/oauser/mvsa/env/bin/python"
if not os.path.exists(ENV_PYTHON):
    ENV_PYTHON = sys.executable

from ingest.manifest import IngestManifest
from ingest.downloader import Downloader
from ingest.drive_scanner import DriveScanner

def run_command(cmd, desc):
    print(f"\n[{threading.current_thread().name}] --- {desc} ---")
    print(f"[{threading.current_thread().name}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{threading.current_thread().name}] ERROR in {desc}:\n{result.stderr}")
    return result.returncode == 0

def download_and_remux_worker(download_queue, inference_queue, manifest, folder_id, base_dir):
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
        
        # 1. Download
        manifest.update_status(file_id, "DOWNLOADING")
        if not downloader.download_file(gdrive_path, local_dav):
            manifest.update_status(file_id, "FAILED", error="Download failed")
            download_queue.task_done()
            continue
            
        # 2. Remux
        manifest.update_status(file_id, "DOWNLOADED", local_path=local_dav)
        remux_cmd = ['tools/ffmpeg_bin/ffmpeg', '-y', '-i', local_dav, '-c', 'copy', local_mp4]
        if not run_command(remux_cmd, f"Remuxing {filename}"):
            manifest.update_status(file_id, "FAILED", error="Remux failed")
            download_queue.task_done()
            continue
            
        # 3. Pass to inference
        item['local_dav'] = local_dav
        item['local_mp4'] = local_mp4
        item['timestamp'] = timestamp
        inference_queue.put(item)
        download_queue.task_done()

def inference_worker(inference_queue, manifest, base_output_dir, config_file=None):
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
        
        if not run_command(inference_cmd, f"Inference {filename}"):
            manifest.update_status(file_id, "FAILED", error="Inference failed")
        else:
            manifest.update_status(file_id, "COMPLETED")
            print(f"\n[{threading.current_thread().name}] SUCCESS: Completed {filename}")
            
        # Cleanup
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
    
    manifest = IngestManifest(db_path=db_path)
    
    stats = manifest.get_stats()
    if args.scan or stats.get('TOTAL', 0) == 0:
        print(f"Scanning Google Drive folder ID: {args.folder_id} ...")
        scanner = DriveScanner(root_folder_id=args.folder_id)
        scanner.scan(manifest)
        stats = manifest.get_stats()
        print(f"Scan complete. Total files in manifest: {stats.get('TOTAL', 0)}")
        
    download_queue = queue.Queue(maxsize=20)
    inference_queue = queue.Queue(maxsize=10)
    
    # Start thread pools
    dl_threads = []
    for i in range(args.dl_threads):
        t = threading.Thread(
            target=download_and_remux_worker,
            args=(download_queue, inference_queue, manifest, args.folder_id, video_base_dir),
            name=f"DL-Worker-{i}"
        )
        t.daemon = True
        t.start()
        dl_threads.append(t)
        
    inf_threads = []
    for i in range(args.inf_threads):
        t = threading.Thread(
            target=inference_worker,
            args=(inference_queue, manifest, output_dir, args.config),
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
