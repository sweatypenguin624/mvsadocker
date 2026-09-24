import sys
import os
import time
import cv2
import threading
import queue
import argparse
import subprocess
import psutil
from pathlib import Path
from ultralytics import YOLO

sys.path.append(os.path.abspath("vehicle-counting"))
from pipeline.counting.tracker import VehicleTracker

def remux_dav_to_mp4(dav_path, mp4_path):
    cmd = ["ffmpeg", "-y", "-i", str(dav_path), "-c:v", "copy", "-an", str(mp4_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_gpu_memory_and_util():
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd).decode('utf-8').strip().split(',')
        if len(output) >= 2:
            return float(output[0]), float(output[1].replace(' MiB',''))
    except Exception:
        pass
    return 0.0, 0.0

def decoder_worker(video_path, worker_id, num_workers, frame_queue, chunk_size, stop_event, times_dict):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        total_frames = chunk_size * num_workers
        
    start_frame = worker_id * (total_frames // num_workers)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_read = 0
    
    t0 = time.time()
    while not stop_event.is_set() and frames_read < chunk_size:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize represents preprocessing
        frame_resized = cv2.resize(frame, (1920, 1080))
        
        try:
            frame_queue.put((start_frame + frames_read, frame_resized), timeout=1.0)
            frames_read += 1
        except queue.Full:
            continue
            
    t1 = time.time()
    cap.release()
    times_dict[worker_id] = {'frames': frames_read, 'time': t1 - t0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-mp4", action="store_true")
    parser.add_argument("--frames-per-worker", type=int, default=300)
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if args.use_mp4:
        mp4_path = video_path.with_suffix(".mp4")
        if not mp4_path.exists():
            print("Remuxing DAV to MP4...")
            remux_dav_to_mp4(video_path, mp4_path)
        video_path = mp4_path

    model = YOLO("models/UVH-26/weights/YOLOv11-X/UVH-26-MV-YOLOv11-X.pt")
    
    frame_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    
    times_dict = {}
    workers = []
    
    start_time = time.time()
    
    for i in range(args.workers):
        w = threading.Thread(target=decoder_worker, args=(video_path, i, args.workers, frame_queue, args.frames_per_worker, stop_event, times_dict))
        workers.append(w)
        w.start()
        
    processed_frames = 0
    infer_time_total = 0.0
    
    total_expected = args.workers * args.frames_per_worker
    
    batch = []
    
    while processed_frames < total_expected:
        try:
            idx, frame = frame_queue.get(timeout=2.0)
            batch.append(frame)
        except queue.Empty:
            if not any(w.is_alive() for w in workers):
                break
            continue
            
        if len(batch) == args.batch_size:
            t0 = time.time()
            if args.batch_size == 1:
                results = model.track(source=batch[0], persist=True, device="cuda", imgsz=640, verbose=False)
            else:
                results = model.track(source=batch, persist=True, device="cuda", imgsz=640, verbose=False)
            t1 = time.time()
            infer_time_total += (t1 - t0)
            
            processed_frames += len(batch)
            batch = []
            
    if batch:
        t0 = time.time()
        results = model.track(source=batch, persist=True, device="cuda", imgsz=640, verbose=False)
        infer_time_total += (time.time() - t0)
        processed_frames += len(batch)
            
    stop_event.set()
    for w in workers:
        w.join()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    gpu_util, gpu_mem = get_gpu_memory_and_util()
    cpu_util = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    
    decode_frames = sum(d["frames"] for d in times_dict.values())
    decode_time = max((d["time"] for d in times_dict.values()), default=1e-5)
    decode_fps = decode_frames / decode_time if decode_time > 0 else 0
    
    infer_fps = processed_frames / infer_time_total if infer_time_total > 0 else 0
    end_to_end_fps = processed_frames / total_time if total_time > 0 else 0
    
    print("---")
    print("## Historical Processor Benchmark")
    print(f"Input: {video_path.name}")
    print(f"Resolution: 1920x1080")
    print(f"Workers: {args.workers}, Batch Size: {args.batch_size}, format: {'MP4' if args.use_mp4 else 'DAV'}")
    print(f"Total Frames: {processed_frames}")
    print(f"Decode:        {decode_fps:.2f} FPS")
    print(f"YOLO11-X Track:{infer_fps:.2f} FPS")
    print(f"End-to-end:    {end_to_end_fps:.2f} FPS")
    print(f"CPU total:     {cpu_util}%")
    print(f"GPU:           {gpu_util}%")
    print(f"GPU memory:    {gpu_mem} MB")
    
if __name__ == '__main__':
    main()
