import time
import torch
import numpy as np
from ultralytics import YOLO

def benchmark(model_path, batch_size, num_warmup=10, num_runs=50):
    print(f"\n--- Benchmarking {model_path} with Batch Size {batch_size} ---")
    model = YOLO(model_path, task='detect')
    
    # Create dummy image batch (H100 can just use random arrays)
    dummy_img = np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)
    batch = [dummy_img] * batch_size
    
    print("Warming up...")
    for _ in range(num_warmup):
        model.predict(batch, imgsz=1280, device='cuda', verbose=False)
        
    print("Benchmarking...")
    start_time = time.time()
    for _ in range(num_runs):
        model.predict(batch, imgsz=1280, device='cuda', verbose=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    frames_processed = batch_size * num_runs
    fps = frames_processed / total_time
    
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Frames: {frames_processed}")
    print(f"FPS: {fps:.2f}")

if __name__ == '__main__':
    # Only test TRT static engine with exactly batch=4
    engine = 'UVH-26-MV-YOLOv11-X.engine'
    try:
        benchmark(engine, batch_size=4, num_warmup=20, num_runs=100)
    except Exception as e:
        print(f"Failed benchmark: {e}")