import sys
import os
from ultralytics import YOLO
import time
import torch

torch.cuda.empty_cache()

pt_dest = 'historical-processor/UVH-26-MV-YOLOv11-X.pt'
model = YOLO(pt_dest)

print('Starting TRT FP16 Export (batch=1) with memory optimizations...')
t0 = time.time()
try:
    model.export(
        format='engine',
        half=True,
        imgsz=1280,
        dynamic=False,
        batch=1,
        workspace=1, # 1GB TRT workspace
        device='0'
    )
    print('Batch 1 export succeeded!')
except Exception as e:
    print('Failed batch 1 export:', e)
t1 = time.time()
