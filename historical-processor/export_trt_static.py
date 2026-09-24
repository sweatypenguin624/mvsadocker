import sys
import os
from ultralytics import YOLO
import shutil

pt_source = '/home/users/oauser/mvsa/models/UVH-26/weights/YOLOv11-X/UVH-26-MV-YOLOv11-X.pt'
pt_dest = 'historical-processor/UVH-26-MV-YOLOv11-X.pt'

if not os.path.exists(pt_dest):
    print(f'Copying {pt_source} to {pt_dest}...')
    shutil.copy2(pt_source, pt_dest)

model = YOLO(pt_dest)

print('\mStarting TRT FP16 STATIC Export with batch=4...')
model.export(
    format='engine',
    half=True,
    imgsz=1280,
    dynamic=False,
    batch=4,
    workspace=24,
    device='0'
)
print('Success with static batch=4!')
