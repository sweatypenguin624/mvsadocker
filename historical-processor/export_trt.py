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

def try_export(batch_size):
    print(f'\nStarting TRT FP16 Dynamic Export with batch={batch_size}...')
    try:
        model.export(
            format='engine',
            half=True,
            imgsz=1280,
            dynamic=True,
            batch=batch_size,
            workspace=16,
            device='0'
        )
        print(f'Success with batch={batch_size}!')
        return True
    except Exception as e:
        print(f'Failed export with batch={batch_size}: {e}')
        return False

for b in [8, 4, 1]:
    if try_export(b):
        break