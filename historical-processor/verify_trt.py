import sys
import os
from ultralytics import YOLO

EXPECTED_CLASSES = {
    0: 'Hatchback', 1: 'Sedan', 2: 'SUV', 3: 'MUV', 4: 'Bus', 
    5: 'Truck', 6: 'Three-wheeler', 7: 'Two-wheeler', 8: 'LCV', 
    9: 'Mini-bus', 10: 'tempo-traveller', 11: 'bicycle', 12: 'Van', 13: 'Others'
}

def verify_classes(model_names, source_name):
    print(f'\n--- Verifying Classes for {source_name} ---')
    all_match = True
    for k, v in EXPECTED_CLASSES.items():
        if model_names.get(k) != v:
            print(f'[ERROR] Class mismatch at ID {k}: Expected {v}, Got {model_names.get(k)}')
            all_match = False
    if all_match:
        print(f'[PASS] {source_name} taxonomy is correct (14 classes).')
    return all_match

def main():
    pt_path = '/home/users/oauser/mvsa/models/UVH-26/weights/YOLOv11-X/UVH-26-MV-YOLOv11-X.pt'
    engine_path = 'UVH-26-MV-YOLOv11-X.engine'
    image_path = '/home/users/oauser/mvsa/frame_344_original.jpg'

    pt_model = YOLO(pt_path)
    verify_classes(pt_model.names, 'PyTorch Model')
    
    trt_model = YOLO(engine_path, task='detect')
    verify_classes(trt_model.names, 'TensorRT Engine')
    
    # We must pass batch=4 for the static engine
    batch_images = [image_path] * 4
        
    print(f'\n--- Running Inference Comparison on Batch of 4 ---')
    pt_results = pt_model.predict(batch_images, imgsz=1280, device='cuda', verbose=False)[0]
    trt_results = trt_model.predict(batch_images, imgsz=1280, device='cuda', verbose=False)[0]
    
    print(f'\nPyTorch Detections (Image 0): {len(pt_results.boxes)}')
    for box in pt_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f'  Class: {pt_model.names[cls_id]:<15} | Conf: {conf:.4f} | BBox: {box.xyxy[0].tolist()}')

    print(f'\nTensorRT Detections (Image 0): {len(trt_results.boxes)}')
    for box in trt_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f'  Class: {trt_model.names[cls_id]:<15} | Conf: {conf:.4f} | BBox: {box.xyxy[0].tolist()}')

if __name__ == '__main__':
    main()