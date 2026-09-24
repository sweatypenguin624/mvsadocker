
import logging
import numpy as np
from typing import Tuple, Any
from ultralytics import YOLO

logger = logging.getLogger("uvh_test.detector")

class VehicleDetector:
    def __init__(self, model_path: str, device: str = "0", conf_thres: float = 0.35, imgsz: int = 1280):
        self.model_path = model_path
        self.device = device
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        
        logger.info(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")
        
        # Patch names if they are missing (common when loading bare .engine files)
        try:
            self.class_names = self.model.names
            if not self.class_names or list(self.class_names.values())[0] == "class0":
                raise ValueError("Bad names")
        except (AttributeError, ValueError):
            self.class_names = {0: 'Hatchback', 1: 'Sedan', 2: 'SUV', 3: 'MUV', 4: 'Bus', 5: 'Truck', 6: 'Three-wheeler', 7: 'Two-wheeler', 8: 'LCV', 9: 'Mini-bus', 10: 'tempo-traveller', 11: 'bicycle', 12: 'Van', 13: 'Others'}
            
        logger.info(f"Loaded {len(self.class_names)} classes.")

    def detect(self, frame: np.ndarray) -> Any:
        results = self.model.predict(
            frame,
            conf=self.conf_thres,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        return results[0]
