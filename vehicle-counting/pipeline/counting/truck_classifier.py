import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

logger = logging.getLogger("uvh_test.truck_classifier")

@dataclass
class ClassificationResult:
    subclass: str
    confidence: float
    reason: str

class TruckClassifier:
    """
    Sub-classifier for Truck and Three-wheeler native classes using the Indian Vehicle Classification model.
    """
    def __init__(self, config_path_or_dict):
        """
        config_path_or_dict can be a path to yaml or dict.
        We expect an ONNX model path.
        """
        import yaml
        if isinstance(config_path_or_dict, (str, Path)):
            with open(config_path_or_dict, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path_or_dict

        # Use ultralytics YOLO to load the model
        from ultralytics import YOLO
        
        cfg = self.config.get("indian_vehicle_classifier", {})
        model_path = cfg.get("weights", "models/trucks/trucks_model.onnx")
        self.confidence_thresh = cfg.get("confidence", 0.25)
        
        logger.info(f"Loading Indian Vehicle Classifier from {model_path}")
        self.model = YOLO(model_path, task="detect")
        self.model_class_names = self.model.names
        
        # Mappings
        self.truck_mapping = {
            "Truck - 2 axle": "2 Axle Truck",
            "Truck - 3 axle": "3 Axle Truck",
            "Truck - 4 axle": "MAV",
            "Truck - 5 axle": "MAV",
            "Truck - 6 axle": "MAV"
        }
        
        self.tw_mapping = {
            "Auto Rickshaw": "Auto",
            "E-Rickshaw": "EV Auto / E-Rickshaw",
            "Mini Loading": "Goods 3 Wheeler"
        }

    def classify(self, crop: np.ndarray, base_class: str) -> ClassificationResult:
        """
        crop: BGR numpy array of the vehicle
        base_class: "Truck" or "Three-wheeler"
        Returns a ClassificationResult with the refined subclass.
        """
        if crop is None or crop.size == 0:
            return ClassificationResult(subclass=base_class, confidence=0.0, reason="empty crop")

        try:
            results = self.model(crop, verbose=False)
        except Exception as e:
            logger.warning(f"Truck classifier inference failed: {e}")
            return ClassificationResult(subclass=base_class, confidence=0.0, reason="inference error")

        if not results or len(results[0].boxes) == 0:
            return ClassificationResult(subclass=base_class, confidence=0.0, reason="no detections")

        boxes = results[0].boxes
        best_box = max(boxes, key=lambda b: float(b.conf[0]))
        conf = float(best_box.conf[0])
        cls_id = int(best_box.cls[0])
        pred_name = self.model_class_names.get(cls_id, "")

        if conf < self.confidence_thresh:
            return ClassificationResult(subclass=base_class, confidence=conf, reason="below confidence threshold")

        if base_class == "Truck":
            if pred_name in self.truck_mapping:
                return ClassificationResult(
                    subclass=self.truck_mapping[pred_name],
                    confidence=conf,
                    reason=f"mapped from {pred_name}"
                )
            else:
                return ClassificationResult(subclass=base_class, confidence=conf, reason=f"ignoring {pred_name} for Truck route")
                
        elif base_class == "Three-wheeler":
            if pred_name in self.tw_mapping:
                return ClassificationResult(
                    subclass=self.tw_mapping[pred_name],
                    confidence=conf,
                    reason=f"mapped from {pred_name}"
                )
            else:
                return ClassificationResult(subclass=base_class, confidence=conf, reason=f"ignoring {pred_name} for 3W route")

        return ClassificationResult(subclass=base_class, confidence=0.0, reason="unsupported base class")
