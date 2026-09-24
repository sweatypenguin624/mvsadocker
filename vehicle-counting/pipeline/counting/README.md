# Vehicle Classification (Vclassification)

This directory contains a complete pipeline for vehicle detection, tracking, counting, and classification using the UVH-26 YOLOv11-X model.

## Overview

The purpose of this module is to process video streams, detect vehicles using a YOLO model, filter the detections, track them across frames, and count/classify them based on their crossing of a defined Region of Interest (ROI) line.

## Files

- `main.py`: The fully-integrated entry point script. It uses `VehicleDetector`, `DetectionFilter`, `VehicleTracker`, and `VehicleCounter` to process an input video. It outputs an annotated video, counts to a CSV file, and a JSON summary.
- `test_uvh_model.py`: An alternative script used for basic testing of the YOLO model on an input video without the full tracking/counting pipeline.
- `detector.py`: Contains the `VehicleDetector` class which wraps the Ultralytics YOLO model for easier detection management.
- `tracker.py`: Contains the `VehicleTracker` and `TrackState` classes. These manage tracking states over multiple frames to establish vehicle tracks and stabilize classification across time.
- `counter.py`: Contains the `VehicleCounter` class with logic for counting vehicles as they cross a defined counting line.
- `filters.py`: Contains the `DetectionFilter` class to filter bounding boxes by area, dimensions, confidence, and class.
- `output.py`: Utilities (`Annotator`, `export_csv`, `export_json`) for rendering annotations (bounding boxes, labels, tracks) and exporting the final results.
- `config/`: Directory containing configuration parameters (e.g., `vehicle_count_config.yaml`) for the classification pipeline.

## Completeness Check

**Status: Complete**

The pipeline is fully integrated and complete. The main script (`main.py`) successfully glues together all the functional modules (`detector`, `tracker`, `counter`, `filters`, and `output`) and runs them according to the configuration settings provided in the `config/` directory.
