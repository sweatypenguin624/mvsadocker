"""RUN_MODE == "train_classifier": fine-tune a YOLO classification head on
labeled crops produced by collect_crops."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO

from config import PipelineConfig


def validate_color_dataset(dataset_dir: Path, cfg: PipelineConfig):
    if not dataset_dir.is_dir():
        raise RuntimeError(
            f"COLOR_DATASET_DIR not found: {dataset_dir}\n"
            "Run mode='collect_crops' first, sort the resulting images into "
            "train/red, train/blue, train/yellow, train/rest under this directory, "
            "then re-run mode='train_classifier'."
        )
    train_dir = dataset_dir / "train"
    if not train_dir.is_dir():
        raise RuntimeError(f"Expected a 'train' subfolder at: {train_dir}")
    for c in cfg.COLOR_CLASSES:
        class_dir = train_dir / c
        if not class_dir.is_dir() or len(os.listdir(class_dir)) == 0:
            raise RuntimeError(
                f"Expected non-empty folder {class_dir} (class '{c}'). "
                "Sort crops from the collect_crops output into all four color folders first."
            )
        print(f"  {c}: {len(os.listdir(class_dir))} images")
    print(f"Dataset validated at {dataset_dir}.")


def run_train_classifier(dataset_dir: Path, cfg: PipelineConfig, device: str):
    validate_color_dataset(dataset_dir, cfg)
    print(f"Training classifier from {cfg.CLASSIFIER_BASE_MODEL} on {dataset_dir} ...")
    cls_model = YOLO(cfg.CLASSIFIER_BASE_MODEL)
    train_results = cls_model.train(
        data=str(dataset_dir),
        epochs=cfg.CLASSIFIER_TRAIN_EPOCHS,
        imgsz=cfg.CLASSIFIER_IMG_SIZE,
        device=device,
    )
    best_weights = os.path.join(train_results.save_dir, "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best_weights}")
    print("Trained class order (index -> name):", cls_model.names)
    return best_weights
