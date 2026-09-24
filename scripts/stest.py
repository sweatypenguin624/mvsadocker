#!/usr/bin/env python3

"""
WUTDet / Naval Ship Detection - H100 Training

Dataset:
    /home/users/oauser/mvsa/dataset/stest

Expected structure:
    stest/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Output:
    /home/users/oauser/mvsa/results/stest/
"""

import sys
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_DIR = Path("/home/users/oauser/mvsa/dataset/stest")
OUTPUT_DIR = Path("/home/users/oauser/mvsa/results/stest")

DATA_YAML = DATASET_DIR / "data.yaml"

# Model
MODEL = "yolo12s.pt"

# H100 80GB starting configuration
BATCH = 64
IMGSZ = 1024

EPOCHS = 150
WORKERS = 16

# ============================================================
# 2. GPU CHECK
# ============================================================

print("=" * 70)
print("H100 SHIP DETECTION TRAINING")
print("=" * 70)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed.")
    sys.exit(1)

print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Make sure this job is running on the H100."
    )

gpu_name = torch.cuda.get_device_name(0)

print(f"GPU           : {gpu_name}")
print(f"GPU count     : {torch.cuda.device_count()}")
print(
    f"GPU memory    : "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
)

if "H100" not in gpu_name.upper():
    print("WARNING: GPU 0 does not appear to be an H100.")

# Enable TensorFloat-32 for H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# 3. DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET")
print("=" * 70)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory does not exist:\n{DATASET_DIR}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Missing data.yaml:\n{DATA_YAML}"
    )

required_dirs = [
    DATASET_DIR / "train" / "images",
    DATASET_DIR / "train" / "labels",
    DATASET_DIR / "valid" / "images",
    DATASET_DIR / "valid" / "labels",
]

for directory in required_dirs:
    if not directory.exists():
        raise FileNotFoundError(
            f"Missing required directory:\n{directory}"
        )

for split in ["train", "valid"]:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    images = [
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    ]

    labels = list(label_dir.glob("*.txt"))

    print(
        f"{split.upper():5s} | "
        f"images = {len(images):6d} | "
        f"labels = {len(labels):6d}"
    )

print(f"\nDataset YAML: {DATA_YAML}")

# ============================================================
# 4. CHECK ULTRALYTICS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ULTRALYTICS")
print("=" * 70)

try:
    import ultralytics
    from ultralytics import YOLO

    print(f"Ultralytics  : {ultralytics.__version__}")

except ImportError:
    print("ERROR: Ultralytics is not installed.")
    print("Run:")
    print("    pip install -U ultralytics")
    sys.exit(1)

# ============================================================
# 5. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RUN_NAME = "WUTDet_YOLO12S_H100"

RUN_DIR = OUTPUT_DIR / RUN_NAME

# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(f"Model         : {MODEL}")
print(f"Image size    : {IMGSZ}")
print(f"Batch         : {BATCH}")
print(f"Epochs        : {EPOCHS}")
print(f"Workers       : {WORKERS}")
print(f"Device        : {gpu_name}")
print("AMP           : True")
print("TF32          : True")
print("Cache         : RAM")
print(f"Dataset       : {DATASET_DIR}")
print(f"Output        : {RUN_DIR}")

print("=" * 70)

# ============================================================
# 7. LOAD MODEL
# ============================================================

print("\nLoading model...")

model = YOLO(MODEL)

print("Model loaded successfully.")

# ============================================================
# 8. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING H100 TRAINING")
print("=" * 70)

results = model.train(

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data=str(DATA_YAML),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    workers=WORKERS,
    cache="ram",
    amp=True,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,

    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,

    fliplr=0.5,

    mosaic=1.0,
    mixup=0.1,

    close_mosaic=10,

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    patience=30,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=42,

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    project=str(OUTPUT_DIR),
    name=RUN_NAME,
    exist_ok=True,

    save=True,
    save_period=5,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    verbose=True,
    plots=True,
)

# ============================================================
# 9. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Results directory:")
print(f"    {RUN_DIR}")

best = RUN_DIR / "weights" / "best.pt"
last = RUN_DIR / "weights" / "last.pt"

print("\nWeights:")

print(f"    Best: {best}")
print(f"    Last: {last}")

if best.exists():
    print("\nSUCCESS: best.pt was created.")

else:
    print("\nWARNING: best.pt was not found.")
    print("Check the training output for errors.")

if last.exists():
    print("SUCCESS: last.pt was created.")

print("\nDone.")