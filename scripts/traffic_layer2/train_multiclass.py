#!/usr/bin/env python3
"""Fine-tune a ResNet18 (ImageNet-pretrained, transfer learning) N-way
classifier from vehicle_dataset_v1/-shaped data
(<data_dir>/<split>/<slugified_label>/*.jpg), producing a checkpoint
loadable by layer2_classifiers.py::TorchStageClassifier.

Unlike train_stage.py (binary, for when only one class has real data),
this trains all of a stage's classes together -- use when at least 2
classes each have a reasonable example count. --labels fixes the class
index order explicitly (must match the order layer2_pipeline.py passes to
get_stage_classifier for that stage, e.g. BUS_LABELS or BROAD_LABELS) --
never inferred from directory listing order, which would silently
mismatch once a checkpoint is loaded for inference.

Usage:
    python scripts/traffic_layer2/train_multiclass.py \\
        --data-dir /tmp/bus_stage_data \\
        --labels "BRTC Bus,City/Private Bus,School Bus" \\
        --output models/layer2/bus_classifier.pt \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "unlabeled"


class MultiClassImageDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, labels: list, transform):
        self.transform = transform
        self.samples = []
        split_dir = data_dir / split
        for idx, label in enumerate(labels):
            class_dir = split_dir / slugify(label)
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader, device, n_classes):
    model.eval()
    correct, total = 0, 0
    confusion = [[0] * n_classes for _ in range(n_classes)]
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1
    return {"accuracy": correct / total if total else 0.0, "n": total, "confusion_matrix": confusion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=str, required=True, help="comma-separated, order matters")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    labels = [l.strip() for l in args.labels.split(",")]
    n_classes = len(labels)

    train_ds = MultiClassImageDataset(args.data_dir, "train", labels, build_train_transform())
    val_ds = MultiClassImageDataset(args.data_dir, "val", labels, build_transform())
    test_ds = MultiClassImageDataset(args.data_dir, "test", labels, build_transform())

    per_class_train = [0] * n_classes
    for _, l in train_ds.samples:
        per_class_train[l] += 1
    print("Classes:", labels)
    print("Train counts per class:", dict(zip(labels, per_class_train)))
    print(f"val: {len(val_ds)}, test: {len(test_ds)}")

    if len(train_ds) == 0:
        print("Error: no training data found", file=sys.stderr)
        sys.exit(1)
    if any(c == 0 for c in per_class_train):
        missing = [l for l, c in zip(labels, per_class_train) if c == 0]
        print(f"Error: zero training examples for class(es) {missing} -- "
              f"either drop them from --labels or label some examples first.", file=sys.stderr)
        sys.exit(1)

    device = args.device
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)

    total = sum(per_class_train)
    class_weights = torch.tensor([total / (n_classes * c) for c in per_class_train]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if len(test_ds) else None

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, lbls in train_loader:
            images, lbls = images.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_ds)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, n_classes)
            print(f"epoch {epoch:2d}  loss={avg_loss:.4f}  val_acc={metrics['accuracy']:.3f}")
            if metrics["accuracy"] >= best_val_acc:
                best_val_acc = metrics["accuracy"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"epoch {epoch:2d}  loss={avg_loss:.4f}  (no val split)")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = {}
    if val_loader is not None:
        final_metrics["val"] = evaluate(model, val_loader, device, n_classes)
    if test_loader is not None:
        final_metrics["test"] = evaluate(model, test_loader, device, n_classes)
    final_metrics["train"] = evaluate(model, DataLoader(train_ds, batch_size=args.batch_size), device, n_classes)
    final_metrics["labels"] = labels

    print("\nFinal metrics:")
    print(json.dumps(final_metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "metrics": final_metrics,
        "n_train": len(train_ds),
    }, args.output)
    print(f"\nSaved checkpoint -> {args.output}")
    args.output.with_suffix(".metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
