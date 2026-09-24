#!/usr/bin/env python3
"""Fine-tune a ResNet18 (ImageNet-pretrained, transfer learning per spec
section 18) binary classifier from a labeled goods_dataset/ folder,
producing a checkpoint loadable by layer2_classifiers.py::TorchStageClassifier.

Deliberately BINARY (positive_class vs everything else), not a full
N-way stage head: as of this run, config/layer2 labeling has 30 "Goods 3
Wheeler" examples but only 1-2 real negatives for any other single class --
nowhere near enough per-class data for a meaningful 4-way Broad classifier.
A binary "is this a Goods 3 Wheeler?" detector is the honest thing to train
on what's actually labeled; see layer2_pipeline.py's goods3w_detector
short-circuit for how it's used (fires ahead of the heuristic broad
classifier only for this one question, falls through otherwise).

Usage:
    python scripts/traffic_layer2/train_stage.py \\
        --data-dir goods_dataset/broad \\
        --positive-class goods_3_wheeler \\
        --output models/layer2/goods_3_wheeler_detector.pt \\
        --epochs 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class BinaryImageDataset(Dataset):
    """Walks <data_dir>/<split>/<class_dir>/*.jpg, labeling positive_class
    as 1 and every other class dir (excluding "_unlabeled") as 0.
    """

    def __init__(self, data_dir: Path, split: str, positive_class: str, transform):
        self.transform = transform
        self.samples = []
        split_dir = data_dir / split
        if not split_dir.exists():
            return
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name == "_unlabeled":
                continue
            label = 1 if class_dir.name == positive_class else 0
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

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


def evaluate(model, loader, device):
    model.eval()
    correct, total, tp, fp, fn = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "n": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="e.g. goods_dataset/broad")
    parser.add_argument("--positive-class", type=str, required=True, help="e.g. goods_3_wheeler")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_ds = BinaryImageDataset(args.data_dir, "train", args.positive_class, build_train_transform())
    val_ds = BinaryImageDataset(args.data_dir, "val", args.positive_class, build_transform())
    test_ds = BinaryImageDataset(args.data_dir, "test", args.positive_class, build_transform())

    n_pos_train = sum(1 for _, l in train_ds.samples if l == 1)
    n_neg_train = sum(1 for _, l in train_ds.samples if l == 0)
    print(f"train: {n_pos_train} positive, {n_neg_train} negative ({len(train_ds)} total)")
    print(f"val: {len(val_ds)} total, test: {len(test_ds)} total")

    if len(train_ds) == 0:
        print("Error: no training data found", file=sys.stderr)
        sys.exit(1)
    if n_neg_train == 0:
        print("Error: zero negative examples in train split -- cannot train a binary classifier "
              "(it would trivially predict positive for everything). Label at least one non-target "
              "example first.", file=sys.stderr)
        sys.exit(1)

    device = args.device
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Inverse-frequency class weighting -- with 22 positive vs 2 negative
    # examples, unweighted cross-entropy would barely penalize predicting
    # "positive" for everything.
    class_weights = torch.tensor([
        len(train_ds) / (2.0 * max(n_neg_train, 1)),
        len(train_ds) / (2.0 * max(n_pos_train, 1)),
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if len(test_ds) else None

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_ds)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            print(f"epoch {epoch:2d}  loss={avg_loss:.4f}  val_acc={metrics['accuracy']:.3f} "
                  f"val_f1={metrics['f1']:.3f} val_precision={metrics['precision']:.3f} val_recall={metrics['recall']:.3f}")
            if metrics["f1"] >= best_val_f1:
                best_val_f1 = metrics["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"epoch {epoch:2d}  loss={avg_loss:.4f}  (no val split)")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = {}
    if val_loader is not None:
        final_metrics["val"] = evaluate(model, val_loader, device)
    if test_loader is not None:
        final_metrics["test"] = evaluate(model, test_loader, device)
    train_loader_eval = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    final_metrics["train"] = evaluate(model, train_loader_eval, device)

    print("\nFinal metrics:")
    print(json.dumps(final_metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": ["Not " + args.positive_class, args.positive_class],
        "positive_class": args.positive_class,
        "metrics": final_metrics,
        "n_train": len(train_ds),
    }, args.output)
    print(f"\nSaved checkpoint -> {args.output}")

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
