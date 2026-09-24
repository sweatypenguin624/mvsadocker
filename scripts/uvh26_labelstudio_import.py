#!/usr/bin/env python3
"""Create a Label Studio project for reviewing UVH-26 detections and import
the sampled frames from dataset/uvh26_review/ as tasks, with the model's
YOLO-format predictions pre-loaded as editable bounding boxes.

Requires Label Studio running locally with LOCAL_FILES_SERVING_ENABLED and
LOCAL_FILES_DOCUMENT_ROOT set to <project>/dataset (see scripts run log /
README for the exact start command).

Usage:
    python scripts/uvh26_labelstudio_import.py --url http://localhost:8080 --api-key <token>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from label_studio_sdk.client import LabelStudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_ROOT / "dataset" / "uvh26_review"
CLASSES_FILE = REVIEW_DIR / "classes.txt"


def load_classes() -> list[str]:
    return [line.strip() for line in CLASSES_FILE.read_text().splitlines() if line.strip()]


def build_label_config(classes: list[str]) -> str:
    labels = "\n".join(f'    <Label value="{c}"/>' for c in classes)
    return f"""<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
</View>"""


def yolo_to_prediction(label_path: Path, classes: list[str], width: int, height: int) -> dict:
    results = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
            x = (xc - w / 2) * 100
            y = (yc - h / 2) * 100
            results.append(
                {
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": x,
                        "y": y,
                        "width": w * 100,
                        "height": h * 100,
                        "rotation": 0,
                        "rectanglelabels": [classes[cls_id]],
                    },
                }
            )
    return {"model_version": "uvh26-yolov11x", "result": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-title", default="UVH-26 review")
    args = parser.parse_args()

    from PIL import Image

    classes = load_classes()
    ls = LabelStudio(base_url=args.url, api_key=args.api_key)

    project = ls.projects.create(
        title=args.project_title,
        label_config=build_label_config(classes),
    )
    print(f"Created project {project.id}: {project.title}")

    images_dir = REVIEW_DIR / "images"
    labels_dir = REVIEW_DIR / "labels"
    image_files = sorted(images_dir.glob("*.jpg"))

    tasks = []
    for img_path in image_files:
        with Image.open(img_path) as img:
            w, h = img.size
        local_url = f"/data/local-files/?d=uvh26_review/images/{img_path.name}"
        prediction = yolo_to_prediction(labels_dir / f"{img_path.stem}.txt", classes, w, h)
        tasks.append({"data": {"image": local_url}, "predictions": [prediction]})

    batch_size = 100
    imported = 0
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        ls.projects.import_tasks(id=project.id, request=batch)
        imported += len(batch)
        print(f"imported {imported}/{len(tasks)} tasks")

    print(f"Done. Open {args.url}/projects/{project.id}/data to start reviewing.")


if __name__ == "__main__":
    main()
