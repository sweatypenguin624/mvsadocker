import os
import shutil
import xml.etree.ElementTree as ET

SRC = "/home/users/oauser/mvsa/dataset/voc"
BASE = "/home/users/oauser/mvsa/dataset/WUTDet_YOLO"

for split in ["train", "valid"]:
    os.makedirs(f"{BASE}/{split}/images", exist_ok=True)
    os.makedirs(f"{BASE}/{split}/labels", exist_ok=True)

with open(f"{SRC}/ImageSets/Main/train.txt") as f:
    train_ids = [x.strip() for x in f if x.strip()]

with open(f"{SRC}/ImageSets/Main/val.txt") as f:
    valid_ids = [x.strip() for x in f if x.strip()]


def convert_split(ids, split):

    converted = 0
    missing_images = 0
    missing_xml = 0
    bad_boxes = 0

    for i, image_id in enumerate(ids):

        image_src = f"{SRC}/JPEGImages/{image_id}.jpg"
        xml_src = f"{SRC}/Annotations/{image_id}.xml"

        image_dst = f"{BASE}/{split}/images/{image_id}.jpg"
        label_dst = f"{BASE}/{split}/labels/{image_id}.txt"

        if not os.path.exists(image_src):
            missing_images += 1
            continue

        if not os.path.exists(xml_src):
            missing_xml += 1
            continue

        try:
            root = ET.parse(xml_src).getroot()

            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)

            labels = []

            for obj in root.findall("object"):

                # One class: ship
                class_id = 0

                bbox = obj.find("bndbox")

                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                if xmax <= xmin or ymax <= ymin:
                    bad_boxes += 1
                    continue

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height

                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))

                labels.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}"
                )

            shutil.copy2(image_src, image_dst)

            with open(label_dst, "w") as f:
                f.write("\n".join(labels))

            converted += 1

        except Exception as e:
            print(f"Error: {image_id}: {e}")

        if (i + 1) % 500 == 0:
            print(f"{split}: {i + 1}/{len(ids)}")

    print(f"\nFinished {split}")
    print("Converted:", converted)
    print("Missing images:", missing_images)
    print("Missing XML:", missing_xml)
    print("Bad boxes:", bad_boxes)


convert_split(train_ids, "train")
convert_split(valid_ids, "valid")

print("\nCONVERSION COMPLETE")