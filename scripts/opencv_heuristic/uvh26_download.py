"""Download the UVH-26 dataset (images + COCO annotations) from a Hugging Face
storage bucket into a local directory laid out for YOLO training.

UVH-26 is an India-specific traffic-camera dataset with 14 vehicle classes --
crucially including 'Three-wheeler' (auto-rickshaw), 'Mini-bus' and
'Tempo-traveller' as classes distinct from 'Bus'. The COCO-pretrained
detectors we were using have no such distinction and confidently call
auto-rickshaws 'bus', which is the false-positive source this dataset is
meant to fix.

The bucket stores images as UVH-26-{Train,Val}/data/<NNN>/<id>.png while the
COCO json refers to them by bare filename (<id>.png), so this script also
writes a filename -> local path manifest for the converter to use.

Usage (token comes from envhf/.env, key 'hftoken'):
    env/bin/python scripts/opencv_heuristic/uvh26_download.py \\
        --out data/uvh26 [--split train|val|both] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import download_bucket_files, list_bucket_tree

BUCKET_ID = "abhaydevpa/UVH-26-bucket"
SPLIT_PREFIX = {"train": "UVH-26-Train/data/", "val": "UVH-26-Val/data/"}


def load_token(env_path: Path) -> str:
    if os.environ.get("hftoken"):
        return os.environ["hftoken"]
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("hftoken="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no 'hftoken' found in {env_path} or environment")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/uvh26"))
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--env-file", type=Path, default=Path("/home/users/oauser/mvsa/envhf/.env"))
    p.add_argument("--batch-size", type=int, default=500, help="files per download_bucket_files call")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    token = load_token(args.env_file)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Listing bucket {BUCKET_ID} ...")
    tree = list(list_bucket_tree(BUCKET_ID, token=token))
    print(f"  {len(tree)} entries")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    manifest: dict = {}

    for split in splits:
        prefix = SPLIT_PREFIX[split]
        items = [it for it in tree if it.path.startswith(prefix) and it.path.lower().endswith(".png")]
        img_dir = args.out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        pairs = []
        for it in items:
            # flatten data/<NNN>/<id>.png -> images/<split>/<id>.png; the COCO json
            # keys on the bare filename, and ids are unique across subfolders.
            dest = img_dir / Path(it.path).name
            manifest[Path(it.path).name] = str(dest)
            if dest.exists() and dest.stat().st_size > 0:
                continue  # resumable: skip what we already have
            pairs.append((it, dest))

        print(f"[{split}] {len(items)} images, {len(pairs)} still to download")
        t0 = time.time()
        done = 0
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i:i + args.batch_size]
            download_bucket_files(BUCKET_ID, batch, token=token)
            done += len(batch)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(pairs) - done) / rate if rate > 0 else 0
            print(f"  [{split}] {done}/{len(pairs)} files | {rate:.1f} files/s | "
                  f"~{remaining/60:.1f} min left", flush=True)

    manifest_path = args.out / "image_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    print(f"\nWrote manifest ({len(manifest)} images): {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
