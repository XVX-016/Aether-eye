from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare VisionBasedAircraftDAA output into ImageFolder classification format."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-split", type=float, default=0.9, help="Split ratio for train set per class.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    state_csv = in_dir / "state_data.csv"
    metadata_json = in_dir / "metadata.json"
    if not state_csv.exists() or not metadata_json.exists():
        raise FileNotFoundError(f"Missing state_data.csv or metadata.json in {in_dir}")

    df = pd.read_csv(state_csv, skipinitialspace=True)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    ranges = [k for k in metadata.keys() if k != "total_images"]
    ranges_sorted = sorted(ranges, key=lambda x: float(x))

    filename_to_aircraft: dict[int, str] = {}
    for rng in ranges_sorted:
        start_s, end_s = str(rng).split(".")
        ac_name = str(metadata[rng]["ac"])
        for i in range(int(start_s), int(end_s) + 1):
            filename_to_aircraft[i] = ac_name

    all_images: list[tuple[Path, str]] = []
    for split in ["train", "valid"]:
        img_dir = in_dir / "images" / split
        for p in sorted(img_dir.glob("*.jpg")):
            idx = int(p.stem)
            ac_name = filename_to_aircraft.get(idx)
            if ac_name is None:
                row = df[df["filename"] == idx]
                if row.empty:
                    continue
                ac_name = str(row.iloc[0]["ac"])
            all_images.append((p, ac_name))

    by_class: dict[str, list[Path]] = {}
    for p, c in all_images:
        by_class.setdefault(c, []).append(p)

    # reset output folders
    for split in ["train", "val"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    for cls, images in by_class.items():
        images_sorted = sorted(images, key=lambda x: x.stem)
        cutoff = max(1, int(len(images_sorted) * args.train_split))
        train_imgs = images_sorted[:cutoff]
        val_imgs = images_sorted[cutoff:] or images_sorted[-1:]

        train_cls_dir = out_dir / "train" / cls
        val_cls_dir = out_dir / "val" / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)
        for p in train_imgs:
            shutil.copy2(p, train_cls_dir / p.name)
        for p in val_imgs:
            shutil.copy2(p, val_cls_dir / p.name)

    print(f"[ok] classification dataset written to: {out_dir}")


if __name__ == "__main__":
    main()

