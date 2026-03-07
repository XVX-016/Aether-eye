from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd


DAW_OPTIONS = {
    "Cessna Skyhawk": 20000.0,
    "Boeing 737-800": 100000.0,
    "King Air C90": 40000.0,
}


def get_bb_size(e0: float, n0: float, u0: float, e1: float, n1: float, u1: float, daw: float) -> tuple[float, float]:
    x = n1 - n0
    y = -(e1 - e0)
    z = u1 - u0
    r = (x * x + y * y + z * z) ** 0.5
    w = (1.0 / r) * daw
    h = (3.0 / 8.0) * w
    return h, w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert VisionBasedAircraftDAA generated output to YOLOv8 layout with deterministic relabeling."
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Path to Stanford generated dataset folder.")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="YOLOv8 output directory (images/train, images/valid, labels/train, labels/valid).",
    )
    p.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping aircraft name -> class index.",
    )
    return p.parse_args()


def load_class_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return {
            "Cessna Skyhawk": 0,
            "Boeing 737-800": 1,
            "King Air C90": 2,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def main() -> None:
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    class_map = load_class_map(args.class_map)

    state_csv = in_dir / "state_data.csv"
    metadata_json = in_dir / "metadata.json"
    if not state_csv.exists() or not metadata_json.exists():
        raise FileNotFoundError(f"Missing state_data.csv or metadata.json in {in_dir}")

    df = pd.read_csv(state_csv, skipinitialspace=True)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

    img_train = in_dir / "images" / "train"
    img_valid = in_dir / "images" / "valid"
    if not img_train.exists() or not img_valid.exists():
        raise FileNotFoundError("Expected images/train and images/valid in Stanford dataset.")

    # Reset output
    for rel in ["images/train", "images/valid", "labels/train", "labels/valid"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)

    ranges = [k for k in metadata.keys() if k != "total_images"]
    ranges_sorted = sorted(ranges, key=lambda x: float(x))

    # Build filename -> aircraft map from metadata ranges
    filename_to_aircraft: dict[int, str] = {}
    for rng in ranges_sorted:
        start_s, end_s = str(rng).split(".")
        start_i = int(start_s)
        end_i = int(end_s)
        ac_name = str(metadata[rng]["ac"])
        for i in range(start_i, end_i + 1):
            filename_to_aircraft[i] = ac_name

    def process_split(split: str, src_img_dir: Path) -> None:
        for img_path in sorted(src_img_dir.glob("*.jpg")):
            idx = int(img_path.stem)
            row = df[df["filename"] == idx]
            if row.empty:
                raise RuntimeError(f"Missing state row for image id {idx}")
            xp = float(row.iloc[0]["intr_x"])
            yp = float(row.iloc[0]["intr_y"])
            e0, n0, u0 = float(row.iloc[0]["e0"]), float(row.iloc[0]["n0"]), float(row.iloc[0]["u0"])
            e1, n1, u1 = float(row.iloc[0]["e1"]), float(row.iloc[0]["n1"]), float(row.iloc[0]["u1"])

            ac_name = filename_to_aircraft.get(idx, str(row.iloc[0]["ac"]))
            if ac_name not in class_map:
                raise KeyError(f"Aircraft '{ac_name}' missing in class map.")
            class_id = class_map[ac_name]
            daw = DAW_OPTIONS.get(ac_name, 40000.0)

            im = cv2.imread(str(img_path))
            if im is None:
                raise RuntimeError(f"Failed to read {img_path}")
            sh, sw = im.shape[:2]
            h_px, w_px = get_bb_size(e0, n0, u0, e1, n1, u1, daw=daw)
            x = xp / sw
            y = yp / sh
            w = w_px / sw
            h = h_px / sh

            dst_img = out_dir / "images" / split / img_path.name
            dst_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n", encoding="utf-8")

    process_split("train", img_train)
    process_split("valid", img_valid)

    names = sorted(class_map.items(), key=lambda kv: kv[1])
    yaml_lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/valid",
        "nc: " + str(len(names)),
        "names: [" + ", ".join(f"'{n[0]}'" for n in names) + "]",
    ]
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"[ok] YOLOv8 dataset written to: {out_dir}")


if __name__ == "__main__":
    main()

