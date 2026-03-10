import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

# 1. Define the new hierarchical classes
HIERARCHICAL_CLASSES = [
    "fighter_jet",           # 0
    "stealth_fighter",       # 1
    "bomber",                # 2
    "transport_aircraft",    # 3
    "tanker_aircraft",       # 4
    "awacs_aircraft",        # 5
    "helicopter_attack",     # 6
    "helicopter_transport",  # 7
    "drone_uav",             # 8
    "tiltrotor",             # 9
    "surveillance_aircraft", # 10
    "experimental_aircraft"  # 11
]

# Map old classes to new hierarchical index
CLASS_MAP = {
    'A10': 0, 'A400M': 3, 'AG600': 3, 'AH64': 6, 'AKINCI': 8, 'AV8B': 0,
    'An124': 3, 'An22': 3, 'An225': 3, 'An72': 3, 'B1': 2, 'B2': 1, 'B21': 2,
    'B52': 2, 'Be200': 3, 'C1': 3, 'C130': 3, 'C17': 3, 'C2': 3, 'C390': 3,
    'C5': 3, 'CH47': 7, 'CH53': 7, 'CL415': 3, 'E2': 5, 'E7': 5, 'EF2000': 0,
    'EMB314': 0, 'F117': 1, 'F14': 0, 'F15': 0, 'F16': 0, 'F18': 0, 'F2': 0,
    'F22': 1, 'F35': 1, 'F4': 0, 'FCK1': 0, 'H6': 2, 'Il76': 3, 'J10': 0,
    'J20': 1, 'J35': 1, 'J36': 1, 'J50': 11, 'JAS39': 0, 'JF17': 0, 'JH7': 0,
    'KAAN': 1, 'KC135': 4, 'KF21': 1, 'KIZILELMA': 8, 'KJ600': 5, 'Ka27': 7,
    'Ka52': 6, 'MQ25': 8, 'MQ9': 8, 'Mi24': 6, 'Mi26': 7, 'Mi28': 6, 'Mi8': 7,
    'Mig29': 0, 'Mig31': 0, 'Mirage2000': 0, 'P3': 10, 'RQ4': 8, 'Rafale': 0,
    'SR71': 10, 'Su24': 0, 'Su25': 0, 'Su34': 0, 'Su47': 11, 'Su57': 1,
    'T50': 0, 'TB001': 8, 'TB2': 8, 'Tejas': 0, 'Tornado': 0, 'Tu160': 2,
    'Tu22M': 2, 'Tu95': 2, 'U2': 10, 'UH60': 7, 'US2': 3, 'V22': 9, 'V280': 9,
    'Vulcan': 2, 'WZ10': 6, 'WZ7': 8, 'WZ9': 6, 'X29': 11, 'X32': 11,
    'XB70': 11, 'XQ58': 8, 'Y20': 3, 'YF23': 1, 'Z10': 6, 'Z19': 6
}

# Old indexes
OLD_CLASSES = list(CLASS_MAP.keys())

def create_hierarchical_dataset(
    source_dir: str = r"C:\Computing\Aether-eye\data\processed\aircraft_detection",
    target_dir: str = r"C:\Computing\Aether-eye\data\processed\aircraft_hierarchical",
    target_samples_per_class: int = 200
):
    print(f"Transforming fine-grained dataset into hierarchical dataset...")
    print(f"Targeting ~{target_samples_per_class} samples per super-class via oversampling.")

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if target_path.exists():
        print("Cleaning up old hierarchical directory...")
        shutil.rmtree(target_path)

    for split in ["train", "test"]:
        (target_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Dictionary to keep track of images per NEW class
    # new_class_id -> list of (image_file_path, label_lines)
    train_class_inventory = defaultdict(list)

    # 1. Process all original data
    for split in ["train", "test"]:
        images_dir = source_path / "images" / split
        labels_dir = source_path / "labels" / split

        if not images_dir.exists():
            continue

        for img_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue

            new_lines = []
            primary_new_class = None

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_class_idx = int(parts[0])
                    
                    if old_class_idx < len(OLD_CLASSES):
                        old_class_name = OLD_CLASSES[old_class_idx]
                        if old_class_name in CLASS_MAP:
                            new_class_idx = CLASS_MAP[old_class_name]
                            new_lines.append(f"{new_class_idx} {' '.join(parts[1:])}\n")
                            if primary_new_class is None:
                                primary_new_class = new_class_idx

            if new_lines and primary_new_class is not None:
                if split == "train":
                    train_class_inventory[primary_new_class].append((img_path, new_lines))
                else:
                    # Write test data immediately (no oversampling)
                    shutil.copy2(img_path, target_path / "images" / split / img_path.name)
                    with open(target_path / "labels" / split / label_path.name, "w") as f:
                        f.writelines(new_lines)

    # 2. Oversample the train set to balance classes
    for new_class_idx, items in train_class_inventory.items():
        if not items:
            continue
            
        current_count = len(items)
        class_name = HIERARCHICAL_CLASSES[new_class_idx]
        print(f"Class '{class_name}' ({new_class_idx}): Found {current_count} unique images.")

        duplication_needed = max(1, target_samples_per_class // current_count)
        remainder_needed = target_samples_per_class % current_count

        if current_count >= target_samples_per_class:
            duplication_needed = 1
            remainder_needed = 0

        # Create duplicated entries
        total_written = 0
        for i, (orig_img_path, lines) in enumerate(items):
            copies_to_make = duplication_needed + (1 if i < remainder_needed else 0)
            
            for copy_idx in range(copies_to_make):
                new_stem = f"{orig_img_path.stem}_copy_{copy_idx}" if copy_idx > 0 else orig_img_path.stem
                new_img_name = f"{new_stem}.jpg"
                new_lbl_name = f"{new_stem}.txt"

                shutil.copy2(orig_img_path, target_path / "images" / "train" / new_img_name)
                with open(target_path / "labels" / "train" / new_lbl_name, "w") as f:
                    f.writelines(lines)
                
                total_written += 1

        print(f"  -> Wrote {total_written} total images to balance '{class_name}'.")

    # 3. Write new data_hierarchical.yaml
    yaml_config = {
        'path': target_dir,
        'train': 'images/train',
        'val': 'images/test',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(HIERARCHICAL_CLASSES)}
    }

    yaml_path = target_path / "data_hierarchical.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    print(f"\n Relabeling complete!")
    print(f"New dataset mapped and balanced. Ready for YOLOv8 architecture retrain at: {yaml_path}")

if __name__ == "__main__":
    create_hierarchical_dataset()
