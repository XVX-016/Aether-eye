import os
import shutil
import pandas as pd
from pathlib import Path

def convert_to_yolo():
    data_dir = Path("C:/Computing/Aether-eye/ml_core/DATASET/Aircraft/Military Aircraft Dataset")
    csv_path = data_dir / "labels_with_split.csv"
    images_dir = data_dir / "dataset"
    
    out_dir = Path("C:/Computing/Aether-eye/data/processed/aircraft_detection")
    
    # Create directories
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get unique classes
    unique_classes = sorted(df['class'].astype(str).unique())
    class_map = {cls: i for i, cls in enumerate(unique_classes)}
    
    print(f"Found {len(unique_classes)} classes.")
    
    # Group by filename
    grouped = df.groupby('filename')
    
    missing_count = 0
    processed_count = 0
    
    for filename, group in grouped:
        src_image = images_dir / f"{filename}.jpg"
        if not src_image.exists():
            missing_count += 1
            continue
            
        # Get split from the first row of the group (assuming all rows for same image have same split)
        split = group.iloc[0]['split']
        if split not in ["train", "val", "test"]:
            split = "train"  # Fallback
            
        dst_image = out_dir / "images" / split / f"{filename}.jpg"
        dst_label = out_dir / "labels" / split / f"{filename}.txt"
        
        # Copy image
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)
            
        # Write labels
        with open(dst_label, 'w') as f:
            for _, row in group.iterrows():
                # YOLO format: class x_center y_center width height (normalized)
                cls_id = class_map[str(row['class'])]
                
                img_w = row['width']
                img_h = row['height']
                
                # Check for invalid dims
                if img_w <= 0 or img_h <= 0: continue
                
                xmin, ymin = row['xmin'], row['ymin']
                xmax, ymax = row['xmax'], row['ymax']
                
                box_w = (xmax - xmin) / img_w
                box_h = (ymax - ymin) / img_h
                x_c = (xmin + xmax) / 2.0 / img_w
                y_c = (ymin + ymax) / 2.0 / img_h
                
                # Clip values to 0-1
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                box_w = max(0.0, min(1.0, box_w))
                box_h = max(0.0, min(1.0, box_h))
                
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}\n")
                
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} images...")

    # Write data.yaml
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"path: {out_dir.absolute().as_posix()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"names:\n")
        # Ensure we write proper quote-wrapped strings in case classes are numbers
        for i, cls in enumerate(unique_classes):
            f.write(f"  {i}: '{cls}'\n")

    print(f"Done. Processed {processed_count} images. Missing images: {missing_count}")
    print(f"YOLO dataset is ready at {out_dir}")

if __name__ == "__main__":
    convert_to_yolo()
