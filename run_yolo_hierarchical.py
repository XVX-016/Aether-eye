import os
from ultralytics import YOLO

def train_hierarchical():
    print("============================================================")
    print(" Starting YOLOv8 Hierarchical Aircraft Training")
    print("   Model:       yolov8l.pt (Large)")
    print("   Resolution:  1024x1024")
    print("   Target:      200 Epochs")
    print("============================================================")
    
    # Load the heavier architecture
    model = YOLO("yolov8l.pt")
    
    # Train the model with the new hierarchical dataset
    model.train(
        data=r"C:\Computing\Aether-eye\data\processed\aircraft_hierarchical\data_hierarchical.yaml",
        epochs=200,
        imgsz=1024,
        batch=4,               # Reduced batch slightly because 1024px + yolov8l takes huge VRAM
        device=0,
        mosaic=1.0,            # Force mosaic augmentation
        mixup=0.2,             # Mixup augmentation for better generalization
        copy_paste=0.3,        # Copy-paste augmentation for aerial data
        project=r"C:\Computing\Aether-eye\runs\detect",
        name="train_hierarchical",
        exist_ok=True,
    )

if __name__ == "__main__":
    train_hierarchical()
