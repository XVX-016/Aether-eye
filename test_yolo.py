import os
from ultralytics import YOLO

def test_model():
    # Load the newly trained model weights
    model_path = r"C:\Computing\Aether-eye\runs\detect\train3\weights\best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    print("Loading model...")
    model = YOLO(model_path)

    # Select a few random test images
    test_dir = r"C:\Computing\Aether-eye\data\processed\aircraft_detection\images\test"
    
    # We'll just grab the first 5 images for a quick test
    images = [
        os.path.join(test_dir, "f0055552c8a0538a76b175765f443b97.jpg"),
        os.path.join(test_dir, "f006aa612a2e0922813d7070136b3445.jpg"),
        os.path.join(test_dir, "f00e3cf1b835061b56337a2807c821a4.jpg"),
        os.path.join(test_dir, "f028462528471182dc7a3772dd34409a.jpg"),
        os.path.join(test_dir, "f056795cc43e24b198e70f25de78ec0e.jpg")
    ]

    print(f"Running inference on {len(images)} images...")
    
    # Run inference and save the results
    # project/name controls where the output goes (runs/detect/test_results)
    results = model.predict(
        source=images,
        save=True,
        project=r"C:\Computing\Aether-eye\runs\detect",
        name="test_results",
        exist_ok=True,
        conf=0.25  # Confidence threshold
    )
    
    print("\n✅ Inference complete!")
    print(r"Results saved to: C:\Computing\Aether-eye\runs\detect\test_results")

if __name__ == "__main__":
    test_model()
