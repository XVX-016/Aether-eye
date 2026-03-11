import argparse
from ultralytics import YOLO

def evaluate_yolo(model_path: str, data_yaml: str):
    """
    Evaluates a trained YOLOv8 aircraft detection model.
    Prints mAP50 and mAP50-95 metrics.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Starting evaluation on {data_yaml}...")
    # Validate the model on the specified dataset
    metrics = model.val(data=data_yaml, split='val', verbose=True)
    
    # Extract map50 and map50-95
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print(f"\nEvaluation Complete!")
    print(f"mAP@0.5:      {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO Aircraft Detection Model")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt YOLO weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    args = parser.parse_args()

    evaluate_yolo(args.weights, args.data)
