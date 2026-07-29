import torch
import sys
import json
import shutil
import warnings
from pathlib import Path

ml_core_dir = Path(__file__).resolve().parent.parent / "ml_core"
sys.path.insert(0, str(ml_core_dir))

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

CHECKPOINT = "runs/aircraft_military_finetune/best_full.pt"
OUTPUT_DIR = "ml_core/artifacts/aircraft_classifier_v2"
NUM_CLASSES = 98  # Military Aircraft Dataset classes
IMG_SIZE = 224

def export():
    ckpt_path = Path(CHECKPOINT)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        print("Training may still be in progress.")
        return
    
    print(f"Loading checkpoint: {CHECKPOINT}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    val_acc = ckpt.get("val_acc", "unknown")
    epoch   = ckpt.get("epoch", "unknown")
    print(f"Checkpoint: epoch={epoch}, val_acc={val_acc}")
    
    # Build model
    model = timm.create_model(
        "convnext_small", pretrained=False, num_classes=NUM_CLASSES
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    
    # Copy .pt
    pt_dest = out / "aircraft_classifier_v2.pt"
    shutil.copy(ckpt_path, pt_dest)
    print(f"Copied .pt -> {pt_dest}")
    
    # Export ONNX
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_dest = out / "aircraft_classifier_v2.onnx"
    
    torch.onnx.export(
        model, dummy, str(onnx_dest),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX exported -> {onnx_dest}")
    print(f"ONNX size: {onnx_dest.stat().st_size / 1e6:.1f} MB")
    
    # Load class names from dataset
    try:
        from aether_ml.datasets.military_aircraft import MilitaryAircraftDataset
        MILITARY_ROOT = str(
            ml_core_dir / "DATASET/Aircraft/Military Aircraft Dataset/crop"
        )
        ds = MilitaryAircraftDataset(MILITARY_ROOT, split="train")
        classes = ds.classes
        display_names = {c: ds.display_name(c) for c in classes}
    except Exception as e:
        print(f"Warning: could not load class names: {e}")
        classes = [f"class_{i}" for i in range(NUM_CLASSES)]
        display_names = {}
    
    # Model card
    card = {
        "model": "ConvNeXt Small (Military Fine-tuned)",
        "version": "v2",
        "architecture": "convnext_small",
        "num_classes": NUM_CLASSES,
        "classes": classes,
        "display_names": display_names,
        "training_data": "Military Aircraft Dataset (crop/), 34718 train samples",
        "val_acc": float(val_acc) if isinstance(val_acc, (int, float)) else val_acc,
        "epoch": epoch,
        "domain": "military aircraft (nadir + oblique mixed)",
        "backbone_pretrained_from": "aircraft_classifier_v1 (FGVC)",
        "status": "production_candidate",
        "input_size": IMG_SIZE,
        "normalization": "imagenet",
    }
    with open(out / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Model card -> {out / 'model_card.json'}")
    print(f"\nDone. Val acc: {val_acc}")

if __name__ == "__main__":
    export()
