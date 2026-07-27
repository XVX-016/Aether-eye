import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Add ml_core to path.
ml_core_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ml_core_dir))

from aether_ml.datasets.military_aircraft import MilitaryAircraftDataset

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")


# Config
MILITARY_ROOT = str(ml_core_dir / "DATASET/Aircraft/Military Aircraft Dataset/crop")
CHECKPOINT_IN = str(
    ml_core_dir.parent / "ml_core/artifacts/aircraft_classifier_v1/aircraft_classifier_v1.pt"
)
OUTPUT_DIR = str(ml_core_dir.parent / "runs/aircraft_military_finetune")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
BATCH_SIZE = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30
IMG_SIZE = 224


# Transforms
train_tf = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total if total else 0.0


def run_phase(model, train_loader, val_loader, optimizer, scheduler, epochs, phase_name, out_dir):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        val_acc = evaluate(model, val_loader)
        if scheduler:
            scheduler.step()

        print(
            f"[{phase_name}] Epoch {epoch + 1}/{epochs} "
            f"loss={total_loss / total:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"{out_dir}/best_{phase_name}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "phase": phase_name,
                    "classes": None,
                },
                save_path,
            )
            print(f"  -> Saved best {phase_name}: {val_acc:.4f}")

    return best_acc


def main():
    print(f"Device:  {DEVICE}")
    print(f"Dataset: {MILITARY_ROOT}")

    train_ds = MilitaryAircraftDataset(MILITARY_ROOT, split="train", transform=train_tf)
    val_ds = MilitaryAircraftDataset(MILITARY_ROOT, split="val", transform=val_tf)

    print(f"Classes ({train_ds.num_classes}): {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model("convnext_small", pretrained=False, num_classes=train_ds.num_classes).to(DEVICE)

    if Path(CHECKPOINT_IN).exists():
        ckpt = torch.load(CHECKPOINT_IN, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        filtered = {
            k: v
            for k, v in state.items()
            if not any(x in k for x in ("head", "classifier", "fc"))
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded v1 backbone. Missing keys: {len(missing)}")
    else:
        print(f"Warning: checkpoint not found at {CHECKPOINT_IN}, training from scratch")

    print("\n=== PHASE 1: Head-only (frozen backbone) ===")
    for name, param in model.named_parameters():
        param.requires_grad = any(x in name for x in ("head", "classifier", "fc"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}")

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EPOCHS)
    best1 = run_phase(model, train_loader, val_loader, opt1, sched1, PHASE1_EPOCHS, "head", OUTPUT_DIR)

    ckpt_head = torch.load(f"{OUTPUT_DIR}/best_head.pt", map_location=DEVICE)
    model.load_state_dict(ckpt_head["model_state_dict"])

    print("\n=== PHASE 2: Full fine-tuning ===")
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}")

    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS)
    best2 = run_phase(model, train_loader, val_loader, opt2, sched2, PHASE2_EPOCHS, "full", OUTPUT_DIR)

    metrics = {
        "military_classes": train_ds.classes,
        "class_display_names": {c: train_ds.display_name(c) for c in train_ds.classes},
        "num_classes": train_ds.num_classes,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "phase1_best_val_acc": best1,
        "phase2_best_val_acc": best2,
        "backbone": "convnext_small",
        "dataset": "Military Aircraft Dataset (crop/)",
    }
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPhase 1 best: {best1:.4f}")
    print(f"Phase 2 best: {best2:.4f}")
    print(f"Metrics -> {OUTPUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
