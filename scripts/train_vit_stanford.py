from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class TrainCfg:
    data_root: Path
    output_dir: Path
    model_name: str = "vit_small_patch16_224"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 4
    device: str = "cuda"


def build_loaders(cfg: TrainCfg):
    train_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(cfg.data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(cfg.data_root / "val", transform=val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def train(cfg: TrainCfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_loaders(cfg)
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": cfg.model_name,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "best_val_acc": best_val_acc,
                },
                cfg.output_dir / "best.pt",
            )

    print(f"[ok] best_val_acc={best_val_acc:.4f}, checkpoint={cfg.output_dir / 'best.pt'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ViT classifier on Stanford-generated aircraft classification dataset."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/stanford_aircraft_cls"),
        help="ImageFolder root containing train/ and val/ class subfolders.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("experiments/aircraft/stanford_vit"))
    p.add_argument("--model", type=str, default="vit_small_patch16_224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = TrainCfg(
        data_root=a.data_root,
        output_dir=a.output_dir,
        model_name=a.model,
        image_size=a.image_size,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        num_workers=a.num_workers,
        device=a.device,
    )
    train(cfg)

