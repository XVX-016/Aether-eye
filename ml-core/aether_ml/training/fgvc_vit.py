from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm

from aether_ml.config import FgvcVitConfig
from aether_ml.datasets import FgvcAircraftDataset


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Data augmentation and preprocessing for training and validation.
    """
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # ImageNet mean
        std=(0.229, 0.224, 0.225),  # ImageNet std
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_tf, val_tf


def _create_dataloaders(cfg: FgvcVitConfig) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders for FGVC Aircraft.
    Returns train_loader, val_loader, num_classes.
    """
    cfg = cfg.resolved()
    train_tf, val_tf = _build_transforms(cfg.image_size)

    train_ds = FgvcAircraftDataset(root=cfg.data_root, split="train")
    val_ds = FgvcAircraftDataset(root=cfg.data_root, split="val")

    num_classes = train_ds.num_classes

    # Wrap datasets with transforms
    from torch.utils.data import Dataset

    class Wrapped(Dataset):
        def __init__(self, base_ds, tf):
            self.base_ds = base_ds
            self.tf = tf

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            img, label = self.base_ds[idx]
            return self.tf(img), label

    train_wrapped = Wrapped(train_ds, train_tf)
    val_wrapped = Wrapped(val_ds, val_tf)

    train_loader = DataLoader(
        train_wrapped,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_wrapped,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def _create_model(cfg: FgvcVitConfig, num_classes: int) -> nn.Module:
    """
    Create a ViT model from timm for fine-tuning.
    """
    model = timm.create_model(
        cfg.model_name,
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def _accuracy_top1(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.size(0)
        return correct / total if total > 0 else 0.0


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float]:
    """
    Run one epoch and return (mean_loss, top1_acc).
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        running_total += batch_size

    mean_loss = running_loss / running_total
    top1 = running_correct / running_total
    return mean_loss, top1


def train_vit_aircraft(cfg: FgvcVitConfig) -> Dict[str, float]:
    """
    Fine-tune a Vision Transformer on the FGVC Aircraft dataset.

    Returns a dictionary with training/validation loss and top-1 accuracy,
    including best validation accuracy.
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" not in cfg.device else "cpu")

    train_loader, val_loader, num_classes = _create_dataloaders(cfg)
    model = _create_model(cfg, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_model_path: Path | None = None

    for epoch in range(cfg.epochs):
        train_loss, train_top1 = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_top1 = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        scheduler.step()

        if cfg.save_best and val_top1 > best_val_acc:
            best_val_acc = val_top1
            best_model_path = cfg.output_dir / "vit_fgvc_aircraft_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_top1": val_top1,
                },
                best_model_path,
            )

    metrics: Dict[str, float] = {
        "best_val_top1": float(best_val_acc),
    }
    if best_model_path is not None:
        metrics["best_model_path"] = str(best_model_path)

    return metrics


def evaluate_vit_aircraft_top1(
    cfg: FgvcVitConfig,
    weights_path: str | Path,
) -> float:
    """
    Evaluate a fine-tuned ViT on the FGVC Aircraft validation split and
    return top-1 accuracy.
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" not in cfg.device else "cpu")

    _, val_loader, num_classes = _create_dataloaders(cfg)
    model = _create_model(cfg, num_classes=num_classes).to(device)

    checkpoint = torch.load(str(weights_path), map_location=device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0

