from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from tqdm import tqdm

from aether_ml.config import SiameseChangeConfig
from aether_ml.datasets import MultiTemporalChangeDataset
from aether_ml.models.siamese_unet import SiameseUNetChangeDetector, BceDiceLoss


class PairedTransformTrain:
    """
    Apply the same geometric augmentations to before/after images and mask.
    """

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, before, after, mask):
        # Random horizontal flip
        if random.random() < 0.5:
            before = F.hflip(before)
            after = F.hflip(after)
            mask = F.hflip(mask)

        # Random vertical flip
        if random.random() < 0.5:
            before = F.vflip(before)
            after = F.vflip(after)
            mask = F.vflip(mask)

        # Resize to fixed size
        before = F.resize(before, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        after = F.resize(after, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        # To tensor + normalization
        before_t = F.to_tensor(before)
        after_t = F.to_tensor(after)

        # Simple normalization to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        before_t = (before_t - mean) / std
        after_t = (after_t - mean) / std

        mask_t = F.to_tensor(mask)  # [1, H, W], values in [0,1]
        mask_t = (mask_t > 0.5).float()

        return before_t, after_t, mask_t


class PairedTransformVal:
    """
    Deterministic preprocessing for validation.
    """

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, before, after, mask):
        before = F.resize(before, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        after = F.resize(after, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        before_t = F.to_tensor(before)
        after_t = F.to_tensor(after)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        before_t = (before_t - mean) / std
        after_t = (after_t - mean) / std

        mask_t = F.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()

        return before_t, after_t, mask_t


def _create_dataloaders(cfg: SiameseChangeConfig) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg.resolved()

    train_tf = PairedTransformTrain(cfg.image_size)
    val_tf = PairedTransformVal(cfg.image_size)

    train_ds = MultiTemporalChangeDataset(
        root=cfg.root,
        list_file=cfg.train_list,
        transform=train_tf,
    )
    val_ds = MultiTemporalChangeDataset(
        root=cfg.root,
        list_file=cfg.val_list,
        transform=val_tf,
    )

    def _collate(batch):
        befores, afters, masks = zip(*batch)
        return (
            torch.stack(befores, dim=0),
            torch.stack(afters, dim=0),
            torch.stack(masks, dim=0),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )

    return train_loader, val_loader


def _create_model(cfg: SiameseChangeConfig, device: torch.device) -> nn.Module:
    model = SiameseUNetChangeDetector(in_channels=3, base_channels=cfg.base_channels)
    model.to(device)
    return model


def _compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Compute mean IoU over a batch.
    logits: [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float]:
    """
    Run one epoch and return (mean_loss, mean_iou).
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_iou = 0.0
    total_batches = 0

    for before, after, mask in tqdm(loader, desc="train" if train else "val", leave=False):
        before = before.to(device, non_blocking=True)
        after = after.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x = torch.cat([before, after], dim=1)  # [B, 6, H, W]

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, mask)

        if train:
            loss.backward()
            optimizer.step()

        batch_iou = _compute_iou(logits.detach(), mask.detach())

        bsz = mask.size(0)
        running_loss += loss.item() * bsz
        running_iou += batch_iou * bsz
        total_batches += bsz

    mean_loss = running_loss / total_batches
    mean_iou = running_iou / total_batches
    return mean_loss, mean_iou


def train_siamese_unet_change(cfg: SiameseChangeConfig) -> Dict[str, float]:
    """
    Train Siamese U-Net for multi-temporal change detection.

    Returns:
        Dictionary containing best validation IoU and optional checkpoint path.
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" not in cfg.device else "cpu")

    train_loader, val_loader = _create_dataloaders(cfg)
    model = _create_model(cfg, device)

    criterion = BceDiceLoss()
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
    best_val_iou = 0.0
    best_model_path: Path | None = None

    for epoch in range(cfg.epochs):
        train_loss, train_iou = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_iou = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        scheduler.step()

        if cfg.save_best and val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = cfg.output_dir / "siamese_unet_change_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
                best_model_path,
            )

    metrics: Dict[str, float] = {
        "best_val_iou": float(best_val_iou),
    }
    if best_model_path is not None:
        metrics["best_model_path"] = str(best_model_path)

    return metrics


def evaluate_siamese_unet_change_iou(
    cfg: SiameseChangeConfig,
    weights_path: str | Path,
) -> float:
    """
    Evaluate a trained Siamese U-Net on the validation split and return mean IoU.
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" not in cfg.device else "cpu")

    _, val_loader = _create_dataloaders(cfg)
    model = _create_model(cfg, device)

    checkpoint = torch.load(str(weights_path), map_location=device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for before, after, mask in tqdm(val_loader, desc="eval", leave=False):
            before = before.to(device, non_blocking=True)
            after = after.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x = torch.cat([before, after], dim=1)
            logits = model(x)
            batch_iou = _compute_iou(logits, mask)

            bsz = mask.size(0)
            total_iou += batch_iou * bsz
            total_samples += bsz

    return total_iou / total_samples if total_samples > 0 else 0.0

