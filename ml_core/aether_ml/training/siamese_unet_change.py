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
from aether_ml.models.siamese_unet import SiameseUNetChangeDetector
from aether_ml.evaluation.metrics import HybridLoss
from aether_ml.models.factory import create_model


import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class PairedTransformTrain:
    """
    Apply the same geometric augmentations to before/after images and mask.
    """

    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size

    def __call__(self, before, after, mask):
        # 1. Random Crop BEFORE resize for spatial invariance
        # Ensure image is large enough, or fallback
        if before.height > 224 and before.width > 224:
            i, j, h, w = T.RandomCrop.get_params(before, output_size=(224, 224))
            before = F.crop(before, i, j, h, w)
            after = F.crop(after, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        # 2. Random horizontal flip
        if random.random() > 0.5:
            before = F.hflip(before)
            after = F.hflip(after)
            mask = F.hflip(mask)

        # 3. Random vertical flip
        if random.random() > 0.5:
            before = F.vflip(before)
            after = F.vflip(after)
            mask = F.vflip(mask)

        # 4. Random 90-degree rotations
        if random.random() > 0.5:
            rot = random.choice([90, 180, 270])
            before = F.rotate(before, rot)
            after = F.rotate(after, rot)
            mask = F.rotate(mask, rot)

        # 5. Color Jitter (images only)
        if random.random() < 0.8:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            
            before = F.adjust_brightness(before, brightness)
            before = F.adjust_contrast(before, contrast)
            before = F.adjust_saturation(before, saturation)
            before = F.adjust_hue(before, hue)
            
            after = F.adjust_brightness(after, brightness)
            after = F.adjust_contrast(after, contrast)
            after = F.adjust_saturation(after, saturation)
            after = F.adjust_hue(after, hue)

        # Resize to fixed size (e.g., back to 256)
        before = F.resize(before, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        after = F.resize(after, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        # To tensor + normalization
        before_t = F.to_tensor(before)
        after_t = F.to_tensor(after)

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


def _collate_change(batch):
    befores, afters, masks = zip(*batch)
    return (
        torch.stack(befores, dim=0),
        torch.stack(afters, dim=0),
        torch.stack(masks, dim=0),
    )


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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate_change,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate_change,
    )

    return train_loader, val_loader


def _create_model_from_factory(cfg: SiameseChangeConfig, device: torch.device) -> nn.Module:
    model = create_model(cfg)
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
    scaler: torch.cuda.amp.GradScaler | None = None,
    train: bool = True,
) -> Tuple[float, float]:
    """
    Run one epoch and return (mean_loss, mean_iou).
    Uses AMP if scaler is provided.
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

        cfg_model_type = getattr(model, "model_type", None) # Fallback is handled by instance check
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.set_grad_enabled(train):
                # If the model is a ResNet34 Siamese U-Net (which requires dual inputs)
                if hasattr(model, 'stem') or 'resnet' in str(type(model)).lower():
                    logits = model(before, after)
                else:
                    x = torch.cat([before, after], dim=1)  # [B, 6, H, W]
                    logits = model(x)
                    
                loss = criterion(logits, mask)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    model = _create_model_from_factory(cfg, device)

    criterion = HybridLoss(bce_weight=0.6, dice_weight=0.4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

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
            scaler=scaler,
            train=True,
        )

        val_loss, val_iou = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=scaler,
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
    model = _create_model_from_factory(cfg, device)

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

            # If the model is a ResNet34 Siamese U-Net
            if hasattr(model, 'stem') or 'resnet' in str(type(model)).lower():
                logits = model(before, after)
            else:
                x = torch.cat([before, after], dim=1)
                logits = model(x)
            batch_iou = _compute_iou(logits, mask)

            bsz = mask.size(0)
            total_iou += batch_iou * bsz
            total_samples += bsz

    return total_iou / total_samples if total_samples > 0 else 0.0

