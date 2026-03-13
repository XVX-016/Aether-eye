from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: object):  # type: ignore
        return iterable

from aether_ml.config import ChangeUnetConfig
from aether_ml.datasets import LevirChangeDataset
from aether_ml.evaluation.metrics import HybridTverskyLoss
from aether_ml.models.siamese_unet_resnet34 import SiameseUNetResNet34


class PairedTransformTrain:
    def __init__(self, image_size: int, min_change_ratio: float = 0.005, crop_retries: int = 10) -> None:
        self.image_size = int(image_size)
        self.min_change_ratio = float(min_change_ratio)
        self.crop_retries = int(crop_retries)

    def __call__(self, before, after, mask):
        # RandomResizedCrop-style positive-aware sampling reduces all-background
        # batches while improving scale robustness.
        chosen = None
        for _ in range(max(1, self.crop_retries)):
            scale = random.uniform(0.6, 1.4)
            crop_h = max(1, min(before.size[1], int(round(self.image_size / max(scale, 1e-6)))))
            crop_w = max(1, min(before.size[0], int(round(self.image_size / max(scale, 1e-6)))))
            i, j, h, w = transforms.RandomCrop.get_params(before, output_size=(crop_h, crop_w))
            m = TF.crop(mask, i, j, h, w)
            m_np = np.asarray(m, dtype=np.uint8)
            ratio = float((m_np > 127).mean())
            chosen = (i, j, h, w)
            if ratio >= self.min_change_ratio:
                break
        assert chosen is not None
        i, j, h, w = chosen
        before = TF.crop(before, i, j, h, w)
        after = TF.crop(after, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() < 0.5:
            before = TF.hflip(before)
            after = TF.hflip(after)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            before = TF.vflip(before)
            after = TF.vflip(after)
            mask = TF.vflip(mask)

        before = TF.resize(before, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        after = TF.resize(after, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        before_t = TF.to_tensor(before)
        after_t = TF.to_tensor(after)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        before_t = (before_t - mean) / std
        after_t = (after_t - mean) / std

        mask_t = TF.to_tensor(mask)  # [1,H,W], usually 0/255-like
        mask_t = (mask_t > (127.0 / 255.0)).float()
        return before_t, after_t, mask_t


class PairedTransformVal:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, before, after, mask):
        # Keep native val resolution (LEVIR is typically 1024x1024) to avoid
        # compressing tiny change regions into ambiguous pixels.
        # If an image is smaller than crop size, resize up minimally.
        if before.size[0] < self.image_size or before.size[1] < self.image_size:
            before = TF.resize(before, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            after = TF.resize(after, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        before_t = TF.to_tensor(before)
        after_t = TF.to_tensor(after)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        before_t = (before_t - mean) / std
        after_t = (after_t - mean) / std
        mask_t = (TF.to_tensor(mask) > (127.0 / 255.0)).float()
        return before_t, after_t, mask_t


def _collate(batch):
    befores, afters, masks = zip(*batch)
    return torch.stack(befores), torch.stack(afters), torch.stack(masks)


def _create_dataloaders(cfg: ChangeUnetConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = LevirChangeDataset(root=cfg.root, split="train", transform=PairedTransformTrain(cfg.image_size))
    val_ds = LevirChangeDataset(root=cfg.root, split="val", transform=PairedTransformVal(cfg.image_size))
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
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    return train_loader, val_loader


def _compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | list[float]]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    tp = float(((preds == 1) & (targets == 1)).sum().item())
    fp = float(((preds == 1) & (targets == 0)).sum().item())
    fn = float(((preds == 0) & (targets == 1)).sum().item())
    tn = float(((preds == 0) & (targets == 0)).sum().item())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    per_image_iou: list[float] = []
    b = preds.shape[0]
    for i in range(b):
        p = preds[i].view(-1)
        t = targets[i].view(-1)
        inter = float((p * t).sum().item())
        union = float(p.sum().item() + t.sum().item() - inter)
        per_image_iou.append((inter + 1e-6) / (union + 1e-6))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_acc": pixel_acc,
        "per_image_iou": per_image_iou,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    epoch_idx: int,
    total_epochs: int,
) -> dict[str, object]:
    if train:
        model.train()
    else:
        model.eval()

    amp_enabled = bool(use_amp and device.type == "cuda")
    total_loss = 0.0
    total_items = 0
    tot_tp = tot_fp = tot_fn = tot_tn = 0.0
    per_image_iou_all: list[float] = []

    phase = "train" if train else "val"
    for before, after, mask in tqdm(loader, desc=f"{phase} e{epoch_idx}/{total_epochs}", leave=True):
        before = before.to(device, non_blocking=True)
        after = after.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(before, after)
                loss = criterion(logits, mask)

        if train and optimizer is not None:
            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bsz = int(mask.shape[0])
        total_items += bsz
        total_loss += float(loss.item()) * bsz
        m = _compute_batch_metrics(logits.detach(), mask.detach())
        tot_tp += float(m["tp"])
        tot_fp += float(m["fp"])
        tot_fn += float(m["fn"])
        tot_tn += float(m["tn"])
        per_image_iou_all.extend(m["per_image_iou"])  # type: ignore[arg-type]

    precision = tot_tp / (tot_tp + tot_fp + 1e-6)
    recall = tot_tp / (tot_tp + tot_fn + 1e-6)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)
    iou = tot_tp / (tot_tp + tot_fp + tot_fn + 1e-6)
    pixel_acc = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn + 1e-6)

    return {
        "loss": total_loss / max(1, total_items),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_acc": pixel_acc,
        "per_image_iou": per_image_iou_all,
    }


def train_change_unet(cfg: ChangeUnetConfig) -> Dict[str, float]:
    cfg = cfg.resolved()
    use_cuda = torch.cuda.is_available() and "cpu" not in cfg.device.lower()
    device = torch.device(cfg.device if use_cuda else "cpu")

    train_loader, val_loader = _create_dataloaders(cfg)
    model = SiameseUNetResNet34(pretrained=True).to(device)
    criterion = HybridTverskyLoss(bce_weight=0.4, tversky_weight=0.6, alpha=0.3, beta=0.7, gamma=0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=bool(cfg.amp and device.type == "cuda"))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_payload: dict[str, float] = {}
    best_model_path = cfg.output_dir / "best.pt"
    history: list[dict[str, object]] = []
    no_improve = 0

    for epoch in range(cfg.epochs):
        print(f"[Epoch {epoch + 1}/{cfg.epochs}] starting")
        train_stats = _run_epoch(
            model, train_loader, criterion, optimizer, device, True, scaler, cfg.amp, epoch + 1, cfg.epochs
        )
        with torch.no_grad():
            val_stats = _run_epoch(
                model, val_loader, criterion, None, device, False, scaler, cfg.amp, epoch + 1, cfg.epochs
            )
        scheduler.step()

        rec = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": {k: train_stats[k] for k in ["loss", "iou", "f1", "precision", "recall", "pixel_acc"]},
            "val": {k: val_stats[k] for k in ["loss", "iou", "f1", "precision", "recall", "pixel_acc"]},
            "val_per_image_iou": val_stats["per_image_iou"],
        }
        history.append(rec)

        print(
            "[Epoch {}/{}] train_loss={:.4f} train_iou={:.4f} val_iou={:.4f} val_f1={:.4f} val_prec={:.4f} val_rec={:.4f} val_acc={:.4f}".format(
                epoch + 1,
                cfg.epochs,
                float(train_stats["loss"]),
                float(train_stats["iou"]),
                float(val_stats["iou"]),
                float(val_stats["f1"]),
                float(val_stats["precision"]),
                float(val_stats["recall"]),
                float(val_stats["pixel_acc"]),
            )
        )

        val_f1 = float(val_stats["f1"])
        if cfg.save_best and val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            best_payload = {
                "best_epoch": float(epoch + 1),
                "best_val_f1": float(val_stats["f1"]),
                "best_val_iou": float(val_stats["iou"]),
                "best_val_precision": float(val_stats["precision"]),
                "best_val_recall": float(val_stats["recall"]),
                "best_val_pixel_accuracy": float(val_stats["pixel_acc"]),
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_f1": float(val_stats["f1"]),
                    "val_iou": float(val_stats["iou"]),
                    "val_precision": float(val_stats["precision"]),
                    "val_recall": float(val_stats["recall"]),
                    "config": {
                        "model_name": "siamese_unet_resnet34",
                        "image_size": cfg.image_size,
                    },
                },
                best_model_path,
            )
        else:
            no_improve += 1

        if no_improve >= cfg.early_stopping_patience:
            print(
                f"[EarlyStop] no val_f1 improvement for {cfg.early_stopping_patience} epochs. Stopping at epoch {epoch + 1}."
            )
            break

    metrics = {
        **best_payload,
        "best_model_path": str(best_model_path) if best_model_path.is_file() else "",
    }
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return metrics
