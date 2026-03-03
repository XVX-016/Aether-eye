from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal envs
    def tqdm(iterable, **_: object):  # type: ignore
        return iterable

from aether_ml.config import FgvcVitConfig
from aether_ml.datasets import FgvcAircraftDataset


class FgvcWrappedDataset(torch.utils.data.Dataset):
    """Windows-safe transform wrapper for FGVC dataset."""

    def __init__(self, base_ds: FgvcAircraftDataset, tf: transforms.Compose):
        self.base_ds = base_ds
        self.tf = tf

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        img, label = self.base_ds[idx]
        return self.tf(img), label


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
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


def _compute_class_counts(ds: FgvcAircraftDataset) -> torch.Tensor:
    counts = torch.zeros(ds.num_classes, dtype=torch.long)
    for sample in ds.samples:
        counts[sample.label] += 1
    return counts


def _build_class_names(ds: FgvcAircraftDataset) -> List[str]:
    class_names = [""] * ds.num_classes
    for name, idx in ds.variant_to_idx.items():
        class_names[idx] = name
    for idx, name in enumerate(class_names):
        if not name:
            class_names[idx] = f"class_{idx}"
    return class_names


def _create_dataloaders(
    cfg: FgvcVitConfig,
) -> Tuple[DataLoader, DataLoader, int, torch.Tensor, List[str]]:
    cfg = cfg.resolved()
    train_tf, val_tf = _build_transforms(cfg.image_size)

    train_ds = FgvcAircraftDataset(root=cfg.data_root, split="train")
    val_ds = FgvcAircraftDataset(root=cfg.data_root, split="val")
    num_classes = train_ds.num_classes
    class_counts = _compute_class_counts(train_ds)
    class_names = _build_class_names(train_ds)

    train_wrapped = FgvcWrappedDataset(train_ds, train_tf)
    val_wrapped = FgvcWrappedDataset(val_ds, val_tf)

    train_sampler = None
    train_shuffle = True
    if cfg.imbalance_mode.lower() == "sampler":
        counts = class_counts.float().clamp(min=1.0)
        per_class_weight = 1.0 / counts
        sample_weights = torch.tensor(
            [float(per_class_weight[s.label].item()) for s in train_ds.samples],
            dtype=torch.double,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_wrapped,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
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

    return train_loader, val_loader, num_classes, class_counts, class_names


def _create_model(cfg: FgvcVitConfig, num_classes: int) -> nn.Module:
    return timm.create_model(
        cfg.model_name,
        pretrained=True,
        num_classes=num_classes,
    )


def _topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    return int(topk.eq(target.unsqueeze(1)).any(dim=1).sum().item())


def _metrics_from_confusion(conf: torch.Tensor) -> tuple[float, List[float]]:
    conf = conf.float()
    tp = conf.diag()
    support = conf.sum(dim=1).clamp(min=1.0)
    pred_pos = conf.sum(dim=0).clamp(min=1.0)

    recall = tp / support
    precision = tp / pred_pos
    f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-12)

    macro_f1 = float(f1.mean().item())
    per_class_recall = [float(v.item()) for v in recall]
    return macro_f1, per_class_recall


def _save_confusion_matrix_png(confusion: torch.Tensor, class_names: List[str], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except Exception:
        return

    arr = confusion.cpu().numpy().astype(np.float32)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    norm = arr / row_sums

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(norm, interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Validation Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Too many classes for full labels; show sparse ticks.
    num_classes = len(class_names)
    step = max(1, num_classes // 10)
    ticks = list(range(0, num_classes, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], rotation=90)
    ax.set_yticklabels([str(i) for i in ticks])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        return images, F.one_hot(labels, num_classes=num_classes).float()

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    y_a = F.one_hot(labels, num_classes=num_classes).float()
    y_b = F.one_hot(labels[perm], num_classes=num_classes).float()
    mixed_targets = lam * y_a + (1.0 - lam) * y_b
    return mixed_images, mixed_targets


def _weighted_soft_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    if class_weights is None:
        loss = -(targets * log_probs).sum(dim=1)
    else:
        w = class_weights.view(1, -1)
        loss = -((targets * w) * log_probs).sum(dim=1)
    return loss.mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_weights: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    num_classes: int,
    epoch_idx: int,
    total_epochs: int,
    mixup_alpha: float,
) -> Dict[str, object]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_total = 0
    running_top1 = 0
    running_top5 = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    amp_enabled = bool(use_amp and device.type == "cuda")

    phase = "train" if train else "val"
    desc = f"{phase} e{epoch_idx}/{total_epochs}"
    for images, labels in tqdm(loader, desc=desc, leave=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            effective_images = images
            soft_targets = None
            if train and mixup_alpha > 0.0:
                effective_images, soft_targets = _mixup_batch(images, labels, mixup_alpha, num_classes)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(effective_images)
                if soft_targets is None:
                    loss = criterion(outputs, labels)
                else:
                    loss = _weighted_soft_ce(outputs, soft_targets, class_weights)

        if train:
            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.size(0))
        running_loss += float(loss.item()) * batch_size
        running_total += batch_size
        running_top1 += _topk_correct(outputs, labels, 1)
        running_top5 += _topk_correct(outputs, labels, 5)

        preds = outputs.argmax(dim=1)
        pairs = torch.stack((labels.detach().cpu(), preds.detach().cpu()), dim=1)
        for true_idx, pred_idx in pairs.tolist():
            confusion[int(true_idx), int(pred_idx)] += 1

    mean_loss = running_loss / max(1, running_total)
    top1 = running_top1 / max(1, running_total)
    top5 = running_top5 / max(1, running_total)
    macro_f1, per_class_recall = _metrics_from_confusion(confusion)

    return {
        "loss": float(mean_loss),
        "top1": float(top1),
        "top5": float(top5),
        "macro_f1": float(macro_f1),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


def train_vit_aircraft(cfg: FgvcVitConfig) -> Dict[str, float]:
    cfg = cfg.resolved()
    use_cuda = torch.cuda.is_available() and "cpu" not in cfg.device.lower()
    device = torch.device(cfg.device if use_cuda else "cpu")

    train_loader, val_loader, num_classes, class_counts, class_names = _create_dataloaders(cfg)
    model = _create_model(cfg, num_classes=num_classes).to(device)

    class_weights = None
    if cfg.imbalance_mode.lower() == "weighted_loss":
        class_weights = 1.0 / class_counts.float().clamp(min=1.0)
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(cfg.label_smoothing),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs - cfg.warmup_epochs),
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=bool(cfg.use_amp and device.type == "cuda"),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_top1 = 0.0
    best_val_macro_f1 = -1.0
    best_model_path = cfg.output_dir / "best.pt"
    history: List[Dict[str, object]] = []
    best_confusion: torch.Tensor | None = None
    best_per_class_recall: List[float] = []

    no_improve_epochs = 0
    initial_lr = float(cfg.learning_rate)

    for epoch in range(cfg.epochs):
        # Linear warmup for early epochs to reduce unstable overfitting spikes.
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            warmup_scale = float(epoch + 1) / float(cfg.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = initial_lr * warmup_scale

        print(f"[Epoch {epoch + 1}/{cfg.epochs}] starting")
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            class_weights=class_weights,
            optimizer=optimizer,
            device=device,
            train=True,
            use_amp=cfg.use_amp,
            scaler=scaler,
            num_classes=num_classes,
            epoch_idx=epoch + 1,
            total_epochs=cfg.epochs,
            mixup_alpha=cfg.mixup_alpha,
        )

        with torch.no_grad():
            val_stats = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                class_weights=class_weights,
                optimizer=optimizer,
                device=device,
                train=False,
                use_amp=cfg.use_amp,
                scaler=scaler,
                num_classes=num_classes,
                epoch_idx=epoch + 1,
                total_epochs=cfg.epochs,
                mixup_alpha=0.0,
            )

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": {
                "loss": train_stats["loss"],
                "top1": train_stats["top1"],
                "top5": train_stats["top5"],
                "macro_f1": train_stats["macro_f1"],
            },
            "val": {
                "loss": val_stats["loss"],
                "top1": val_stats["top1"],
                "top5": val_stats["top5"],
                "macro_f1": val_stats["macro_f1"],
            },
        }
        history.append(epoch_record)
        print(
            "[Epoch {}/{}] train_loss={:.4f} train_top1={:.4f} val_loss={:.4f} val_top1={:.4f} val_top5={:.4f} val_macro_f1={:.4f}".format(
                epoch + 1,
                cfg.epochs,
                float(train_stats["loss"]),
                float(train_stats["top1"]),
                float(val_stats["loss"]),
                float(val_stats["top1"]),
                float(val_stats["top5"]),
                float(val_stats["macro_f1"]),
            )
        )

        val_top1 = float(val_stats["top1"])
        val_macro_f1 = float(val_stats["macro_f1"])
        if cfg.save_best and val_macro_f1 > best_val_macro_f1:
            best_val_top1 = val_top1
            best_val_macro_f1 = val_macro_f1
            best_confusion = val_stats["confusion"]  # type: ignore[assignment]
            best_per_class_recall = list(val_stats["per_class_recall"])  # type: ignore[arg-type]
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_top1": float(val_stats["top1"]),
                    "val_top5": float(val_stats["top5"]),
                    "val_macro_f1": float(val_stats["macro_f1"]),
                    "config": {
                        "model_name": cfg.model_name,
                        "image_size": cfg.image_size,
                        "num_classes": num_classes,
                    },
                    "class_names": class_names,
                },
                best_model_path,
            )
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= cfg.early_stopping_patience:
            print(
                "[EarlyStop] no val_macro_f1 improvement for {} epochs. Stopping at epoch {}.".format(
                    cfg.early_stopping_patience,
                    epoch + 1,
                )
            )
            break

    metrics_payload: Dict[str, object] = {
        "best_val_top1": float(best_val_top1),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_model_path": str(best_model_path) if best_model_path.is_file() else "",
        "num_classes": num_classes,
        "class_names": class_names,
        "class_counts_train": [int(v.item()) for v in class_counts],
        "history": history,
        "best_val_per_class_recall": best_per_class_recall,
    }

    metrics_path = cfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if best_confusion is not None:
        torch.save(best_confusion, cfg.output_dir / "confusion_matrix.pt")
        _save_confusion_matrix_png(best_confusion, class_names, cfg.output_dir / "confusion_matrix.png")

    return {
        "best_val_top1": float(best_val_top1),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_model_path": str(best_model_path) if best_model_path.is_file() else "",
        "metrics_path": str(metrics_path),
    }


def evaluate_vit_aircraft_top1(
    cfg: FgvcVitConfig,
    weights_path: str | Path,
) -> float:
    cfg = cfg.resolved()
    use_cuda = torch.cuda.is_available() and "cpu" not in cfg.device.lower()
    device = torch.device(cfg.device if use_cuda else "cpu")

    _, val_loader, num_classes, _, _ = _create_dataloaders(cfg)
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
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    return correct / total if total > 0 else 0.0
