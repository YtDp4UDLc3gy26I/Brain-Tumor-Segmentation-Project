from __future__ import annotations
import torch
import torch.nn.functional as F
from ..metrics.confusion_meter import EpochConfusionMeter
from ..losses.combined import soft_dice_no_bg

def _ce_and_dice_parts(logits, targets):
    ce   = F.cross_entropy(logits, targets)
    dice = soft_dice_no_bg(logits, targets)
    return ce, dice

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = ce_sum = dice_sum = 0.0
    meter = EpochConfusionMeter(num_classes=4, device=device)

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward(); optimizer.step()
        total_loss += float(loss.item())

        ce_b, dice_b = _ce_and_dice_parts(logits.detach(), y)
        ce_sum   += float(ce_b.item())
        dice_sum += float(dice_b.item())
        meter.update_logits(logits, y)

    metrics = meter.compute_all(include_background=False)
    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "ce_loss": ce_sum / n,
        "dice_loss": dice_sum / n,
        "accuracy": metrics["accuracy"],
        "mean_iou": metrics["mean_iou"],
        "dice_no_bg": metrics["dice_no_bg"],
        "precision": metrics["precision"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
    }

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes=4):
    model.eval()
    total_loss = ce_sum = dice_sum = 0.0
    meter = EpochConfusionMeter(num_classes=num_classes, device=device)

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total_loss += float(loss_fn(logits, y).item())
        ce_b, dice_b = _ce_and_dice_parts(logits, y)
        ce_sum   += float(ce_b.item())
        dice_sum += float(dice_b.item())
        meter.update_logits(logits, y)

    metrics = meter.compute_all(include_background=False)
    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "ce_loss": ce_sum / n,
        "dice_loss": dice_sum / n,
        "accuracy": metrics["accuracy"],
        "mean_iou": metrics["mean_iou"],
        "dice_no_bg": metrics["dice_no_bg"],
        "precision": metrics["precision"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
    }