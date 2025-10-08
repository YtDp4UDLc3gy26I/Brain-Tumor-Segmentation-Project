import torch

class EpochConfusionMeter:
    """Epoch-level multiclass confusion accumulator with derived metrics."""
    def __init__(self, num_classes: int = 4, device: torch.device | None = None):
        self.num_classes = num_classes
        self.conf = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device or 'cpu')
        self._device_fixed = device is not None

    @torch.no_grad()
    def update_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        if not self._device_fixed and self.conf.device != logits.device:
            self.conf = self.conf.to(logits.device); self._device_fixed = True
        preds = logits.argmax(1)  # [B,H,W]
        k = (targets >= 0) & (targets < self.num_classes)
        if not k.any(): return
        yv = targets[k].reshape(-1); pv = preds[k].reshape(-1)
        cm = torch.bincount(self.num_classes * yv + pv,
                            minlength=self.num_classes * self.num_classes)
        self.conf += cm.view(self.num_classes, self.num_classes)

    def reset(self):
        self.conf.zero_()

    @torch.no_grad()
    def _reduced_conf(self, include_background: bool, bg_class: int = 0):
        conf = self.conf.float()
        if include_background or self.num_classes <= 1:
            idx = torch.arange(self.num_classes, device=conf.device)
        else:
            idx = torch.tensor([i for i in range(self.num_classes) if i != bg_class],
                               device=conf.device)
        rc = conf.index_select(0, idx).index_select(1, idx)
        return rc, rc.sum()

    @torch.no_grad()
    def compute_all(self, include_background: bool = False, bg_class: int = 0):
        eps = 1e-7
        total = self.conf.sum().clamp_min(1.0)
        acc = self.conf.diag().sum() / total

        conf, tot = self._reduced_conf(include_background, bg_class)
        tp = conf.diag()
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        tn = tot - tp - fp - fn

        iou  = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)

        return {
            "accuracy": float(acc.item()),
            "mean_iou": float(iou.mean().item()),
            "dice_no_bg": float(dice.mean().item()),
            "precision": float(prec.mean().item()),
            "sensitivity": float(rec.mean().item()),
            "specificity": float(spec.mean().item()),
            "per_class": {
                "iou":  iou.tolist(),
                "dice": dice.tolist(),
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "specificity": spec.tolist(),
            },
        }