from __future__ import annotations
import os
from collections import OrderedDict
from typing import Sequence
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..models.unet2d import UNet2D
from .overlays import _to01, _overlay_on_base, CLASS_COLORS

def imageLoader(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()

@torch.no_grad()
def load_model(
    weights_path: str,
    model_ctor=UNet2D,
    in_ch: int = 2,
    n_classes: int = 4,
    dropout: float = 0.35,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model_ctor(in_ch=in_ch, n_classes=n_classes, dropout=dropout).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        stg = state.get("settings", {})
        if "dropout" in stg and abs(stg["dropout"] - dropout) > 1e-8:
            model = model_ctor(in_ch=in_ch, n_classes=n_classes, dropout=stg["dropout"]).to(device)
        state = state["model_state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())

    incompatible = model.load_state_dict(state, strict=False)
    miss = getattr(incompatible, "missing_keys", []) or []
    unex = getattr(incompatible, "unexpected_keys", []) or []
    if miss or unex:
        print("==> load_state_dict report")
        print("   Missing keys:", miss)
        print("   Unexpected keys:", unex)
    model.eval()
    return model

@torch.no_grad()
def predict_volume_stack(
    case_dir: str, case_id: str, model,
    img_size: int = 128, volume_slices: int = 100, volume_start_at: int = 22,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    flair_path = os.path.join(case_dir, f"{case_id}_flair.nii.gz")
    if not os.path.exists(flair_path):
        flair_path = os.path.join(case_dir, f"{case_id}_flair.nii")
    t1ce_path  = os.path.join(case_dir, f"{case_id}_t1ce.nii.gz")
    if not os.path.exists(t1ce_path):
        t1ce_path  = os.path.join(case_dir, f"{case_id}_t1ce.nii")

    flair = imageLoader(flair_path).astype(np.float32)
    ce    = imageLoader(t1ce_path).astype(np.float32)

    def _p99_scale(vol: np.ndarray) -> float:
        m = vol > 0
        if np.any(m):
            s = float(np.percentile(vol[m], 99.0))
            if not np.isfinite(s) or s <= 0:
                s = float(vol[m].max())
        else:
            s = float(max(1.0, vol.max()))
        return s

    s_flair = _p99_scale(flair)
    s_ce    = _p99_scale(ce)

    X = np.empty((volume_slices, img_size, img_size, 2), dtype=np.float32)
    for j in range(volume_slices):
        z = j + volume_start_at
        X[j, :, :, 0] = cv2.resize(flair[:, :, z], (img_size, img_size), interpolation=cv2.INTER_AREA)
        X[j, :, :, 1] = cv2.resize(ce[:, :, z],    (img_size, img_size), interpolation=cv2.INTER_AREA)

    eps = 1e-8
    X[..., 0] = np.clip(X[..., 0] / (s_flair + eps), 0.0, 1.0)
    X[..., 1] = np.clip(X[..., 1] / (s_ce + eps),    0.0, 1.0)

    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().to(device)
    logits = model(Xt)
    probs  = F.softmax(logits, dim=1).cpu().numpy()
    return np.transpose(probs, (0, 2, 3, 1))  # [B,H,W,C]

@torch.no_grad()
def showPredictsById(
    case: str, root_dir: str, model,
    img_size: int = 128, volume_slices: int = 100, volume_start_at: int = 22,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    start_slice: int = 60
):
    case_dir = os.path.join(root_dir, case)
    p = predict_volume_stack(case_dir, case, model, img_size, volume_slices, volume_start_at, device)

    gt_path = os.path.join(case_dir, f"{case}_seg.nii.gz")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(case_dir, f"{case}_seg.nii")
    flair_path = os.path.join(case_dir, f"{case}_flair.nii.gz")
    if not os.path.exists(flair_path):
        flair_path = os.path.join(case_dir, f"{case}_flair.nii")

    gt_vol = imageLoader(gt_path)
    flair  = imageLoader(flair_path)
    z  = start_slice
    import cv2 as _cv2
    fl = _cv2.resize(flair[:, :, z + volume_start_at], (img_size, img_size),
                     interpolation=_cv2.INTER_AREA)
    gt = _cv2.resize(gt_vol[:, :, z + volume_start_at], (img_size, img_size),
                     interpolation=_cv2.INTER_NEAREST)
    gt[gt == 4] = 3

    base = np.stack([_to01(fl)] * 3, axis=-1)

    gt_rgb = base.copy()
    for cls, col in CLASS_COLORS.items():
        gt_rgb = _overlay_on_base(gt_rgb, (gt == cls), color=col, alpha=0.45)

    pred_lbl = np.argmax(p[z, :, :, :], axis=-1)
    all_rgb  = base.copy()
    for cls, col in CLASS_COLORS.items():
        all_rgb = _overlay_on_base(all_rgb, (pred_lbl == cls), color=col, alpha=0.45)

    nec_rgb = _overlay_on_base(base, (pred_lbl == 1), color=CLASS_COLORS[1], alpha=0.45)
    ede_rgb = _overlay_on_base(base, (pred_lbl == 2), color=CLASS_COLORS[2], alpha=0.45)
    enh_rgb = _overlay_on_base(base, (pred_lbl == 3), color=CLASS_COLORS[3], alpha=0.45)

    fig, axarr = plt.subplots(1, 6, figsize=(18, 5))
    axarr[0].imshow(_to01(fl), cmap="gray"); axarr[0].set_title("FLAIR"); axarr[0].axis("off")
    axarr[1].imshow(gt_rgb); axarr[1].set_title("Ground truth"); axarr[1].axis("off")
    axarr[2].imshow(all_rgb); axarr[2].set_title("All predicted"); axarr[2].axis("off")
    axarr[3].imshow(nec_rgb); axarr[3].set_title("NCR/NET"); axarr[3].axis("off")
    axarr[4].imshow(ede_rgb); axarr[4].set_title("Edema"); axarr[4].axis("off")
    axarr[5].imshow(enh_rgb); axarr[5].set_title("Enhancing"); axarr[5].axis("off")
    plt.tight_layout()
    return fig