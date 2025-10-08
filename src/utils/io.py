import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

EXTS = (".nii.gz", ".nii")

def make_output_dirs(run_id: str, root: str = "Output"):
    base = Path(root) / f"Output_{run_id}"
    paths = {
        "base": base,
        "weights": base / "weights",
        "logs": base / "logs",
        "model": base / "model",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def save_splits(log_dir: Path, run_id: str, splits: dict):
    with open(log_dir / f"splits_{run_id}.json", "w") as f:
        json.dump(splits, f, indent=2)

def _resolve_case_file(case_root: str, case_id: str, suffix: str):
    base = os.path.join(case_root, case_id, f"{case_id}_{suffix}")
    for ext in EXTS:
        p = base + ext
        if os.path.exists(p):
            return p
    return None

def _case_has_all(case_root: str, case_id: str, modalities: tuple[str, ...], require_seg: bool = True) -> bool:
    for m in modalities:
        if _resolve_case_file(case_root, case_id, m) is None:
            return False
    if require_seg and _resolve_case_file(case_root, case_id, "seg") is None:
        return False
    return True

def list_case_ids_under(root_dir: str, modalities: tuple[str, ...], require_seg: bool = True):
    if not (root_dir and os.path.isdir(root_dir)):
        return []
    subdirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    ids = []
    for p in subdirs:
        cid = p[p.rfind(os.sep)+1:]
        if re.match(r"BraTS20_Training_\d+|BraTS20_Validation_\d+", cid):
            if _case_has_all(root_dir, cid, modalities, require_seg=require_seg):
                ids.append(cid)
    return sorted(ids)

# ---- plotting ----
def save_pair_plot_final(train_vals, val_vals, metric_name, save_dir: Path, run_id: str | None = None):
    if not train_vals and not val_vals:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.clf()
    if train_vals:
        x = np.arange(1, len(train_vals) + 1)
        plt.plot(x, train_vals, label=f"train_{metric_name}")
    if val_vals:
        xv = np.arange(1, len(val_vals) + 1)
        plt.plot(xv, val_vals, label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name.replace("_", " ").title())
    plt.legend(); plt.grid(True); plt.tight_layout()
    fname = f"{metric_name}_{run_id}.png" if run_id else f"{metric_name}.png"
    plt.savefig(save_dir / fname, dpi=120)
    plt.close()

def save_final_plots(history: dict, save_dir: Path, run_id: str | None = None):
    for m in ["accuracy", "dice_no_bg", "loss", "mean_iou"]:
        train_vals = history.get(m, [])
        val_vals   = history.get(f"val_{m}", [])
        save_pair_plot_final(train_vals, val_vals, m, save_dir, run_id=run_id)