from __future__ import annotations
import os
from collections import OrderedDict
from typing import Sequence
import random
import numpy as np
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class BraTSSliceDataset(Dataset):
    """
    Convert 3D BraTS case volumes into a stream of 2D axial slices.
    Modalities are stacked as channels. Seg labels 4→3 remap is applied.
    """
    def __init__(
        self,
        roots: Sequence[str] | str,
        case_ids: Sequence[str],
        img_size: int,
        volume_slices: int,
        volume_start_at: int,
        cache_size: int,
        modalities: tuple[str, ...] = ("flair","t1ce"),
        normalize: str = "per_modality_p99",
        augment: bool = False,
    ):
        super().__init__()
        self.roots = list(roots) if isinstance(roots, (list, tuple)) else [roots]
        self.case_ids = list(case_ids)
        self.img_size = int(img_size)
        self.volume_slices = int(volume_slices)
        self.volume_start_at = int(volume_start_at)
        self.cache_size = int(cache_size)
        self.modalities = tuple(modalities)
        self.normalize = normalize
        self.augment = augment

        self.case_to_root = {}
        for cid in self.case_ids:
            found = None
            for r in self.roots:
                if os.path.isdir(os.path.join(r, cid)):
                    found = r; break
            if found is None:
                raise FileNotFoundError(f"Case {cid} not found under any of: {self.roots}")
            self.case_to_root[cid] = found

        self.cache: OrderedDict[str, dict] = OrderedDict()

        # Build indices using actual depth
        self.index = []
        for cid in self.case_ids:
            data = self._load_case(cid)  # cached once
            D = data["seg"].shape[2]
            start = min(self.volume_start_at, max(0, D-1))
            end = min(start + self.volume_slices, D)
            for s in range(start, end):
                self.index.append((cid, s))

    def __len__(self): return len(self.index)

    @staticmethod
    def _resize_volume(v: np.ndarray, out_hw: int, interp) -> np.ndarray:
        H = W = out_hw
        D = v.shape[2]
        out = np.empty((H, W, D), dtype=np.float32)
        for s in range(D):
            out[:, :, s] = cv2.resize(v[:, :, s], (W, H), interpolation=interp)
        return out

    def _load_case(self, case_id: str):
        if case_id in self.cache:
            self.cache.move_to_end(case_id)
            return self.cache[case_id]

        case_root = self.case_to_root[case_id]
        vols = {}
        for m in self.modalities:
            p = os.path.join(case_root, case_id, f"{case_id}_{m}.nii.gz")
            if not os.path.exists(p):
                p = os.path.join(case_root, case_id, f"{case_id}_{m}.nii")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing modality {m} for {case_id}")
            vols[m] = nib.load(p).get_fdata(dtype=np.float32)

        pseg = os.path.join(case_root, case_id, f"{case_id}_seg.nii.gz")
        if not os.path.exists(pseg):
            pseg = os.path.join(case_root, case_id, f"{case_id}_seg.nii")
        seg = nib.load(pseg).get_fdata(dtype=np.float32)

        # normalization
        if self.normalize == "global_max":
            case_max = float(max(v.max() for v in vols.values()))
            scales = {m: (case_max if case_max > 0 else 1.0) for m in vols}
        elif self.normalize == "per_modality_p99":
            scales = {}
            for m, v in vols.items():
                mask = v > 0
                scales[m] = float(np.percentile(v[mask], 99.0)) if np.any(mask) else 1.0
        else:
            raise ValueError(f"Unknown normalize='{self.normalize}'")

        eps = 1e-8
        mods_r = {}
        for m, v in vols.items():
            r = self._resize_volume(v, self.img_size, cv2.INTER_AREA)
            r = np.clip(r / (scales[m] + eps), 0.0, 1.0)
            mods_r[m] = r.astype(np.float32)

        seg_r = self._resize_volume(seg, self.img_size, cv2.INTER_NEAREST).astype(np.int16)
        seg_r[seg_r == 4] = 3

        X = np.stack([mods_r[m] for m in self.modalities], axis=0).astype(np.float32)
        data = {"X": X, "seg": seg_r.astype(np.int64)}
        self.cache[case_id] = data
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data

    def _augment(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C, H, W = X.shape
        k = random.randint(0, 3)
        if k:
            X = np.rot90(X, k=k, axes=(1, 2)).copy()
            y = np.rot90(y, k=k, axes=(0, 1)).copy()
        if random.random() < 0.5:
            X = X[:, :, ::-1].copy(); y = y[:, ::-1].copy()
        if random.random() < 0.5:
            X = X[:, ::-1, :].copy(); y = y[::-1, :].copy()
        for c in range(C):
            if random.random() < 0.5:
                gamma = random.uniform(0.95, 1.05)
                X[c] = np.clip(X[c] ** gamma, 0.0, 1.0)
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.01, size=X[c].shape).astype(np.float32)
                X[c] = np.clip(X[c] + noise, 0.0, 1.0)
        return X, y

    def __getitem__(self, idx: int):
        cid, sl = self.index[idx]
        data = self._load_case(cid)
        X = data["X"][:, :, :, sl]  # [C,H,W]
        y = data["seg"][:, :, sl]   # [H,W]
        if self.augment:
            X, y = self._augment(X, y)
        return torch.from_numpy(X).float(), torch.from_numpy(y).long()

def build_loaders(
    roots: list[str],
    train_ids: list[str], val_ids: list[str], test_ids: list[str],
    img_size: int, volume_slices: int, volume_start_at: int, cache_size: int,
    modalities: tuple[str, ...], batch_size: int, num_workers: int
):
    train_ds = BraTSSliceDataset(roots, train_ids, img_size, volume_slices, volume_start_at,
                                 cache_size, modalities=modalities, augment=True)
    val_ds   = BraTSSliceDataset(roots, val_ids, img_size, volume_slices, volume_start_at,
                                 cache_size, modalities=modalities, augment=False)
    test_ds  = BraTSSliceDataset(roots, test_ids, img_size, volume_slices, volume_start_at,
                                 cache_size, modalities=modalities, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=max(2, num_workers//2), pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=max(2, num_workers//2), pin_memory=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=max(2, num_workers//2), pin_memory=False, drop_last=False)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader