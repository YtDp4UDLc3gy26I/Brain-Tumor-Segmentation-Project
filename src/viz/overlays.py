import numpy as np

CLASS_COLORS = {
    1: (1.0, 0.30, 0.30),  # red-ish (NCR/NET)
    2: (0.30, 1.0, 0.30),  # green-ish (Edema)
    3: (0.30, 0.50, 1.0),  # blue-ish (Enhancing)
}

def _to01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-6)

def _overlay_on_base(base_rgb: np.ndarray, mask: np.ndarray,
                     color=(1.0, 0.3, 0.3), alpha=0.45):
    out = base_rgb.copy()
    if mask.any():
        c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        out[mask] = (1 - alpha) * out[mask] + alpha * c
    return out