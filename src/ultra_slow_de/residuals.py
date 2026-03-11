import numpy as np


def delta_h(h_model: np.ndarray, h_baseline: np.ndarray) -> np.ndarray:
    hm = np.asarray(h_model, dtype=float)
    hb = np.asarray(h_baseline, dtype=float)
    return (hm - hb) / hb


def delta_dl(dl_model: np.ndarray, dl_baseline: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    dm = np.asarray(dl_model, dtype=float)
    db = np.asarray(dl_baseline, dtype=float)
    out = np.empty_like(dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:] = (dm - db) / db
    if z is not None:
        z_arr = np.asarray(z, dtype=float)
        out = np.where(np.isclose(z_arr, 0.0), 0.0, out)
    return out