from dataclasses import dataclass

import numpy as np

from .data_sources import SourceRecord


@dataclass(frozen=True)
class GaussianDataset:
    name: str
    kind: str
    z: np.ndarray
    y_obs: np.ndarray
    cov: np.ndarray
    source: SourceRecord


def covariance_from_sigma(sigma: np.ndarray) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=float)
    return np.diag(sigma_arr**2)


def validate_dataset(ds: GaussianDataset) -> None:
    z = np.asarray(ds.z, dtype=float)
    y = np.asarray(ds.y_obs, dtype=float)
    c = np.asarray(ds.cov, dtype=float)

    if z.ndim != 1 or y.ndim != 1:
        raise ValueError("z and y_obs must be 1D arrays")
    if len(z) != len(y):
        raise ValueError("z and y_obs length mismatch")
    if c.shape != (len(y), len(y)):
        raise ValueError("cov shape must match y_obs length")
    if not np.all(np.isfinite(c)):
        raise ValueError("cov contains non-finite values")
    if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov must be symmetric")

    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov must be positive definite") from exc