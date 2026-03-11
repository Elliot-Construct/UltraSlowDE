"""Dataset ingestion conventions.

Defines loaders that convert raw public data files into GaussianDataset
objects compatible with the likelihood pipeline.  Each loader:
  1. Reads from ``data/<dataset_id>/`` relative to project root.
  2. Returns a GaussianDataset with the correct SourceRecord attached.
  3. Selects only the columns/rows relevant to a given observable.
"""

from pathlib import Path

import numpy as np

from .data_sources import SourceRecord, acquire_dataset, built_in_sources
from .datasets import GaussianDataset


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def load_pantheon_plus(data_dir: Path | None = None,
                       hubble_flow_only: bool = False) -> GaussianDataset:
    """Load Pantheon+ standardised apparent magnitudes + full covariance.

    Parameters
    ----------
    data_dir : root data directory (default: ``<project>/data``)
    hubble_flow_only : if True, keep only the 277 SH0ES Hubble-flow SNe.
        If False (default), keep the 1624 non-calibrator SNe used for
        the standard Pantheon+ cosmological analysis.

    Returns
    -------
    GaussianDataset with kind='mu', y_obs = m_b_corr, cov = STAT+SYS
    """
    root = data_dir or DATA_ROOT
    base = acquire_dataset("pantheon_plus", root)  # raises if missing

    dat_file = base / "Pantheon+SH0ES.dat"
    cov_file = base / "Pantheon+SH0ES_STAT+SYS.cov"

    # --- data vector ---
    with open(dat_file, "r") as f:
        header = f.readline().split()
    col_z = header.index("zHD")
    col_mb = header.index("m_b_corr")
    col_cal = header.index("IS_CALIBRATOR")
    col_hf = header.index("USED_IN_SH0ES_HF")

    raw = np.genfromtxt(dat_file, skip_header=1, dtype=float,
                        usecols=[col_z, col_mb, col_cal, col_hf])
    z_all = raw[:, 0]
    mb_all = raw[:, 1]
    is_cal = raw[:, 2]
    hf_flag = raw[:, 3]

    # --- covariance ---
    with open(cov_file, "r") as f:
        n_cov = int(f.readline().strip())
        cov_flat = np.fromfile(f, sep="\n", count=n_cov * n_cov)
    cov_full = cov_flat.reshape(n_cov, n_cov)

    assert n_cov == len(z_all), (
        f"Covariance dimension {n_cov} != data length {len(z_all)}"
    )

    if hubble_flow_only:
        mask = hf_flag == 1
    else:
        mask = is_cal == 0  # all non-calibrator SNe

    z_sel = z_all[mask]
    mb_sel = mb_all[mask]
    idx = np.where(mask)[0]
    cov_sel = cov_full[np.ix_(idx, idx)]

    src = built_in_sources()["pantheon_plus"]
    return GaussianDataset(
        name="pantheon_plus",
        kind="mb",
        z=z_sel,
        y_obs=mb_sel,
        cov=cov_sel,
        source=src,
    )


def load_planck_compressed(
    R: float = 1.7502,
    lA: float = 301.471,
    omega_b: float = 0.02236,
    sigma_R: float = 0.0046,
    sigma_lA: float = 0.090,
    sigma_omega_b: float = 0.00015,
) -> dict[str, float]:
    """Return Planck 2018 compressed distance priors (Table 6).

    These are scalar constraints, not a GaussianDataset.  The inference
    module evaluates them analytically.
    """
    return {
        "R": R, "sigma_R": sigma_R,
        "lA": lA, "sigma_lA": sigma_lA,
        "omega_b": omega_b, "sigma_omega_b": sigma_omega_b,
    }
