"""Built-in cosmological datasets from published survey results.

These datasets are hardcoded from peer-reviewed tables so the pipeline
can run without downloading external files.  Each function returns a
GaussianDataset ready for the likelihood engine.

Sources:
    - DESI BAO DR2: DESI Collaboration 2025, arXiv:2503.14738,
        desi_gaussian_bao_ALL_GCcomb_mean/cov
  - eBOSS DR16: Alam et al. 2021, PRD 103, 083533, Table 3
  - Planck 2018 compressed priors: Planck 2020, A&A 641, A6, Table 1
"""

import numpy as np

from .data_sources import SourceRecord, built_in_sources
from .datasets import GaussianDataset


# =========================================================================
#  DESI BAO DR2  (arXiv:2503.14738; Y3 BAO catalog)
# =========================================================================
# Isotropic D_V/r_d measurements
_DESI_DV = {
    "z_eff": np.array([0.295]),
    "dv_rd": np.array([7.94167639]),
    "sigma": np.array([0.07609196324185624]),
}

# Anisotropic D_M/r_d measurements
_DESI_DM = {
    "z_eff": np.array([0.510, 0.706, 0.934, 1.321, 1.484, 2.330]),
    "dm_rd": np.array([13.58758434, 17.35069094, 21.57563956, 27.60085612, 30.51190063, 38.988973961958784]),
    "sigma": np.array([0.16836678472905517, 0.17993122074837373, 0.1617815860968114, 0.32455587500459765, 0.763557644844186, 0.5316820280957407]),
}

# Anisotropic D_H/r_d measurements (same z_eff as D_M)
_DESI_DH = {
    "z_eff": np.array([0.510, 0.706, 0.934, 1.321, 1.484, 2.330]),
    "dh_rd": np.array([21.86294686, 19.45534918, 17.64149464, 14.17602155, 12.81699964, 8.631545674846294]),
    "sigma": np.array([0.42886832478046216, 0.33387003159912393, 0.20104324858099562, 0.22455135092000672, 0.5180117691713191, 0.10106245296844917]),
}


def load_desi_dv() -> GaussianDataset:
    """DESI DR2 isotropic BAO: D_V/r_d at z_eff = {0.295}."""
    src = built_in_sources()["desi_bao"]
    return GaussianDataset(
        name="desi_dr2_dv",
        kind="dv_rd",
        z=_DESI_DV["z_eff"],
        y_obs=_DESI_DV["dv_rd"],
        cov=np.diag(_DESI_DV["sigma"] ** 2),
        source=src,
    )


def load_desi_dm() -> GaussianDataset:
    """DESI DR2 anisotropic BAO: D_M/r_d at 6 effective redshifts."""
    src = built_in_sources()["desi_bao"]
    return GaussianDataset(
        name="desi_dr2_dm",
        kind="dm_rd",
        z=_DESI_DM["z_eff"],
        y_obs=_DESI_DM["dm_rd"],
        cov=np.diag(_DESI_DM["sigma"] ** 2),
        source=src,
    )


def load_desi_dh() -> GaussianDataset:
    """DESI DR2 anisotropic BAO: D_H/r_d at 6 effective redshifts."""
    src = built_in_sources()["desi_bao"]
    return GaussianDataset(
        name="desi_dr2_dh",
        kind="dh_rd",
        z=_DESI_DH["z_eff"],
        y_obs=_DESI_DH["dh_rd"],
        cov=np.diag(_DESI_DH["sigma"] ** 2),
        source=src,
    )


# =========================================================================
#  eBOSS DR16  (Alam et al. 2021, PRD 103, 083533, Table 3)
# =========================================================================
_EBOSS_DM = {
    "z_eff": np.array([0.698, 1.480, 2.334]),
    "dm_rd": np.array([17.86, 30.69, 37.60]),
    "sigma": np.array([0.33, 0.80, 1.90]),
}

_EBOSS_DH = {
    "z_eff": np.array([0.698, 1.480, 2.334]),
    "dh_rd": np.array([19.33, 13.26, 8.93]),
    "sigma": np.array([0.53, 0.55, 0.28]),
}

_EBOSS_FSIG8 = {
    "z_eff": np.array([0.698, 1.480]),
    "fsig8": np.array([0.473, 0.462]),
    "sigma": np.array([0.044, 0.045]),
}


def load_eboss_dm() -> GaussianDataset:
    """eBOSS DR16 anisotropic BAO: D_M/r_d."""
    src = built_in_sources()["eboss_bao_rsd"]
    return GaussianDataset(
        name="eboss_dr16_dm",
        kind="dm_rd",
        z=_EBOSS_DM["z_eff"],
        y_obs=_EBOSS_DM["dm_rd"],
        cov=np.diag(_EBOSS_DM["sigma"] ** 2),
        source=src,
    )


def load_eboss_dh() -> GaussianDataset:
    """eBOSS DR16 anisotropic BAO: D_H/r_d."""
    src = built_in_sources()["eboss_bao_rsd"]
    return GaussianDataset(
        name="eboss_dr16_dh",
        kind="dh_rd",
        z=_EBOSS_DH["z_eff"],
        y_obs=_EBOSS_DH["dh_rd"],
        cov=np.diag(_EBOSS_DH["sigma"] ** 2),
        source=src,
    )


def load_eboss_fsig8() -> GaussianDataset:
    """eBOSS DR16 RSD: fσ₈(z) growth-rate measurements."""
    src = built_in_sources()["eboss_bao_rsd"]
    return GaussianDataset(
        name="eboss_dr16_fsig8",
        kind="fsig8",
        z=_EBOSS_FSIG8["z_eff"],
        y_obs=_EBOSS_FSIG8["fsig8"],
        cov=np.diag(_EBOSS_FSIG8["sigma"] ** 2),
        source=src,
    )


# =========================================================================
#  Planck 2018 compressed distance priors
#  (Planck 2020, A&A 641, A6, Table 1 — "shift parameter" formulation)
# =========================================================================
_PLANCK_COMPRESSED = {
    "R": 1.7502,          # shift parameter
    "lA": 301.471,        # acoustic scale
    "omega_b": 0.02236,   # baryon density
    "sigma_R": 0.0046,
    "sigma_lA": 0.090,
    "sigma_omega_b": 0.00015,
}


def load_planck_compressed() -> GaussianDataset:
    """Planck 2018 compressed CMB distance priors as a 2-point dataset.

    Observable vector: [R, l_A].  ω_b is not used as a data point
    because it does not directly constrain late-time DE; instead it
    feeds into the sound-horizon calculation.
    """
    src = built_in_sources()["planck_cmb"]
    y = np.array([_PLANCK_COMPRESSED["R"], _PLANCK_COMPRESSED["lA"]])
    sigma = np.array([_PLANCK_COMPRESSED["sigma_R"],
                      _PLANCK_COMPRESSED["sigma_lA"]])
    return GaussianDataset(
        name="planck_2018_compressed",
        kind="cmb_compressed",
        z=np.array([1089.92, 1089.92]),   # z* for both R and l_A
        y_obs=y,
        cov=np.diag(sigma ** 2),
        source=src,
    )


# =========================================================================
#  Convenience: load all BAO datasets at once
# =========================================================================
def load_all_bao() -> list[tuple[GaussianDataset, str]]:
    """Return all BAO datasets paired with their observable name."""
    return [
        (load_desi_dv(), "dv_rd"),
        (load_desi_dm(), "dm_rd"),
        (load_desi_dh(), "dh_rd"),
        (load_eboss_dm(), "dm_rd"),
        (load_eboss_dh(), "dh_rd"),
    ]
