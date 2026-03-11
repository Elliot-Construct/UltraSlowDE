import numpy as np

from ultra_slow_de.data_sources import built_in_sources
from ultra_slow_de.datasets import GaussianDataset, covariance_from_sigma
from ultra_slow_de.likelihood import dataset_loglike, gaussian_loglike, joint_loglike


def test_gaussian_loglike_matches_manual_diagonal_case():
    residual = np.array([1.0, -1.0])
    cov = np.diag([4.0, 9.0])
    got = gaussian_loglike(residual, cov)

    quad = (1.0**2) / 4.0 + ((-1.0) ** 2) / 9.0
    logdet = np.log(4.0 * 9.0)
    expected = -0.5 * (quad + logdet + 2.0 * np.log(2.0 * np.pi))
    assert np.isclose(got, expected)


def test_dataset_loglike_zero_residual_is_finite():
    src = built_in_sources()["desi_bao"]
    z = np.array([0.2, 0.5, 1.0])
    y = np.array([80.0, 95.0, 120.0])
    cov = covariance_from_sigma(np.array([2.0, 2.0, 2.0]))
    ds = GaussianDataset("desi", "bao", z, y, cov, src)
    ll = dataset_loglike(ds, y_model=y)
    assert np.isfinite(ll)


def test_joint_loglike_is_sum_of_parts():
    src = built_in_sources()["eboss_bao_rsd"]
    z = np.array([0.3, 0.7])
    y1 = np.array([90.0, 110.0])
    y2 = np.array([0.45, 0.50])
    c1 = covariance_from_sigma(np.array([3.0, 3.0]))
    c2 = covariance_from_sigma(np.array([0.05, 0.05]))
    ds1 = GaussianDataset("bao", "bao", z, y1, c1, src)
    ds2 = GaussianDataset("rsd", "rsd", z, y2, c2, src)

    ym1 = np.array([91.0, 109.0])
    ym2 = np.array([0.44, 0.51])
    ll1 = dataset_loglike(ds1, ym1)
    ll2 = dataset_loglike(ds2, ym2)
    llj = joint_loglike([(ds1, ym1), (ds2, ym2)])
    assert np.isclose(llj, ll1 + ll2)