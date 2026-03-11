import numpy as np

from ultra_slow_de.residuals import delta_dl, delta_h


def test_delta_h_zero_when_arrays_match():
    x = np.array([1.0, 2.0, 3.0])
    d = delta_h(x, x)
    assert np.allclose(d, 0.0)


def test_delta_dl_zero_when_arrays_match_including_z0_override():
    z = np.array([0.0, 0.5, 1.0])
    x = np.array([0.0, 10.0, 20.0])
    d = delta_dl(x, x, z=z)
    assert np.allclose(d, 0.0)