"""rocket_fft.special matches scipy.special, in Python and in jitted code."""

import numba as nb
import numpy as np
import pytest
import scipy.special
from numba.core.errors import TypingError

from rocket_fft.special import loggamma, poch


@nb.njit
def jit_loggamma(z):
    return loggamma(z)


@nb.njit
def jit_poch(z, m):
    return poch(z, m)


REAL = np.array([-np.inf, -2.0, -1.5, -0.0, 0.0, 1e-8, 0.5, 1.0, 2.0, 7.5, 1e3, np.inf, np.nan])
rng = np.random.default_rng(0)
COMPLEX = (rng.uniform(-20, 20, 200) + 1j * rng.uniform(-20, 20, 200))


@pytest.mark.parametrize("fn", [loggamma, jit_loggamma])
def test_loggamma_real(fn):
    np.testing.assert_allclose(fn(REAL), scipy.special.loggamma(REAL), rtol=1e-13)
    np.testing.assert_allclose(fn(REAL.astype(np.float32)), scipy.special.loggamma(REAL), rtol=1e-6)


@pytest.mark.parametrize("fn", [loggamma, jit_loggamma])
def test_loggamma_complex(fn):
    np.testing.assert_allclose(fn(COMPLEX), scipy.special.loggamma(COMPLEX), rtol=1e-12)
    assert fn(COMPLEX.astype(np.complex64)).dtype == np.complex128


def test_loggamma_scalars():
    for z in [0.5, 3.0, 1 + 2j, -2.5 + 0.5j]:
        assert np.isclose(jit_loggamma(z), scipy.special.loggamma(z))
        assert np.isclose(loggamma(z), scipy.special.loggamma(z))


@pytest.mark.parametrize("fn", [poch, jit_poch])
def test_poch(fn):
    z = np.array([-3.0, -2.5, -1.0, 0.0, 0.5, 1.0, 4.2, 1e5])
    m = np.array([-1.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.7])
    zz, mm = np.meshgrid(z, m)
    np.testing.assert_allclose(fn(zz, mm), scipy.special.poch(zz, mm), rtol=1e-12)


def test_invalid_types():
    with pytest.raises(TypingError):
        nb.njit(lambda: poch(1.0 + 1j, 1.0))()
