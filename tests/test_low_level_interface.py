"""
These tests are a modified version of pypocketfft's tests:
https://gitlab.mpcdf.mpg.de/mtr/pypocketfft/-/blob/master/test.py
"""

from functools import partial

import numba as nb
import numpy as np
import pytest
from helpers import numba_cache_cleanup
from numpy.testing import assert_

# Only test the functions that are not used in the SciPy or NumPy interface
# c2c is needed for comparison
from rocket_fft import (c2c, r2r_fftpack, r2r_genuine_hartley,
                        r2r_separable_hartley)

# All functions should be cacheable and run without the GIL
njit = partial(nb.njit, cache=True, nogil=True)


@njit
def jit_c2c(ain, aout, axes, forward, fct, nthreads):
    c2c(ain, aout, axes, forward, fct, nthreads)
    return aout


@njit
def jit_r2r_separable_hartley(ain, aout, axes, fct, nthreads):
    r2r_separable_hartley(ain, aout, axes, fct, nthreads)
    return aout


@njit
def jit_r2r_genuine_hartley(ain, aout, axes, fct, nthreads):
    r2r_genuine_hartley(ain, aout, axes, fct, nthreads)
    return aout


@njit
def jit_r2r_fftpack(ain, aout, axes, real2hermitian, forward, fct, nthreads):
    r2r_fftpack(ain, aout, axes, real2hermitian, forward, fct, nthreads)
    return aout


shapes1D = ((10,), (127,))
shapes2D = ((128, 128), (128, 129), (1, 129), (129, 1))
shapes3D = ((32, 17, 39),)
shapes = shapes1D + shapes2D + shapes3D
len1D = range(1, 2048)

ctype = {np.float32: np.complex64, np.float64: np.complex128}
dtypes = [np.float32, np.float64]
tol = {np.float32: 6e-7, np.float64: 1.5e-15}


def _assert_close(a, b, epsilon):
    err = _l2error(a, b)
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_(err < epsilon)


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))



@pytest.mark.parametrize("shp", shapes2D+shapes3D)
def test_genuine_hartley(shp):
    a = np.random.rand(*shp) - 0.5
    aout = np.empty_like(a)
    axes = np.arange(len(shp), dtype=np.int64)
    fct = 1.0
    nthreads = 1

    v1 = jit_r2r_genuine_hartley(a, aout, axes, fct, nthreads)
    v2 = np.fft.fftn(a.astype(np.complex128))
    v2 = v2.real+v2.imag
    assert_(_l2error(v1, v2) < 1e-15)


@pytest.mark.parametrize("shp", shapes)
def test_hartley_identity(shp):
    a = np.random.rand(*shp) - 0.5
    aout = np.empty_like(a)
    axes = np.arange(len(shp), dtype=np.int64)
    fct = 1.0
    nthreads = 1

    v1 = jit_r2r_separable_hartley(
        jit_r2r_separable_hartley(a, aout, axes, fct, nthreads),
        aout, axes, fct, nthreads,
    ) / a.size
    assert_(_l2error(a, v1) < 1e-15)


@pytest.mark.parametrize("shp", shapes)
def test_genuine_hartley_identity(shp):
    a = np.random.rand(*shp) - 0.5
    aout = np.empty_like(a)
    axes = np.arange(len(shp), dtype=np.int64)
    fct = 1.0
    nthreads = 1

    v1 = jit_r2r_genuine_hartley(
        jit_r2r_genuine_hartley(a, aout, axes, fct, nthreads),
        aout, axes, fct, nthreads
    ) / a.size
    assert_(_l2error(a, v1) < 1e-15)

    v1 = a.copy()
    fct = 1 / np.float64(np.prod(shp)**0.5)

    assert_(
        jit_r2r_genuine_hartley(
            jit_r2r_genuine_hartley(v1, v1, axes, fct, nthreads),
            v1, axes, fct, nthreads
        ) is v1)
    assert_(_l2error(a, v1) < 1e-15)


@pytest.mark.parametrize("shp", shapes2D+shapes3D)
@pytest.mark.parametrize("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_genuine_hartley_2D(shp, axes):
    a = np.random.rand(*shp) - 0.5
    aout = np.empty_like(a)
    axes = np.array(axes, dtype=np.int64)
    fct = 1.0
    nthreads = 1

    fct2 = 1.0
    for ax in axes:
        fct2 /= shp[ax]

    aout = jit_r2r_genuine_hartley(
        jit_r2r_genuine_hartley(a, aout, axes, fct, nthreads),
        aout, axes, fct2, nthreads)
    assert_(_l2error(aout, a) < 1e-15)


@pytest.mark.parametrize("len", (3, 4, 5, 6, 7, 8, 9, 10))
@pytest.mark.parametrize("dtype", dtypes)
def testfftpack_extra(len, dtype):
    rng = np.random.default_rng(42)

    a1 = (rng.random(len) - 0.5).astype(dtype)
    out1 = np.empty_like(a1)

    a2 = a1.astype(np.complex128)
    out2 = np.empty_like(a2)

    axes = np.array([0], dtype=np.int64)
    fct = 1.0
    nthreads = 1

    if len != 3:
        return 
    
    eps = tol[dtype]
    test = jit_r2r_fftpack(a1, out1, axes, True, False, fct, nthreads) 
    ref = jit_c2c(a2, out2, axes, False, fct, nthreads) 

    out1 = np.empty_like(a1)
    out2 = np.empty_like(a2)

    test = jit_r2r_fftpack(test, out1, axes, False, True, fct, nthreads)
    ref = jit_c2c(ref, out2, axes, True, fct, nthreads)
    _assert_close(ref, test, eps)
