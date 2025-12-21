"""
Most of these tests are borrowed from Scipy.
Thanks to all who contributed to these tests.
https://github.com/scipy/scipy/blob/main/scipy/fft/tests/test_fftlog.py
Whenever I changed a test, I left a note.
"""

from contextlib import redirect_stdout

import numpy as np
from numba.core.errors import TypingError
import pytest
import scipy.fft
from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    ScipyFFT,
)
from numpy.testing import assert_allclose, assert_raises
from scipy.special import poch

set_numba_capture_errors_new_style()


def test_fht_agrees_with_fftlog():
    # check that fht numerically agrees with the output from Fortran FFTLog,
    # the results were generated with the provided `fftlogtest` program,
    # after fixing how the k array is generated (divide range by n-1, not n)

    # test function, analytical Hankel transform is of the same form
    def f(r, mu):
        return r ** (mu + 1) * np.exp(-(r**2) / 2)

    r = np.logspace(-4, 4, 16)

    dln = np.log(r[1] / r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0

    a = f(r, mu)

    # test 1: compute as given
    ours = ScipyFFT.fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [
        -0.1159922613593045e-02,
        +0.1625822618458832e-02,
        -0.1949518286432330e-02,
        +0.3789220182554077e-02,
        +0.5093959119952945e-03,
        +0.2785387803618774e-01,
        +0.9944952700848897e-01,
        +0.4599202164586588e00,
        +0.3157462160881342e00,
        -0.8201236844404755e-03,
        -0.7834031308271878e-03,
        +0.3931444945110708e-03,
        -0.2697710625194777e-03,
        +0.3568398050238820e-03,
        -0.5554454827797206e-03,
        +0.8286331026468585e-03,
    ]
    assert_allclose(ours, theirs)

    # test 2: change to optimal offset
    offset = ScipyFFT.fhtoffset(dln, mu, bias=bias)
    ours = ScipyFFT.fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [
        +0.4353768523152057e-04,
        -0.9197045663594285e-05,
        +0.3150140927838524e-03,
        +0.9149121960963704e-03,
        +0.5808089753959363e-02,
        +0.2548065256377240e-01,
        +0.1339477692089897e00,
        +0.4821530509479356e00,
        +0.2659899781579785e00,
        -0.1116475278448113e-01,
        +0.1791441617592385e-02,
        -0.4181810476548056e-03,
        +0.1314963536765343e-03,
        -0.5422057743066297e-04,
        +0.3208681804170443e-04,
        -0.2696849476008234e-04,
    ]
    assert_allclose(ours, theirs)

    # test 3: positive bias
    bias = 0.8
    offset = ScipyFFT.fhtoffset(dln, mu, bias=bias)
    ours = ScipyFFT.fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [
        -7.3436673558316850e00,
        +0.1710271207817100e00,
        +0.1065374386206564e00,
        -0.5121739602708132e-01,
        +0.2636649319269470e-01,
        +0.1697209218849693e-01,
        +0.1250215614723183e00,
        +0.4739583261486729e00,
        +0.2841149874912028e00,
        -0.8312764741645729e-02,
        +0.1024233505508988e-02,
        -0.1644902767389120e-03,
        +0.3305775476926270e-04,
        -0.7786993194882709e-05,
        +0.1962258449520547e-05,
        -0.8977895734909250e-06,
    ]
    assert_allclose(ours, theirs)

    # test 4: negative bias
    bias = -0.8
    offset = ScipyFFT.fhtoffset(dln, mu, bias=bias)
    ours = ScipyFFT.fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [
        +0.8985777068568745e-05,
        +0.4074898209936099e-04,
        +0.2123969254700955e-03,
        +0.1009558244834628e-02,
        +0.5131386375222176e-02,
        +0.2461678673516286e-01,
        +0.1235812845384476e00,
        +0.4719570096404403e00,
        +0.2893487490631317e00,
        -0.1686570611318716e-01,
        +0.2231398155172505e-01,
        -0.1480742256379873e-01,
        +0.1692387813500801e00,
        +0.3097490354365797e00,
        +2.7593607182401860e00,
        10.5251075070045800e00,
    ]
    assert_allclose(ours, theirs)


@pytest.mark.parametrize("optimal", [True, False])
@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("bias", [0, 0.1, -0.1])
@pytest.mark.parametrize("n", [64, 63])
def test_fht_identity(n, bias, offset, optimal):
    rng = np.random.RandomState(3491349965)

    a = rng.standard_normal(n)
    dln = rng.uniform(-1, 1)
    mu = rng.uniform(-2, 2)

    if optimal:
        offset = ScipyFFT.fhtoffset(dln, mu, initial=offset, bias=bias)

    A = ScipyFFT.fht(a, dln, mu, offset=offset, bias=bias)
    a_ = ScipyFFT.ifht(A, dln, mu, offset=offset, bias=bias)

    assert_allclose(a, a_)


class Buffer:
    def __init__(self):
        self.written = False

    def write(self, *args, **kwargs):
        self.written = True


def test_fht_special_cases():
    # NOTE: We can't warn about singularity so we check if something is printed

    rng = np.random.RandomState(3491349965)

    a = rng.standard_normal(64)
    dln = rng.uniform(-1, 1)

    # let xp = (mu+1+q)/2, xm = (mu+1-q)/2, M = {0, -1, -2, ...}

    # case 1: xp in M, xm in M => well-defined transform
    mu, bias = -4.0, 1.0
    # with warnings.catch_warnings(record=True) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        ScipyFFT.fht(a, dln, mu, bias=bias)
        assert not buf.written, "fht warned about a well-defined transform"
    # assert not record, "fht warned about a well-defined transform"

    # case 2: xp not in M, xm in M => well-defined transform
    mu, bias = -2.5, 0.5
    # with warnings.catch_warnings(record=True) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        ScipyFFT.fht(a, dln, mu, bias=bias)
        assert not buf.written, "fht warned about a well-defined transform"
    # assert not record, "fht warned about a well-defined transform"

    # case 3: xp in M, xm not in M => singular transform
    mu, bias = -3.5, 0.5
    # with pytest.warns(Warning) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        ScipyFFT.fht(a, dln, mu, bias=bias)
        assert buf.written, "fht did not warn about a singular transform"
    # assert record, "fht did not warn about a singular transform"

    # case 4: xp not in M, xm in M => singular inverse transform
    mu, bias = -2.5, 0.5
    # with pytest.warns(Warning) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        ScipyFFT.ifht(a, dln, mu, bias=bias)
        assert buf.written, "ifht did not warn about a singular transform"
    # assert record, "ifht did not warn about a singular transform"


@pytest.mark.parametrize("n", [64, 63])
def test_fht_exact(n):
    rng = np.random.RandomState(3491349965)

    # for a(r) a power law r^\gamma, the fast Hankel transform produces the
    # exact continuous Hankel transform if biased with q = \gamma

    mu = rng.uniform(0, 3)

    # convergence of HT: -1-mu < gamma < 1/2
    gamma = rng.uniform(-1 - mu, 1 / 2)

    r = np.logspace(-2, 2, n)
    a = r**gamma

    dln = np.log(r[1] / r[0])

    offset = ScipyFFT.fhtoffset(dln, mu, initial=0.0, bias=gamma)

    A = ScipyFFT.fht(a, dln, mu, offset=offset, bias=gamma)

    k = np.exp(offset) / r[::-1]

    # analytical result
    At = (2 / k) ** gamma * poch((mu + 1 - gamma) / 2, gamma)

    assert_allclose(A, At)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.uint8, np.uint16, np.int32, np.int64)
)
def test_compare_with_scipy(dtype):
    a = np.arange(42).astype(dtype)
    dln = dtype(1.0)
    mu = dtype(1.0)
    offset = dtype(1.0)
    bias = dtype(0.0)
    initial = dtype(0.0)

    assert_allclose(
        ScipyFFT.fht(a, dln, mu, offset, bias),
        scipy.fft.fht(a, dln, mu, offset, bias),
        rtol=1e-5,
        atol=1e20,
    )
    assert_allclose(
        ScipyFFT.ifht(a, dln, mu, offset, bias),
        scipy.fft.ifht(a, dln, mu, offset, bias),
        rtol=1e-5,
        atol=1e20,
    )
    assert_allclose(
        ScipyFFT.fhtoffset(dln, mu, initial, bias),
        scipy.fft.fhtoffset(dln, mu, initial, bias),
        rtol=1e-5,
        atol=1e20,
    )

    assert (
        ScipyFFT.fht(a, dln, mu, offset, bias).dtype
        == scipy.fft.fht(a, dln, mu, offset, bias).dtype
    )
    assert (
        ScipyFFT.ifht(a, dln, mu, offset, bias).dtype
        == scipy.fft.ifht(a, dln, mu, offset, bias).dtype
    )
    assert type(ScipyFFT.fhtoffset(dln, mu, initial, bias)) == np.dtype(
        type(scipy.fft.fhtoffset(dln, mu, initial, bias))
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_raises(dtype):
    a = np.arange(42).astype(dtype)

    assert_raises(TypingError, ScipyFFT.fht, a, 1.0, 2.0, 3.0, 4.0)
    assert_raises(TypingError, ScipyFFT.ifht, a, 1.0, 2.0, 3.0, 4.0)
