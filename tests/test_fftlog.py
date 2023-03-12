"""
Most of these tests are borrowed from Scipy. 
Thanks to all who contributed to these tests.
https://github.com/scipy/scipy/blob/main/scipy/fft/tests/test_fftlog.py
Whenever I changed a test, I left a note.
"""

from contextlib import redirect_stdout
from functools import partial

import numba as nb
import numpy as np
import pytest
import scipy.fft
from helpers import numba_cache_cleanup
from numpy.testing import assert_allclose, assert_raises
from scipy.special import poch

# All functions should be cacheable and run without the GIL
njit = partial(nb.njit, cache=True, nogil=True)


@njit
def fht(a, dln, mu, offset=0.0, bias=0.0):
    return scipy.fft.fht(a, dln, mu, offset, bias)


@njit
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    return scipy.fft.ifht(A, dln, mu, offset, bias)


@njit
def fhtoffset(dln, mu, initial=0.0, bias=0.0):
    return scipy.fft.fhtoffset(dln, mu, initial, bias)


def test_fht_agrees_with_fftlog():
    # check that fht numerically agrees with the output from Fortran FFTLog,
    # the results were generated with the provided `fftlogtest` program,
    # after fixing how the k array is generated (divide range by n-1, not n)

    # test function, analytical Hankel transform is of the same form
    def f(r, mu):
        return r**(mu+1)*np.exp(-r**2/2)

    r = np.logspace(-4, 4, 16)

    dln = np.log(r[1]/r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0

    a = f(r, mu)

    # test 1: compute as given
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [-0.1159922613593045E-02, +0.1625822618458832E-02,
              -0.1949518286432330E-02, +0.3789220182554077E-02,
              +0.5093959119952945E-03, +0.2785387803618774E-01,
              +0.9944952700848897E-01, +0.4599202164586588E+00,
              +0.3157462160881342E+00, -0.8201236844404755E-03,
              -0.7834031308271878E-03, +0.3931444945110708E-03,
              -0.2697710625194777E-03, +0.3568398050238820E-03,
              -0.5554454827797206E-03, +0.8286331026468585E-03]
    assert_allclose(ours, theirs)

    # test 2: change to optimal offset
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [+0.4353768523152057E-04, -0.9197045663594285E-05,
              +0.3150140927838524E-03, +0.9149121960963704E-03,
              +0.5808089753959363E-02, +0.2548065256377240E-01,
              +0.1339477692089897E+00, +0.4821530509479356E+00,
              +0.2659899781579785E+00, -0.1116475278448113E-01,
              +0.1791441617592385E-02, -0.4181810476548056E-03,
              +0.1314963536765343E-03, -0.5422057743066297E-04,
              +0.3208681804170443E-04, -0.2696849476008234E-04]
    assert_allclose(ours, theirs)

    # test 3: positive bias
    bias = 0.8
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [-7.3436673558316850E+00, +0.1710271207817100E+00,
              +0.1065374386206564E+00, -0.5121739602708132E-01,
              +0.2636649319269470E-01, +0.1697209218849693E-01,
              +0.1250215614723183E+00, +0.4739583261486729E+00,
              +0.2841149874912028E+00, -0.8312764741645729E-02,
              +0.1024233505508988E-02, -0.1644902767389120E-03,
              +0.3305775476926270E-04, -0.7786993194882709E-05,
              +0.1962258449520547E-05, -0.8977895734909250E-06]
    assert_allclose(ours, theirs)

    # test 4: negative bias
    bias = -0.8
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [+0.8985777068568745E-05, +0.4074898209936099E-04,
              +0.2123969254700955E-03, +0.1009558244834628E-02,
              +0.5131386375222176E-02, +0.2461678673516286E-01,
              +0.1235812845384476E+00, +0.4719570096404403E+00,
              +0.2893487490631317E+00, -0.1686570611318716E-01,
              +0.2231398155172505E-01, -0.1480742256379873E-01,
              +0.1692387813500801E+00, +0.3097490354365797E+00,
              +2.7593607182401860E+00, 10.5251075070045800E+00]
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
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)

    A = fht(a, dln, mu, offset=offset, bias=bias)
    a_ = ifht(A, dln, mu, offset=offset, bias=bias)

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
        fht(a, dln, mu, bias=bias)
        assert not buf.written, "fht warned about a well-defined transform"
    # assert not record, "fht warned about a well-defined transform"

    # case 2: xp not in M, xm in M => well-defined transform
    mu, bias = -2.5, 0.5
    # with warnings.catch_warnings(record=True) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        fht(a, dln, mu, bias=bias)
        assert not buf.written, "fht warned about a well-defined transform"
    # assert not record, "fht warned about a well-defined transform"

    # case 3: xp in M, xm not in M => singular transform
    mu, bias = -3.5, 0.5
    # with pytest.warns(Warning) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        fht(a, dln, mu, bias=bias)
        assert buf.written, "fht did not warn about a singular transform"
    # assert record, "fht did not warn about a singular transform"

    # case 4: xp not in M, xm in M => singular inverse transform
    mu, bias = -2.5, 0.5
    # with pytest.warns(Warning) as record:
    buf = Buffer()
    with redirect_stdout(buf):
        ifht(a, dln, mu, bias=bias)
        assert buf.written, "ifht did not warn about a singular transform"
    # assert record, "ifht did not warn about a singular transform"


@pytest.mark.parametrize("n", [64, 63])
def test_fht_exact(n):
    rng = np.random.RandomState(3491349965)

    # for a(r) a power law r^\gamma, the fast Hankel transform produces the
    # exact continuous Hankel transform if biased with q = \gamma

    mu = rng.uniform(0, 3)

    # convergence of HT: -1-mu < gamma < 1/2
    gamma = rng.uniform(-1-mu, 1/2)

    r = np.logspace(-2, 2, n)
    a = r**gamma

    dln = np.log(r[1]/r[0])

    offset = fhtoffset(dln, mu, initial=0.0, bias=gamma)

    A = fht(a, dln, mu, offset=offset, bias=gamma)

    k = np.exp(offset)/r[::-1]

    # analytical result
    At = (2/k)**gamma * poch((mu+1-gamma)/2, gamma)

    assert_allclose(A, At)


@pytest.mark.parametrize("dtype", (np.float32, np.float64,
                                   np.uint8, np.uint16,
                                   np.int32, np.int64))
def test_compare_with_scipy(dtype):
    a = np.arange(42).astype(dtype)
    dln = dtype(1.0)
    mu = dtype(1.0)
    offset = dtype(1.0)
    bias = dtype(0.0)
    initial = dtype(0.0)
    
    assert_allclose(fht(a, dln, mu, offset, bias),
                    scipy.fft.fht(a, dln, mu, offset, bias),
                    rtol=1e-5, atol=1e20)
    assert_allclose(ifht(a, dln, mu, offset, bias),
                    scipy.fft.ifht(a, dln, mu, offset, bias),
                    rtol=1e-5, atol=1e20)
    assert_allclose(fhtoffset(dln, mu, initial, bias),
                    scipy.fft.fhtoffset(dln, mu, initial, bias),
                    rtol=1e-5, atol=1e20)

    assert (fht(a, dln, mu, offset, bias).dtype == 
            scipy.fft.fht(a, dln, mu, offset, bias).dtype)
    assert (ifht(a, dln, mu, offset, bias).dtype == 
            scipy.fft.ifht(a, dln, mu, offset, bias).dtype)
    assert (type(fhtoffset(dln, mu, initial, bias)) == 
            np.dtype(type(scipy.fft.fhtoffset(dln, mu, initial, bias))))


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_raises(dtype):
    a = np.arange(42).astype(dtype)

    assert_raises(nb.TypingError, fht, a, 1.0, 2.0, 3.0, 4.0)
    assert_raises(nb.TypingError, ifht, a, 1.0, 2.0, 3.0, 4.0)