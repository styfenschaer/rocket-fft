import numba as nb
import numpy as np
import pytest
from numba import types
from pytest import raises as assert_raises

from rocket_fft import scipy_like
from rocket_fft.unsafe import (disable_typing_check, enable_typing_check,
                               get_builder, get_builders, get_mapping_table,
                               get_overloaded_functions, get_siblings, maps_to,
                               update_mapping_table)

scipy_like()


def fft2(a, s=None, axes=None, norm=None):
    return np.fft.fft2(a, s, axes, norm)


def fft(a, n=None, axis=-1, norm=None):
    return np.fft.fft(a, n, axis, norm)


@pytest.fixture(autouse=True)
def backup_mapping_table():
    lut_real = get_mapping_table(real=True)
    lut_cmplx = get_mapping_table(real=False)
    lut_real_bak = lut_real.copy()
    lut_cmplx_bak = lut_cmplx.copy()
    yield
    lut_real.update(lut_real_bak)
    lut_cmplx.update(lut_cmplx_bak)


def test_unsafe_mapping():
    with assert_raises(TypeError):
        lut = get_mapping_table()
    lut1 = get_mapping_table(real=True)
    assert isinstance(lut1, dict)
    lut2 = get_mapping_table(real=False)
    assert isinstance(lut2, dict)
    assert lut1 is not lut2

    with assert_raises(TypeError):
        d = maps_to(types.complex128)
    d = maps_to(types.complex128, real=True)
    assert d == types.float64
    d = maps_to(types.complex128, real=False)
    assert d == types.complex128
    d = maps_to(types.byte, real=True)
    assert d == types.float64
    d = maps_to(types.float64, real=True)
    assert d == types.float64
    d = maps_to(types.byte, real=False)
    assert d == types.complex128
    d = maps_to(types.float32, real=False)
    assert d == types.complex64

    with assert_raises(TypeError):
        update_mapping_table(types.complex64, types.float32, real=False)
    with assert_raises(TypeError):
        update_mapping_table(types.complex64, np.float32)
    update_mapping_table(types.complex64, types.float32, 1)

    with assert_raises(TypeError):
        update_mapping_table(types.float32, types.complex64, real=True)
    update_mapping_table(types.complex64, types.float64, real=True)
    d = maps_to(types.complex64, real=True)
    assert d == types.float64
    update_mapping_table(types.complex64, types.float32)
    d = maps_to(types.complex64, real=True)
    assert d == types.float32

    with assert_raises(TypeError):
        update_mapping_table(types.complex64, types.float64, real=False)
    update_mapping_table(types.float64, types.complex64, real=False)
    d = maps_to(types.float64, real=False)
    assert d == types.complex64
    update_mapping_table(types.float32, types.complex64)
    d = maps_to(types.float32, real=False)
    assert d == types.complex64


def test_unsafe_typing():
    x = np.random.rand(1, 1)

    with assert_raises(nb.TypingError):
        nb.njit(fft)(x, norm=False)

    with assert_raises(ValueError):
        disable_typing_check()
        nb.njit(fft)(x, norm=False)

    with assert_raises(nb.TypingError):
        enable_typing_check()
        nb.njit(fft2)(x, norm=False)

    with assert_raises(ValueError):
        disable_typing_check()
        nb.njit(fft2)(x, norm=False)

    overloaded_func = np.fft.fft
    builder = get_builder(overloaded_func)
    assert builder.register[overloaded_func][0] is builder

    with assert_raises(ValueError):
        builder = get_builder(range)

    with assert_raises(TypeError):
        builder = get_builders(unique=-1)

    builders = get_builders()
    assert builder in builders
    assert len(get_builders(False)) > len(get_builders(True))

    assert np.fft.fft in get_overloaded_functions(builder)
    with assert_raises(TypeError):
        get_overloaded_functions(None)

    assert len(get_siblings(np.fft.fft)) == 5
    assert np.fft.ifft in get_siblings(np.fft.fft)


if __name__ == "__main__":
    pytest.main([__file__])
