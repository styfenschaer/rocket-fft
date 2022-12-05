"""
The numpy.roll tests are adopted from Numpy.
"""

from functools import partial

import numba as nb
import numpy as np
import pytest
from helpers import numba_cache_cleanup
from numba import types
from numpy.testing import assert_equal
from pytest import raises as assert_raises

from rocket_fft.unsafe import (get_mapping_table, is_mapped_to,
                               is_mapped_to_cmplx, is_mapped_to_real,
                               update_dtype_mapping,
                               update_dtype_mapping_cmplx,
                               update_dtype_mapping_real)

# All functions should be cacheable and run without the GIL
nb.njit = partial(nb.njit, cache=True, nogil=True)


@nb.njit
def roll(a, shift, axis=None):
    return np.roll(a, shift, axis)


class TestRoll:
    def test_roll1d(self):
        x = np.arange(10)
        xr = roll(x, 2)
        assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))

    def test_roll2d(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r = roll(x2, 1)
        assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))

        x2r = roll(x2, 1, axis=0)
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = roll(x2, 1, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # Roll multiple axes at once.
        x2r = roll(x2, 1, axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = roll(x2, (1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = roll(x2, (-1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = roll(x2, (0, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = roll(x2, (0, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]))

        x2r = roll(x2, (1, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = roll(x2, (-1, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[6, 7, 8, 9, 5], [1, 2, 3, 4, 0]]))

        # Roll the same axis multiple times.
        x2r = roll(x2, 1, axis=(0, 0))
        assert_equal(x2r, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))

        x2r = roll(x2, 1, axis=(1, 1))
        assert_equal(x2r, np.array([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]))

        # Roll more than one turn in either direction.
        x2r = roll(x2, 6, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = roll(x2, -4, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

    def test_roll_empty(self):
        x = np.array([])
        assert_equal(roll(x, 1), np.array([]))


def mk_match(pos, name):
    lut = {0: '1st', 1: '2nd', 2: '3rd', 3: '4th',
           4: '5th', 5: '6th', 6: '7th', 7: '8th'}
    pos = lut.get(pos)
    return fr".* {pos} .* '{name}' .*"


class TestRollTyping:
    x = np.arange(10)
    x2 = np.arange(10).reshape(5, 2)

    def test_a(self):
        with assert_raises(nb.TypingError, match=mk_match(0, 'a')):
            roll(list(self.x), shift=1)
        with assert_raises(nb.TypingError, match=mk_match(0, 'a')):
            roll(tuple(self.x), shift=1)
        with assert_raises(TypeError):
            roll(x=self.x, shift=1)
        roll(self.x, shift=1)

    def test_shift(self):
        with assert_raises(nb.TypingError, match=mk_match(1, 'shift')):
            roll(self.x, shift=None)
        with assert_raises(nb.TypingError, match=mk_match(1, 'shift')):
            roll(self.x, shift=1.0)
        with assert_raises(nb.TypingError, match=mk_match(1, 'shift')):
            roll(self.x, shift=((1,),))
        with assert_raises(TypeError):
            roll(self.x)
        roll(self.x, shift=1)
        roll(self.x, shift=(1,))
        roll(self.x, shift=np.array([1]))
        roll(self.x2, shift=(1, 0), axis=1)
        roll(self.x2, shift=(1, 0), axis=None)

    def test_axis(self):
        with assert_raises(nb.TypingError, match=mk_match(2, 'axis')):
            roll(self.x, shift=1, axis=1.)
        with assert_raises(nb.TypingError, match=mk_match(2, 'axis')):
            roll(self.x, shift=1, axis=((1,),))
        roll(self.x, shift=1, axis=(0,))
        roll(self.x, shift=1, axis=0)
        roll(self.x2, shift=1, axis=(1,))
        roll(self.x2, shift=1, axis=(1, 0))

    def test_shift_and_axis(self):
        roll(self.x, shift=1, axis=None)
        roll(self.x, shift=1, axis=0)
        roll(self.x2, shift=1, axis=(0,))
        roll(self.x, shift=(1,), axis=None)
        roll(self.x, shift=np.array([1], dtype=np.int64), axis=0)
        roll(self.x2, shift=np.array([1]), axis=np.array([0, 1]))


@pytest.fixture(autouse=True)
def backup_mapping_table():
    lut_real = get_mapping_table(real=True)
    lut_cmplx = get_mapping_table(real=False)
    lut_real_bak = lut_real.copy()
    lut_cmplx_bak = lut_cmplx.copy()
    yield
    lut_real = get_mapping_table(real=True)
    lut_cmplx = get_mapping_table(real=False)
    lut_real.update(lut_real_bak)
    lut_cmplx.update(lut_cmplx_bak)


def test_unsafe_features():
    with assert_raises(TypeError):
        lut = get_mapping_table()
    lut1 = get_mapping_table(real=True)
    assert isinstance(lut1, dict)
    lut2 = get_mapping_table(real=False)
    assert isinstance(lut2, dict)
    assert lut1 is not lut2

    with assert_raises(TypeError):
        d = is_mapped_to(types.complex128)
    d = is_mapped_to(types.complex128, real=True)
    assert d == types.float64
    d = is_mapped_to(types.complex128, real=False)
    assert d == types.complex128
    d = is_mapped_to(types.byte, real=True)
    assert d == types.float32
    d = is_mapped_to(types.float64, real=True)
    assert d == types.float64
    d = is_mapped_to(types.byte, real=False)
    assert d == types.complex64

    d = is_mapped_to_real(types.complex128)
    assert d == types.float64
    d = is_mapped_to_real(types.byte)
    assert d == types.float32
    d = is_mapped_to_real(types.float64)
    assert d == types.float64

    d = is_mapped_to_cmplx(types.complex128)
    assert d == types.complex128
    d = is_mapped_to_cmplx(types.float32)
    assert d == types.complex64
    d = is_mapped_to_cmplx(types.byte)
    assert d == types.complex64

    with assert_raises(TypeError):
        lut = update_dtype_mapping(types.complex64, types.float32)
    with assert_raises(TypeError):
        lut = update_dtype_mapping(types.complex64, np.float32)
    with assert_raises(TypeError):
        lut = update_dtype_mapping(types.complex64, types.float32, 1)

    with assert_raises(TypeError):
        lut = update_dtype_mapping(types.float32, types.complex64, real=True)
    update_dtype_mapping(types.complex64, types.float64, real=True)
    d = is_mapped_to_real(types.complex64)
    assert d == types.float64

    with assert_raises(TypeError):
        lut = update_dtype_mapping(types.complex64, types.float64, real=False)
    update_dtype_mapping(types.float64, types.complex64, real=False)
    d = is_mapped_to_cmplx(types.float64)
    assert d == types.complex64

    with assert_raises(TypeError):
        lut = update_dtype_mapping_real(types.float32, types.complex64)
    update_dtype_mapping_real(types.complex64, types.float64)
    d = is_mapped_to_real(types.complex64)
    assert d == types.float64

    with assert_raises(TypeError):
        lut = update_dtype_mapping_cmplx(types.complex64, types.float64)
    update_dtype_mapping_cmplx(types.float64, types.complex64)
    d = is_mapped_to_cmplx(types.float64)
    assert d == types.complex64
