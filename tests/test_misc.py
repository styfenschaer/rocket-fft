"""
The numpy.roll tests are adopted from Numpy.
"""

import numba as nb
import numpy as np
import pytest
from helpers import numba_cache_cleanup, set_numba_capture_errors_new_style
from numba.core.errors import TypingError
from numpy.testing import assert_equal
from pytest import raises as assert_raises

set_numba_capture_errors_new_style()


@nb.njit(cache=True, nogil=True)
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


class TestCompareNumpy:
    @staticmethod
    def C(arr):
        return np.ascontiguousarray(arr)

    @staticmethod
    def F(arr):
        return np.asfortranarray(arr)

    @staticmethod
    def A(arr, mode):
        if arr.ndim < 2:
            arr = np.random.rand(arr.size * 2)
            return arr[::2]
        shape = arr.shape
        if mode == 1:
            arr = np.random.rand(*shape[:-1], 2 * shape[-1])
            return arr[..., ::2]
        if mode == 2:
            arr = np.random.rand(2 * shape[0], *shape[1:])
            return arr[::2, ...]
        if mode == 3:
            arr = np.random.rand(2 * shape[0], *shape[1:-1], 2 * shape[-1])
            return arr[::2, ..., ::2]
        raise ValueError(f"Invalid mode {mode}")

    def A1(self, arr):
        return self.A(arr, mode=1)

    def A2(self, arr):
        return self.A(arr, mode=2)

    def A3(self, arr):
        return self.A(arr, mode=3)

    @pytest.mark.parametrize(
        "shape, shift, axis",
        [
            *zip(
                [
                    (0,),
                    (1,),
                    (83,),
                    (83,),
                    (3, 42, 42),
                    (99, 50, 25),
                    (100, 3, 19, 19),
                ],  # shape
                [1, -1, 21, -21, (-12, 85, 1), (5, 2), (3, 3)],  # shift
                [0, 0, (0,), None, (-3, 1, 0), (0, 0), (2, 3)],  # axis
            )
        ],
    )
    def test_all(self, shape, shift, axis):
        for layout in (self.C, self.F, self.A1, self.A2, self.A3):
            a = layout(np.random.rand(*shape))

            got = roll(a, shift, axis)
            expected = np.roll(a, shift, axis)

            assert np.allclose(got, expected)
            assert got.dtype == expected.dtype
            assert got.flags.c_contiguous == expected.flags.c_contiguous
            assert got.flags.f_contiguous == expected.flags.f_contiguous


def mk_match(pos, name):
    lut = {
        0: "1st",
        1: "2nd",
        2: "3rd",
        3: "4th",
        4: "5th",
        5: "6th",
        6: "7th",
        7: "8th",
    }
    pos = lut.get(pos)
    return rf".* {pos} .* '{name}' .*"


class TestRollTyping:
    x = np.arange(10)
    x2 = np.arange(10).reshape(5, 2)

    def test_a(self):
        with assert_raises(TypeError):
            roll(x=self.x, shift=1)
        with assert_raises(TypingError):
            roll("abc", shift=1)
        roll(self.x, shift=1)
        roll(nb.typed.List(self.x), shift=1)
        roll(tuple(self.x), shift=1)
        roll(42.0, shift=1)
        roll(True, shift=1)
        roll(42, shift=1)

    def test_shift(self):
        with assert_raises(TypingError, match=mk_match(1, "shift")):
            roll(self.x, shift=None)
        with assert_raises(TypingError, match=mk_match(1, "shift")):
            roll(self.x, shift=1.0)
        with assert_raises(TypingError, match=mk_match(1, "shift")):
            roll(self.x, shift=((1,),))
        with assert_raises(AttributeError):  # because shift is 2d
            roll(self.x, shift=np.array([[1]]), axis=np.array([0]))
        with assert_raises(TypeError):
            roll(self.x)
        roll(self.x, shift=1)
        roll(self.x, shift=True)
        roll(self.x, shift=(1,))
        roll(self.x, shift=np.array([1]))
        roll(self.x2, shift=(1, 0), axis=1)
        roll(self.x2, shift=(1, 0), axis=None)

    def test_axis(self):
        with assert_raises(TypingError, match=mk_match(2, "axis")):
            roll(self.x, shift=1, axis=1.0)
        with assert_raises(TypingError, match=mk_match(2, "axis")):
            roll(self.x, shift=1, axis=((1,),))
        with assert_raises(AttributeError):  # because shift is 2d
            roll(self.x, shift=np.array([1]), axis=np.array([[0]]))
        roll(self.x, shift=1, axis=(0,))
        roll(self.x, shift=1, axis=0)
        roll(self.x, shift=True, axis=False)
        roll(self.x2, shift=1, axis=(1,))
        roll(self.x2, shift=1, axis=(1, 0))

    def test_shift_and_axis(self):
        roll(self.x, shift=1, axis=None)
        roll(self.x, shift=1, axis=0)
        roll(self.x2, shift=True, axis=True)
        roll(self.x2, shift=1, axis=(0,))
        roll(self.x, shift=(1,), axis=None)
        roll(self.x, shift=np.array([1], dtype=np.int64), axis=0)
        roll(self.x2, shift=np.array([1]), axis=np.array([0, 1]))
