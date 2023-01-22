"""
The numpy.roll tests are adopted from Numpy.
"""

from functools import partial

import numba as nb
import numpy as np
from helpers import numba_cache_cleanup
from numba import TypingError
from numpy.testing import assert_equal
from pytest import raises as assert_raises

# All functions should be cacheable and run without the GIL
njit = partial(nb.njit, cache=True, nogil=True)


@njit
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
    lut = {0: "1st", 1: "2nd", 2: "3rd", 3: "4th",
           4: "5th", 5: "6th", 6: "7th", 7: "8th"}
    pos = lut.get(pos)
    return fr".* {pos} .* '{name}' .*"


class TestRollTyping:
    x = np.arange(10)
    x2 = np.arange(10).reshape(5, 2)

    def test_a(self):
        with assert_raises(TypingError, match=mk_match(0, "a")):
            roll(list(self.x), shift=1)
        with assert_raises(TypingError, match=mk_match(0, "a")):
            roll(tuple(self.x), shift=1)
        with assert_raises(TypeError):
            roll(x=self.x, shift=1)
        roll(self.x, shift=1)

    def test_shift(self):
        with assert_raises(TypingError, match=mk_match(1, "shift")):
            roll(self.x, shift=None)
        with assert_raises(TypingError, match=mk_match(1, "shift")):
            roll(self.x, shift=1.0)
        with assert_raises(TypingError, match=mk_match(1, "shift")):
            roll(self.x, shift=((1,),))
        with assert_raises(TypeError):
            roll(self.x)
        roll(self.x, shift=1)
        roll(self.x, shift=(1,))
        roll(self.x, shift=np.array([1]))
        roll(self.x2, shift=(1, 0), axis=1)
        roll(self.x2, shift=(1, 0), axis=None)

    def test_axis(self):
        with assert_raises(TypingError, match=mk_match(2, "axis")):
            roll(self.x, shift=1, axis=1.)
        with assert_raises(TypingError, match=mk_match(2, "axis")):
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
