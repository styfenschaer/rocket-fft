import numba as nb
import numpy as np
import pytest
from numba.np.numpy_support import as_dtype

from rocket_fft import numpy_like
from rocket_fft.overloads import _numpy_cmplx_lut

numpy_like()


@nb.njit
def fft(a, n=None, axis=-1, norm=None):
    return np.fft.fft(a, n, axis, norm)


def test_numpy_like():
    x = np.random.rand(42)

    for ty in _numpy_cmplx_lut.keys():
        ty = as_dtype(ty)
        
        dty1 = np.fft.fft(x.astype(ty)).dtype
        dty2 = fft(x.astype(ty)).dtype
        assert dty1 == dty2


if __name__ == "__main__":
    pytest.main([__file__])