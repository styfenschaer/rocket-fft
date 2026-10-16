"""
Quick end-to-end check of an installed wheel: loads the native library and
compiles one transform per interface. Runs in CI on every Python version and
without SciPy, so it must not depend on anything beyond the runtime requirements.
"""

import numba as nb
import numpy as np

import rocket_fft


@nb.njit
def numpy_fft(x):
    return np.fft.fft(x)


@nb.njit
def numpy_irfftn(x):
    return np.fft.irfftn(x)


def main():
    from rocket_fft._native import SIMD

    print(f"rocket-fft {rocket_fft.__version__} ({SIMD} build) from {rocket_fft.__file__}")
    print(f"numba {nb.__version__}, numpy {np.__version__}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(30) + 1j * rng.standard_normal(30)
    np.testing.assert_allclose(numpy_fft(x), np.fft.fft(x))

    y = rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
    np.testing.assert_allclose(numpy_irfftn(y), np.fft.irfftn(y))

    try:
        import scipy
        import scipy.fft
    except ImportError:
        print("scipy not installed, skipping scipy.fft check")
    else:
        print(f"scipy {scipy.__version__}")

        @nb.njit
        def scipy_dct(x):
            return scipy.fft.dct(x, type=2, norm="ortho")

        z = rng.standard_normal(17)
        np.testing.assert_allclose(scipy_dct(z), scipy.fft.dct(z, type=2, norm="ortho"))

    print("smoke test passed")


if __name__ == "__main__":
    main()
