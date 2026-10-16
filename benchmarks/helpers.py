"""Runtime of the frequency and shift helpers (@njit) against NumPy and SciPy.

    python benchmarks/helpers.py
"""

import numba as nb
import numpy as np
import scipy.fft
from _common import print_table, time_callable

import rocket_fft

SIZES = [16, 256, 4096, 65536, 1048576]
SHIFT_CASES = [
    ("1D N=1,024", (1024,), None),
    ("1D N=1,048,576", (1048576,), None),
    ("2D 256x256", (256, 256), None),
    ("2D 1024x1024", (1024, 1024), None),
    ("2D 256x256 axes=0", (256, 256), 0),
    ("3D 64x64x64", (64, 64, 64), None),
    ("3D 64x64x64 axes=(0,2)", (64, 64, 64), (0, 2)),
]


@nb.njit
def nb_fftfreq(n, d):
    return np.fft.fftfreq(n, d)


@nb.njit
def nb_rfftfreq(n, d):
    return np.fft.rfftfreq(n, d)


@nb.njit
def nb_fftshift(a, axes):
    return np.fft.fftshift(a, axes)


@nb.njit
def nb_ifftshift(a, axes):
    return np.fft.ifftshift(a, axes)


def bench_freq(name, jitted):
    rows = []
    for n in SIZES:
        rows.append({
            "test": f"{name} (N={n:,})",
            "Rocket-FFT": time_callable(jitted, n, 0.1),
            "NumPy": time_callable(getattr(np.fft, name), n, 0.1),
            "SciPy": time_callable(getattr(scipy.fft, name), n, 0.1),
        })
    print_table(name, rows, references=("NumPy", "SciPy"))


def bench_shift(name, jitted):
    rows = []
    for desc, shape, axes in SHIFT_CASES:
        a = np.ones(shape)
        rows.append({
            "test": f"{name} {desc}",
            "Rocket-FFT": time_callable(jitted, a, axes),
            "NumPy": time_callable(getattr(np.fft, name), a, axes),
            "SciPy": time_callable(getattr(scipy.fft, name), a, axes),
        })
    print_table(name, rows, references=("NumPy", "SciPy"))


def main():
    print(f"rocket-fft {rocket_fft.__version__}, numba {nb.__version__}, "
          f"numpy {np.__version__}, scipy {scipy.__version__}")
    bench_freq("fftfreq", nb_fftfreq)
    bench_freq("rfftfreq", nb_rfftfreq)
    bench_shift("fftshift", nb_fftshift)
    bench_shift("ifftshift", nb_ifftshift)


if __name__ == "__main__":
    main()
