import os
import subprocess
import sys

import numpy as np

test_fft = """
from time import perf_counter
import numpy as np
import numba as nb
import scipy.fft
nb.njit(lambda: None)()

@nb.njit
def func(a):
    return scipy.fft.fft(a)

a = np.ones(1, dtype=np.complex128)

tic = perf_counter()
func(a)
toc = perf_counter()
elapsed = toc - tic

print(elapsed)
"""

test_fht = """
from time import perf_counter
import numpy as np
import numba as nb
import scipy.fft
nb.njit(lambda: None)()

@nb.njit
def func(a):
    return scipy.fft.fht(a, 1.0, 1.0)

a = np.ones(1)

tic = perf_counter()
func(a)
toc = perf_counter()
elapsed = toc - tic

print(elapsed)
"""

test_dct = """
from time import perf_counter
import numpy as np
import numba as nb
import scipy.fft
nb.njit(lambda: None)()

@nb.njit
def func(a):
    return scipy.fft.dct(a)

a = np.ones(2)

tic = perf_counter()
func(a)
toc = perf_counter()
elapsed = toc - tic

print(elapsed)
"""


def test(src, n_iter):
    try:
        filename = "profile_compile.py"
        with open(filename, "w") as file:
            file.write(src)

        timings = []
        for i in range(n_iter):
            elapsed = subprocess.check_output([sys.executable, filename])
            elapsed = float(elapsed)
            print(f"Iteration {i+1}/{n_iter}: {elapsed:.3f}")
            timings.append(elapsed)

        print(f"   Min: {np.amin(timings):.3f}")
        print(f"  Mean: {np.median(timings):.3f}")
        print(f"Median: {np.mean(timings):.3f}")
        print(f"   Max: {np.amax(timings):.3f}")

    except Exception as e:
        raise e
    finally:
        os.unlink(filename)


def main(n_iter):
    for src in (test_fft, test_fht, test_dct):
        test(src, n_iter)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
