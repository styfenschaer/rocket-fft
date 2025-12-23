import os
import subprocess
import sys

from helpers import numba_cache_cleanup, set_numba_capture_errors_new_style

set_numba_capture_errors_new_style()

src = """
import numpy as np
import numba as nb
import scipy.fft
njit = nb.njit(cache=True, nogil=True)
fft = njit(lambda a: scipy.fft.fft(a))
dct = njit(lambda a: scipy.fft.dct(a))
fht = njit(lambda a: scipy.fft.fht(a, 1.0, 1.0))

a = np.ones(42)

fft(a)
dct(a)
fht(a)
"""


def test_caching():
    try:
        filename = "tmp_test_caching.py"
        with open(filename, "w") as file:
            file.write(src)

        subprocess.call([sys.executable, filename])
        subprocess.call([sys.executable, filename])
    except Exception as e:
        raise e
    finally:
        os.unlink(filename)
