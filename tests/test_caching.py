import os
import subprocess

src = """
import numpy as np 
import numba as nb 
import scipy.fft

njit = nb.njit(cache=True, nogil=True)

fft = njit(lambda a: scipy.fft.fft(a))
dct = njit(lambda a: scipy.fft.dct(a))
fht = njit(lambda a: scipy.fft.fht(a, 1.0, 1.0))
    
a = np.ones(42) * 1j
    
fft(a)
dct(a.real)
fht(a.real)
"""


def test_caching():
    try:
        filename = "tmp_test_caching.py"
        with open(filename, "w") as file:
            file.write(src)
            
        subprocess.check_output(['python', filename])
        subprocess.check_output(['python', filename])
            
    except Exception as e:
        raise e
    finally:
        os.unlink(filename)
