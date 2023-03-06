import os
import subprocess
import sys

import numpy as np

src = """
from time import perf_counter
import numpy as np 
import numba as nb 
import scipy.fft
nb.njit(lambda: None)()

@nb.njit
def func(a):
    return scipy.fft.fft(a)
    
a = np.ones(1, dtype=np.complex64)
    
tic = perf_counter()
func(a)
toc = perf_counter() 
elapsed = toc - tic

print(elapsed)
"""


# src = """
# from time import perf_counter
# import numpy as np 
# import numba as nb 
# from rocket_fft import c2c
# nb.njit(lambda: None)()

# @nb.njit
# def func(ain, aout, axes):
#     return c2c(ain, aout, axes, True, 1.0, 1)
    
# ain = np.ones(1, dtype=np.float64)
# aout = np.empty_like(ain)
# axes = np.array([0], dtype=np.int64)
   
# tic = perf_counter()
# func(ain, aout, axes)
# toc = perf_counter() 
# elapsed = toc - tic

# print(elapsed)
# """


def main(n_iter):
    try:
        filename = "profile_compile.py"
        with open(filename, "w") as file:
            file.write(src)
            
        timings = []
        for i in range(n_iter):
            elapsed = subprocess.check_output(['python', filename])
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
        

if __name__ == "__main__":
    n_iter_default = 5
    main(int(sys.argv[1]) if len(sys.argv) > 1 else n_iter_default)