from multiprocessing import Pool

import numba as nb
import numpy as np
import scipy.fft
from helpers import numba_cache_cleanup

njit = nb.njit(cache=True, nogil=True)

@njit
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft(x, n, axis, norm, overwrite_x, workers)


@njit
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    return scipy.fft.dct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


@njit
def fht(a, dln, mu, offset=0.0, bias=0.0):
    return scipy.fft.fht(a, dln, mu, offset, bias)
    
    
def func1(a): return np.allclose(scipy.fft.fft(a), fft(a))
def func2(a): return np.allclose(scipy.fft.dct(a), dct(a))
def func3(a): return np.allclose(scipy.fft.fht(a, 1.0, 1.0), fht(a, 1.0, 1.0))


def test_all():
    inputs = [np.random.rand(2**20) for _ in range(42)]
    for func in (func1, func2, func3):
        # Run once to cache functions
        list(map(func, inputs)) 
        
        with Pool(processes=8) as pool:
            ret = pool.map(func, inputs)
            assert all(ret)
          
          
if __name__ == '__main__':  
    test_all()