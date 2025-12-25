# Rocket-FFT
[![PyPI version](https://img.shields.io/pypi/v/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![License](https://img.shields.io/pypi/l/rocket-fft?color=%2376519B)](https://opensource.org/licenses/BSD-3-Clause)
[![python](https://img.shields.io/pypi/pyversions/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![status](https://img.shields.io/pypi/status/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![downloads](https://img.shields.io/pypi/dm/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)

![](https://raw.githubusercontent.com/styfenschaer/rocket-fft/release0.3.0/assets/fourier.gif)

Rocket-FFT makes [Numba](https://numba.pydata.org/) aware of `numpy.fft` and `scipy.fft`. It takes its name from the [PocketFFT](https://github.com/mreineck/pocketfft) Fast Fourier Transformation library that powers it, and Numba's goal of making your scientific Python code blazingly fast - like a rocket. 

## Getting Started
The easiest way to get Rocket-FFT is to:
```
$ pip install rocket-fft
```
Alternatively, you can build it from source:
```
$ git clone https://github.com/styfenschaer/rocket-fft.git
$ cd rocket-fft
$ python setup.py install
``` 
The latter requires a C++ compiler compatible with your Python installation.

Once installed successfully, the following will work (no import required):
```python
import numba as nb
import numpy as np

@nb.njit
def jit_fft(x):
    return np.fft.fft(x)

a = np.array([1, 6, 1, 8, 0, 3, 3, 9])
jit_fft(a)
```

## Performance Tip
Rocket-FFT makes extensive use of Numba's polymorphic dispatching to achieve both flexible function signatures similar to SciPy and NumPy, and low compilation times. Compilation takes only a few hundred milliseconds in most cases. Calls with default arguments follow a fast path and compile fastest.

## NumPy-like and SciPy-like interfaces
NumPy and SciPy show a subtle difference in how they handle the `axes` argument in some functions<sup>1</sup>. Rocket-FFT implements both ways and lets its users choose between them.

You can set the interface by using the `scipy_like` or `numpy_like` function from the `rocket_fft` namespace:
```python
from rocket_fft import numpy_like, scipy_like

numpy_like()
```
Both functions can be used regardless of whether SciPy is installed<sup>2</sup>. By default, Rocket-FFT uses the SciPy-like interface if SciPy is installed, and the NumPy-like interface otherwise. Note that the interface cannot be changed after the compilation of Rocket-FFT's internals.

<sup>1</sup>NumPy allows duplicate axes in `fft2`, `ifft2`, `fftn` and `ifftn`, whereas SciPy doesn't
<br/>
<sup>2</sup>SciPy is an optional runtime dependency

## Known limitations
- Rocket-FFT implements NumPy's FFT interface and behavior as introduced in NumPy 2.0.
  Note, however, that the `out` argument is not currently supported. Attempting to
  specify it will result in a compile-time error.

- There is a known issue in the implementations of `scipy.fft.dst`, `scipy.fft.idst`,
  `scipy.fft.dstn`, and `scipy.fft.idstn` that may produce incorrect results when
  `norm` and/or `orthogonalize` are set to non-default values. Specifying these
  arguments with values other than their defaults emits a warning, and it is the
  user's responsibility to verify the results. Using the default values is safe.


## Low-Level Interface
Rocket-FFT also provides a low-level interface to the PocketFFT library. Using the low-level interface can significantly reduce compile time, minimize overhead and give more flexibility to the user. It also provides some functions that are not available through the SciPy-like and NumPy-like interfaces. You can import its functions from the `rocket_fft` namespace:
```python
from rocket_fft import c2c, dct, ...
```
The low-level interface includes the following functions:
```python
def c2c(ain: NDArray[c8] | NDArray[c16], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: b1, fct: f4 | f8, nthreads: i8) -> None: ...
def r2c(ain: NDArray[f4] | NDArray[f8], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: b1, fct: f4 | f8, nthreads: i8) -> None: ...
def c2r(ain: NDArray[c8] | NDArray[c16], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], forward: b1, fct: f4 | f8, nthreads: i8) -> None: ...
def c2c_sym(ain: NDArray[f4] | NDArray[f8], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: b1, fct: f4 | f8, nthreads: i8) -> None: ...
def dst(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], type: i8, fct: f4 | f8, ortho: b1, nthreads: i8) -> None: ...
def dct(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], type: i8, fct: f4 | f8, ortho: b1, nthreads: i8) -> None: ...
def r2r_separable_hartley(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], fct: f4 | f8, nthreads: i8) -> None: ...
def r2r_genuine_hartley(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], fct: f4 | f8, nthreads: i8) -> None: ...
def r2r_fftpack(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], real2hermitian: b1, forward: b1, fct: f4 | f8, nthreads: i8) -> None: ...
def good_size(target: i8, real: b1) -> i8: ...
```
Note that the low-level interface provides a lower level of safety and convenience compared to the SciPy-like and NumPy-like interfaces. 
There is almost no safety net, and it is up to the user to ensure proper usage. You may want to consult the original [PocketFFT](https://github.com/mreineck/pocketfft) C++ implementation before using it.