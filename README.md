# Rocket-FFT
[![PyPI version](https://img.shields.io/pypi/v/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![License](https://img.shields.io/pypi/l/rocket-fft?color=%2376519B)](https://opensource.org/licenses/BSD-3-Clause)
[![python](https://img.shields.io/pypi/pyversions/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![status](https://img.shields.io/pypi/status/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![downloads](https://img.shields.io/pypi/dm/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)

<img src="https://github.com/styfenschaer/rocket-fft/blob/v0.1.0/assets/fourier.gif" width="300" />

Rocket-FFT makes [Numba](https://numba.pydata.org/) aware of `numpy.fft` and `scipy.fft`. It takes its name from the [PocketFFT](https://github.com/hayguen/pocketfft) Fast Fourier Transformation library that powers it, and Numba's goal of making your scientific Python code blazingly fast - like a rocket. 🚀

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

a = np.array([2, 7, 1, 8, 2, 8, 1, 8])
jit_fft(a)
```

## Supported NumPy Functions
The whole `numpy.fft` module is supported, which contains all the functions listed below:
- [x] `numpy.fft.fft`
- [x] `numpy.fft.ifft`
- [x] `numpy.fft.fft2`*
- [x] `numpy.fft.ifft2`*
- [x] `numpy.fft.fftn`*
- [x] `numpy.fft.ifftn`*
- [x] `numpy.fft.rfft`
- [x] `numpy.fft.irfft`
- [x] `numpy.fft.rfft2`
- [x] `numpy.fft.irfft2`
- [x] `numpy.fft.rfftn`
- [x] `numpy.fft.irfftn`
- [x] `numpy.fft.hfft`
- [x] `numpy.fft.ihfft`
- [x] `numpy.fft.fftfreq`
- [x] `numpy.fft.rfftfreq`
- [x] `numpy.fft.fftshift`
- [x] `numpy.fft.ifftshift`

\*Rocket-FFT follows SciPy's approach of not allowing duplicate axes

## Supported SciPy Functions
If you have SciPy installed, you will have support for most of the `scipy.fft` module, including:
- [x] `scipy.fft.fft`
- [x] `scipy.fft.ifft`
- [x] `scipy.fft.fft2`
- [x] `scipy.fft.ifft2`
- [x] `scipy.fft.fftn`
- [x] `scipy.fft.ifftn`
- [x] `scipy.fft.rfft`
- [x] `scipy.fft.irfft`
- [x] `scipy.fft.rfft2`
- [x] `scipy.fft.irfft2`
- [x] `scipy.fft.rfftn`
- [x] `scipy.fft.irfftn`
- [x] `scipy.fft.hfft`
- [x] `scipy.fft.ihfft`
- [x] `scipy.fft.hfft2`
- [x] `scipy.fft.ihfft2`
- [x] `scipy.fft.hfftn`
- [x] `scipy.fft.ihfftn`
- [x] `scipy.fft.dct`
- [x] `scipy.fft.dct2`
- [x] `scipy.fft.dctn`
- [x] `scipy.fft.idctn`
- [x] `scipy.fft.dst`
- [x] `scipy.fft.idst`
- [x] `scipy.fft.dstn`
- [x] `scipy.fft.idstn`
- [ ] `scipy.fft.fht`
- [ ] `scipy.fft.ifht`
- [x] `scipy.fft.fftshift`
- [x] `scipy.fft.ifftshift`
- [x] `scipy.fft.fftfreq`
- [x] `scipy.fft.ifftfreq`
- [ ] `scipy.fft.fhtoffset`
- [x] `scipy.fft.next_fast_len`

## Type Conversion
If SciPy is installed, Rocket-FFT follows SciPy's approach to type conversion, otherwise it follows NumPy's approach. 
You can change the type conversion rule by calling the `scipy_like` or `numpy_like` function from the `rocket_fft` namespace:
```python
from rocket_fft import numpy_like, scipy_like

numpy_like()
```
Both functions can be used regardless of whether SciPy is installed.
Note that this change is irreversible after the compilation of Rocket-FFT's internal functions.

## Performance Tip
Rocket-FFT achieves both, flexibility in function signatures similar to `scipy.fft` and `numpy.fft` and low compilation times.
Compilation takes a few hundred milliseconds in most cases. Calls with default arguments are treated specially and compile fastest.

## Low-Level Interface
Rocket-FFT also provides a low-level interface to the PocketFFT library. Using the low-level interface can significantly reduce compile time, minimize overhead and give more flexibility to the user. It also provides some functions that are not available through the SciPy and NumPy interfaces. You can import the functions of the low-level interface from the `rocket_fft` namespace:
```python
from rocket_fft import c2c, dct, ...
```
The low-level interface includes the following functions:
```python
def c2c(ain: NDArray[c8] | NDArray[c16], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: b1, fct: f8, nthreads: i8) -> None: ...
def r2c(ain: NDArray[f4] | NDArray[f8], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: b1, fct: f8, nthreads: i8) -> None: ...
def c2r(ain: NDArray[c8] | NDArray[c16], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], forward: b1, fct: f8, nthreads: i8) -> None: ...
# Like c2c but on real-valued input array and thus using symmetry
def c2c_sym(ain: NDArray[f4] | NDArray[f8], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: b1, fct: f8, nthreads: i8) -> None: ...
def dst(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], type: i8, fct: f8, ortho: b1, nthreads: i8) -> None: ...
def dct(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], type: i8, fct: f8, ortho: b1, nthreads: i8) -> None: ...
def r2r_separable_hartley(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], fct: f8, nthreads: i8) -> None: ...
def r2r_genuine_hartley(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], fct: f8, nthreads: i8) -> None: ...
def r2r_fftpack(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], real2hermitian: b1, forward: b1, fct: f8, nthreads: i8) -> None: ...
def good_size(target: i8, real: b1) -> i8: ...
```
Note that the low-level interface does not provide the same level of safety and convenience as the SciPy and NumPy interfaces. There is no safety net, and it is up to the user to ensure proper usage. You may want to look at the original [PocketFFT](https://github.com/hayguen/pocketfft) C++ implementation before using it.
