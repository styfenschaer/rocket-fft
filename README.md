# rocket-fft

Rocket-FFT makes [Numba](https://numba.pydata.org/) aware of `numpy.fft` and `scipy.fft`. Rocket-FFT takes its name from the [PocketFFT](https://github.com/hayguen/pocketfft) Fast Fourier Transformation library that powers it, and Numba's goal of making your scientific Python code blazingly fast - like a rocket 🚀.

Rocket-FFT has been tested against both the [SciPy](https://scipy.org/) and [Numpy](https://numpy.org/) test suites, plus some additional typing tests. Therefore, it is considered safe to use, but the author still welcomes bug reports to help improve the project. 

## Getting started

The easiest way to get Rocket-FFT is to `pip install rocket-fft`. Alternatively, you can build it yourself by cloning this repository and running `pip install .`. The latter will require a C++ compiler to be installed on your system.

Rocket-FFT uses setuptools entry points to register itself as an extension of Numba, so there is no need for additional imports. Once installed, you can use it in your code like this:

```python
import numba as nb
import numpy as np

@nb.njit
def jit_fft(x):
    return np.fft.fft(x)
```

## Supported Numpy functions

The whole `numpy.fft` module is supported, which contains all the functions listed below:

- [x] `numpy.fft.fft`
- [x] `numpy.fft.ifft`
- [x] `numpy.fft.fft2`
- [x] `numpy.fft.ifft2`
- [x] `numpy.fft.fftn`
- [x] `numpy.fft.ifftn`
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

## Supported SciPy functions

Most of the `scipy.fft` module is supported as well, including:

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

## Type conversion

By default, Rocket-FFT follows SciPy's approach to type conversion. However, if you would like to use Numpy's approach instead, you can do so by calling the `numpy_like` function from the rocket_fft module, like this:

```python
from rocket_fft import numpy_like

numpy_like()
```

It is also possible to undo this change and switch back to the SciPy approach by calling the `scipy_like` function. Keep in mind that this change must be made before the internal functions of Rocket-FFT are compiled, as the type conversion rule is frozen upon compilation.

## Compilation time

Rocket-FFT uses several techniques to keep compilation times low, despite the flexible signatures of the `scipy.fft` and `numpy.fft` functions. One of the key strategies is to do as much work as possible at compile time. This allows Rocket-FFT to keep compilation times at a few hundred milliseconds. Calls with default arguments are handled specially and therefore compile the fastest

## Limitations on Linux and MacOS

Rocket-FFT uses a C-interface that is wrapped with `ctypes` on Linux and MacOS, due to an unresolved issue related to LLVM. As a result, all functions are not cacheable. However, on Windows the functions are cached without any issues.

## Low-level interface

If you don't need the convenience of the flexible signatures provided by SciPy and Numpy, you can use the low-level Pocketfft interface instead. Using the low-level interface can significantly reduce compilation times and has less overhead. Additionally, it offers some additional functions that are not available through the Scipy and Numpy interfaces. It is an option to consider if you prioritize speed on small transforms and have a good understanding of how to use FFT. You can use it like this:

```python
from rocket_fft import c2c

c2c(...)
```

The low-level interface implements the following functions:

```python
from numpy import complex64, complex128, float32, float64, int64, ndarray

# Note that single and double precision can't be mixed
complex_array = ndarray[complex64] | ndarray[complex128]
real_array = ndarray[float32] | ndarray[float64]

def c2c(ain: complex_array, aout: real_array, axes: ndarray[int64], forward: bool, fct: float64, nthreads: int64): ...
def r2c(ain: real_array, aout: complex_array, axes: ndarray[int64], forward: bool, fct: float64, nthreads: int64): ...
def c2r(ain: complex_array, aout: real_array, axes: ndarray[int64], forward: bool, fct: float64, nthreads: int64): ...
# Like c2c but on real-valued input array and thus using symmetry
def c2c_sym(ain: real_array, aout: complex_array, axes: ndarray[int64], forward: bool, fct: float64, nthreads: int64): ...
def dst(ain: real_array, aout: real_array, axes: ndarray[int64], type: int64, fct: float64, ortho: bool, nthreads: int64): ...
def dct(ain: real_array, aout: real_array, axes: ndarray[int64], type: int64, fct: float64, ortho: bool, nthreads: int64): ...
def separable_hartley(ain: real_array, aout: real_array, axes: ndarray[int64], fct: float64, nthreads: int64): ...
def genuine_hartley(ain: real_array, aout: real_array, axes: ndarray[int64], fct: float64, nthreads: int64): ...
def fftpack(ain: real_array, aout: real_array, axes: ndarray[int64], real2hermitian: bool, forward: bool, fct: float64, nthreads: int64): ...
def good_size(target: int64, real: bool): ...
```

It's important to note that the low-level Pocketfft interface does not provide the same level of safety and convenience as SciPy and Numpy. There is no safety net, and it is up to the user to ensure proper usage. You may need to refer to the original Pocketfft C++ implementation to understand how to use the functions properly. It is recommended to be familiar with the usage of FFT before using the low-level interface to avoid errors.
