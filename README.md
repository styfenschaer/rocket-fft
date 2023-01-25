# rocket-fft
Rocket-FFT makes [Numba](https://numba.pydata.org/) aware of `numpy.fft` and `scipy.fft`. Rocket-FFT takes its name from the [PocketFFT](https://github.com/hayguen/pocketfft) Fast Fourier Transformation library that powers it, and Numba's goal of making your scientific Python code blazingly fast - like a rocket. 🚀

Rocket-FFT has been tested against both the [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) test suites, plus some additional typing tests. Therefore, it is considered safe to use, but the author still welcomes bug reports to help improve the project. 

## Getting started
The easiest way to get Rocket-FFT is to:
```
pip install rocket-fft
```
Alternatively, you can build it from source:
```
git clone https://github.com/styfenschaer/rocket-fft.git
cd rocket-fft
python setup.py install
``` 
The latter requires a C++ compiler compatible with your Python installation.

Rocket-FFT uses `setuptools` entry points to register itself as an extension of Numba, so there is no need for additional imports. Once installed successfully, the following will work:
```python
import numba as nb
import numpy as np

@nb.njit
def jit_fft(x):
    return np.fft.fft(x)

a = np.array([2, 7, 1, 8, 2, 8, 1, 8])
jit_fft(a)
```

## Supported NumPy functions
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

## Supported SciPy functions
SciPy is an optional dependency of Numba and therefore also for Rocket-FFT. 
If SciPy is installed, you get support for most of the `scipy.fft` module, including:
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
If SciPy is installed, Rocket-FFT follows SciPy's approach to type conversion, otherwise it follows NumPy's approach. You can change the type conversion by calling the `scipy_like` or `numpy_like` function from the `rocket_fft` namespace:
```python
from rocket_fft import numpy_like, scipy_like

numpy_like()
scipy_like()
```
Keep in mind that these changes must be made before the internal functions of Rocket-FFT are compiled, as the type conversion rule is frozen upon the first compilation.
Both functions can be used regardless of whether SciPy is installed, since they do not rely on the package itself.

## Compilation time
Rocket-FFT employs multiple strategies to achieve both flexibility in function signatures similar to `scipy.fft` and `numpy.fft`, while minimizing compilation times.
One of the key approaches is to generate and assemble specialized functions based on the provided type information prior to compilation. This avoids compiling code that will never be executed, and Rocket-FFT can thus limit compilation times to a few hundred milliseconds in most cases. Calls with default arguments are handled specially, resulting in the fastest compilation times.

## Limitations on Linux and MacOS
Rocket-FFT uses a C interface to the PocketFFT C++ library. On Linux and MacOS, the C interface functions are accessed via the Python built-in `ctypes` module due to an unresolved issue with LLVM. This fallback solution results in all function not being cachable, which means that `numba.njit(cache=True)` does not work. On Windows the C interface is accessed differently and therefore all functions can be cached.

## Low-level interface
If you don't need the convenience of the flexible signatures provided by SciPy and NumPy, you can use the low-level interface to the PocketFFT library instead. Using the low-level interface can significantly reduce compilation times and has slightly lower overhead. Additionally, it offers some additional functions that are not available through the SciPy and NumPy interfaces. It is an option to consider if you prioritize speed on small transforms and have a good understanding of how to use FFT. You can import the low-level interface functions from `rocket_fft`:
```python
from rocket_fft import c2c, dct, ...
```
The low-level interface provides the following functions:
```python
from numpy import bool8, complex64, complex128, float32, float64, int64

# Note that single and double precision can't be mixed
complex_array = ndarray[complex64] | ndarray[complex128]
real_array = ndarray[float32] | ndarray[float64]

def c2c(ain: complex_array, aout: complex_array, axes: ndarray[int64], forward: bool8, fct: float64, nthreads: int64) -> None: ...
def r2c(ain: real_array, aout: complex_array, axes: ndarray[int64], forward: bool8, fct: float64, nthreads: int64) -> None: ...
def c2r(ain: complex_array, aout: real_array, axes: ndarray[int64], forward: bool8, fct: float64, nthreads: int64) -> None: ...
# Like c2c but on real-valued input array and thus using symmetry
def c2c_sym(ain: real_array, aout: complex_array, axes: ndarray[int64], forward: bool8, fct: float64, nthreads: int64) -> None: ...
def dst(ain: real_array, aout: real_array, axes: ndarray[int64], type: int64, fct: float64, ortho: bool8, nthreads: int64) -> None: ...
def dct(ain: real_array, aout: real_array, axes: ndarray[int64], type: int64, fct: float64, ortho: bool8, nthreads: int64) -> None: ...
def separable_hartley(ain: real_array, aout: real_array, axes: ndarray[int64], fct: float64, nthreads: int64) -> None: ...
def genuine_hartley(ain: real_array, aout: real_array, axes: ndarray[int64], fct: float64, nthreads: int64) -> None: ...
def fftpack(ain: real_array, aout: real_array, axes: ndarray[int64], real2hermitian: bool8, forward: bool8, fct: float64, nthreads: int64) -> None: ...
def good_size(target: int64, real: bool8) -> int64: ...
```
It's important to note that the low-level interface does not provide the same level of safety and convenience as SciPy and NumPy. There is no safety net, and it is up to the user to ensure proper usage. You may even need to refer to the original [PocketFFT](https://github.com/hayguen/pocketfft) C++ implementation to understand how to use the functions properly. 
