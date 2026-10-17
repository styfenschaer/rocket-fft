# Rocket-FFT
[![PyPI version](https://img.shields.io/pypi/v/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![License](https://img.shields.io/pypi/l/rocket-fft?color=%2376519B)](https://opensource.org/licenses/BSD-3-Clause)
[![python](https://img.shields.io/pypi/pyversions/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![status](https://img.shields.io/pypi/status/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)
[![downloads](https://img.shields.io/pypi/dm/rocket-fft?color=%2376519B)](https://pypi.org/project/rocket-fft/)

![](https://raw.githubusercontent.com/styfenschaer/rocket-fft/main/assets/fourier.gif)

Rocket-FFT makes [Numba](https://numba.pydata.org/) aware of `numpy.fft` and `scipy.fft`. It takes its name from the [PocketFFT](https://github.com/mreineck/pocketfft) library that originally powered it, and Numba's goal of making your scientific Python code blazingly fast - like a rocket. Today, Rocket-FFT is powered by PocketFFT's successor [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc), the same library that runs `scipy.fft`.

## Getting Started
The easiest way to get Rocket-FFT is to:
```
$ pip install rocket-fft
```
Alternatively, you can build it from source:
```
$ git clone https://github.com/styfenschaer/rocket-fft.git
$ cd rocket-fft
$ pip install .
```
The latter requires a C++17 compiler. Rocket-FFT requires Python 3.10 or newer, Numba 0.60 or newer and NumPy 2.0 or newer. SciPy is optional.

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

## Supported Functions
Rocket-FFT supports all functions of `numpy.fft` and `scipy.fft` in jitted code, with all their arguments. The results match NumPy and SciPy, down to the data type and the memory layout of the returned arrays. The exceptions are:

- SciPy's backend functions (`set_backend`, `skip_backend`, `set_global_backend` and `register_backend`) and its `set_workers` and `get_workers`. Use the `workers` argument or Rocket-FFT's own `set_workers` (see below) instead.
- The `plan` argument must be `None`, `xp` must be `None` or NumPy, and `device` must be `None` or `"cpu"`.
- An empty `axes` tuple is not allowed.
- Half precision (`float16`) and extended precision (`longdouble`) are not supported.

Rocket-FFT also extends Numba's `numpy.roll` to NumPy's full signature `roll(a, shift, axis=None)`, and provides `loggamma` and `poch` (as in `scipy.special`) in `rocket_fft.special`. These are ufuncs that work in pure Python and in jitted code.

## Performance Tips
Compiling a call takes well below 200 milliseconds, whichever arguments you pass. Compiled functions can be cached with `@nb.njit(cache=True)`.

Multidimensional transforms can use several threads. Pass `workers` to the `scipy.fft` functions, or set the default for all transforms with `set_workers`, which also affects functions that were compiled before:
```python
import rocket_fft

rocket_fft.set_workers(4)
rocket_fft.get_workers()  # 4
```
FFTs are fastest for lengths with small prime factors. `scipy.fft.next_fast_len` and `scipy.fft.prev_fast_len` find such lengths, also in jitted code:
```python
import scipy.fft

@nb.njit
def padded_fft(x):
    n = scipy.fft.next_fast_len(x.size)
    return np.fft.fft(x, n)
```
If SciPy is not installed, use `rocket_fft.next_fast_len` and `rocket_fft.prev_fast_len` instead. They work the same way.

On x86-64, Rocket-FFT contains two builds of its native library and picks the faster one your CPU supports: one that uses AVX2 instructions, and a generic one. You can force one of them by setting the environment variable `ROCKET_FFT_SIMD` to `generic` or `avx2`.
This must happen before Rocket-FFT is imported or any function using it is compiled:
```python
import os
os.environ["ROCKET_FFT_SIMD"] = "generic"
```

## NumPy and SciPy Semantics
NumPy and SciPy show a subtle difference in how they handle the `axes` argument: NumPy allows duplicate axes in its multidimensional transforms (as in `fftn(x, axes=(0, 0))`), whereas SciPy doesn't. Rocket-FFT follows each library: calls to `numpy.fft` behave like NumPy, calls to `scipy.fft` like SciPy.

## Low-Level Interface
Rocket-FFT also provides a low-level interface to DUCC0. It has the least overhead, gives the most flexibility, and includes transforms that are not available through `numpy.fft` and `scipy.fft`. Its functions can only be called from jitted code. You can import them from the `rocket_fft` namespace:
```python
from rocket_fft import c2c, dct, ...
```
The low-level interface includes the following functions:
```python
def c2c(ain: NDArray[c8] | NDArray[c16], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: bool, fct: float, nthreads: int) -> None: ...
def r2c(ain: NDArray[f4] | NDArray[f8], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: bool, fct: float, nthreads: int) -> None: ...
def c2r(ain: NDArray[c8] | NDArray[c16], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], forward: bool, fct: float, nthreads: int) -> None: ...
def c2c_sym(ain: NDArray[f4] | NDArray[f8], aout: NDArray[c8] | NDArray[c16], axes: NDArray[i8], forward: bool, fct: float, nthreads: int) -> None: ...
def dct(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], type: int, fct: float, ortho: bool, nthreads: int) -> None: ...
def dst(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], type: int, fct: float, ortho: bool, nthreads: int) -> None: ...
def r2r_separable_hartley(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], fct: float, nthreads: int) -> None: ...
def r2r_genuine_hartley(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], fct: float, nthreads: int) -> None: ...
def r2r_fftpack(ain: NDArray[f4] | NDArray[f8], aout: NDArray[f4] | NDArray[f8], axes: NDArray[i8], real2hermitian: bool, forward: bool, fct: float, nthreads: int) -> None: ...
def good_size(target: int, real: bool = False) -> int: ...
def prev_good_size(target: int, real: bool = False) -> int: ...
```
The functions write their result into `aout`, whose shape determines the transform lengths: where `ain` and `aout` differ in size along an axis, the input is zero-padded or truncated. The scale factor `fct` is applied as is, without any normalization. For example:
```python
from rocket_fft import c2c

@nb.njit
def unnormalized_fft(x):
    out = np.empty_like(x)
    c2c(x, out, np.array([0]), True, 1.0, 1)
    return out
```
Note that the low-level interface provides less safety and convenience than `numpy.fft` and `scipy.fft`. You may want to consult the [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) documentation before using it.

## Contributing
Contributions are welcome. [CONTRIBUTING.md](https://github.com/styfenschaer/rocket-fft/blob/main/CONTRIBUTING.md) explains how to set up a development environment, run the tests and benchmarks, and release; [ARCHITECTURE.md](https://github.com/styfenschaer/rocket-fft/blob/main/ARCHITECTURE.md) explains how Rocket-FFT works internally.

## License
Rocket-FFT is distributed under the [BSD-3-Clause License](https://github.com/styfenschaer/rocket-fft/blob/main/LICENSE). The bundled DUCC0 sources are dual-licensed under BSD-3-Clause or GPL-2.0-or-later and are used under the BSD-3-Clause terms. Parts of Rocket-FFT are adapted from [SciPy](https://scipy.org), [NumPy](https://numpy.org) and [pypocketfft](https://gitlab.mpcdf.mpg.de/mtr/pypocketfft), and its interface to [Numba](https://numba.pydata.org)'s runtime mirrors Numba's headers. All their licenses are in [LICENSES](https://github.com/styfenschaer/rocket-fft/tree/main/LICENSES).
