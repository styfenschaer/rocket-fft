# rocket-fft

Rocket-FFT makes [Numba](https://numba.pydata.org/) aware of `numpy.fft` and `scipy.fft`. It takes its name from the [PocketFFT](https://github.com/hayguen/pocketfft) Fast Fourier Transformation library used in the backend, and Numba's goal of making your scientific Python code blazingly fast — like a rocket 🚀.

Rocket-FFT is a *work in progress* project that has been a one-man show so far. I wrote it a long time ago and never published it because I thought it was wrong to publish it incomplete. Finally, I convinced myself that it is better to share a good Numba extension than to never share a perfect one. What does it mean for the user? Rocket-FFT has been tested against almost all [SciPy](https://scipy.org/) and [Numpy](https://numpy.org/) test suites (with some limitations due to Numba), which include more than 1,000 test cases. Therefore I would consider it somewhat safe to use. I still expect some bugs, probably related to the typing of the function overloads, for which I have not yet written tests. I very much welcome you to report bugs to me so I can iron them out. 

## Getting started
To use Rocket-FFT, just clone this repository, type
```
pip install -e .
```
and you are done. Rocket-FFT uses the *entry points* of `setuptools` to register itself as an extension of Numba. This means **no import**. Just do:
```python
import numba as nb 
import numpy as np

@nb.njit
def my_jitted_fft(x):
    return np.fft.fft(x)
```
Unfortunately I don't provide Python wheels yet and you have to build the shared library for the PocketFFT backend locally on your machine. For this you need to have a C++ compiler installed.
This is partly due to limited time and partly due to my lack of knowledge on how to distribute Python packages properly.

## Good to know
There are a number of points you may want to know when using Rocket-FFT for your project. Most of these relate to performance.
- When I wrote Rocket-FFT, I made some design decisions that do not match the behavior of SciPy and Numpy. For example, all functions will only transform `numpy.array`'s. This gives the user more responsibility, but also has the potential to avoid unnecessary copying. Rocket-FFT casts types in different ways compared to SciPy/Numpy. All non-floating-point types with less than 64 bits are cast to either `float32` or `complex64`. Larger types are cast to `float64` or `complex128`. 
- If the user requests to zero-pad/truncate the array and the datatype requires casting, Rocket-FFT is inefficient with the copies. This is not a big performance penalty, but it is better to make sure that the data type matches the one used for the transformation (one of `float32`, `complex64`, `float64`, `complex128`, depending on the transformation). 
- When performing transformations on half-precision data, Rocket-FFT is much faster than SciPy. If you don't need double precision but need speed, this may be good for you. I never checked why this happens (note that they have the same backend and give the same return type). It could be a bug in SciPy (or in Rocket-FFT 😄) 
- All transformations are based on the n-dimensional transformation. As a result, the n-dimensional transformation will *always* have the shortest compile time. E.g. `fftn` on a 1-dimensional array will compile faster than `fft`, even though they perform exactly the same operations behind the scene (`fft` has a tiny bit higher overhead). If compile time is important to you, always choose `fftn`. Ideally you also use the default arguments if possible. Rocket-FFT is written to minimize compile time for the default cases. I think this worked quite well, but test it yourself. 
- When performing the Fourier transform (`fft`) on real-valued data, it is slower than SciPy, even though we use the same backend. The reason is that Rocket-FFT does not yet take advantage of symmetry.
- The helper functions are not optimized in any way. They are implemented naively, just to provide full support for Numpy.
- Last but not least there is a little bonus. You get `numpy.roll` with all arguments (Numba supports only the first two) for free, because I used it for some other functions. But don't use it for large arrays on multiple axes if performance matters, because it`s a pretty naive implementation.

## Supported Numpy functions
The whole `numpy.fft` module is supported, which contains all the functions listed below.
Since Rocket-FFT was written around SciPy, the signature and argument list is the same as used in `scipy.fft`. This is not a big deal, since the signature of Numpy is just a subset of that of SciPy. However, when passing keyword arguments, care must be taken that the first argument, the array, is always `x` instead of `a`.
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
Most of the `scipy.fft` module is supported, including:
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

# Missing features and outlook
There are many small, but also big things I have planned with Rocket-FFT. I list the most important ones below.
- As you may have noticed, Rocket-FFT has to be build by the user, which is a big inconvenience for many Windows users. When I have time, I will take care of this and place it on PyPi as well.
- There are a few missing functions in the `scipy.fft` module what we can definitely fix. 
- Currently Rocket-FFT still lacks a test suite to test the typing, which I think is important to fix.
- Rocket-FFT is sometimes inefficient, for example it doesn`t use symmetry or makes copies. Fortunately, this is easy to fix.
- PocketFFT also implements two *Hartley transformations*. The C++ interface for them is written, but the Python part is still missing.
- Finally, when talking about FFT and Numba, one can't help but mention the Intel® MKL FFT library. It is **much** faster than PocketFFT and has a C interface. Using it within Numba in just-in-time compiled code would be great and should be quite easy to do (but still a ton of work).


