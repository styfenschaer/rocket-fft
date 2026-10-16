# Architecture

This document explains how Rocket-FFT is put together: what happens when jitted
code calls `np.fft.fft`, where each part lives, and which rules the code relies
on. It describes the design, not every detail; the comments in the code are the
reference for those. For setting up a development environment, testing and
releasing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## The big picture

Rocket-FFT teaches Numba to compile calls to `numpy.fft` and `scipy.fft` (plus
`numpy.roll` and two special functions). It has three layers:

```
  jitted user code:  y = np.fft.fft(x, n, axis, norm)
          │
          │  compile time (Python, rocket_fft/*.py)
          ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Numba layer: type the call, validate the argument types,  │
  │ decide the result type, emit ONE call into the library    │
  └───────────────────────────────────────────────────────────┘
          │
          │  run time (C++, rocket_fft/_rocket_fft_numba.cpp)
          ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Native frontend: validate values, plan the transform,     │
  │ allocate the result, convert/pad the input, call DUCC0    │
  └───────────────────────────────────────────────────────────┘
          │
          ▼
  ┌───────────────────────────────────────────────────────────┐
  │ DUCC0 (rocket_fft/ducc0/, vendored, unmodified)           │
  └───────────────────────────────────────────────────────────┘
```

The key design decision is that **almost all work happens at run time in C++**.
The code Numba compiles for each call site is a single function call with a
handful of arguments. That keeps compile times short (well below 200 ms per call
site, compared to seconds in early versions in certain cases) and makes every
combination of arguments equally cheap to compile.

## Following a call

Take `np.fft.fft(x)` inside an `@njit` function.

1. **Registration.** Importing `rocket_fft` registers every supported function
   with Numba through `_dispatch.implement` (called from `_numpy_fft.py`,
   `_scipy_fft.py`, `_utils.py` and `_extras.py`). Users do not even need to
   import it: Numba loads it through the `numba_extensions` entry point declared
   in `pyproject.toml`.

2. **Typing.** When Numba sees the call, it asks the typing template that
   `implement` created. The template binds the arguments to a *stub* function
   (for example `fft_1d_stub(a, n=None, axis=-1, norm=None, out=None)`), so
   keyword arguments and defaults behave like in NumPy. It then runs the
   function's `Signature` from `_signatures.py`, which raises a `TypingError`
   naming the offending argument (`The 2nd argument 'n' must be an integer...`).
   Finally the *kernel* computes the return type: dtype (`_kernels.result_dtype`)
   and Numba layout (`_numpy_layout`/`_scipy_layout`).

3. **Lowering.** The kernel's `codegen` emits the LLVM IR for the call, using
   helpers from `_codegen.py`: it packs the options into an `Options` struct,
   passes arrays as pointers to Numba's array structs and integer sequences
   (`s`, `axes`) as `IntSeq` descriptors, and folds everything known at compile
   time (omitted arguments, `None`, literal strings) into constants. Then it calls
   `rf_transform` and checks the returned status code.

4. **Run time, in C++.** `rf_transform` validates the values and builds a `Plan`
   (axes, transform lengths, normalization factor, thread count), allocates the
   result with exactly the memory layout NumPy or SciPy would produce
   (`numpy_result`/`scipy_result`), and dispatches to a runner (`run_c2c`,
   `run_r2c`, `run_dcst`, ...) that converts or pads the input if needed and calls
   DUCC0.

5. **Errors.** C++ never throws across the library boundary. Every entry point
   returns a `Status`; `check_status` in the generated code turns a non-zero code
   into the matching Python exception. The table of codes, messages and exception
   types lives in the C++ file and is read once at import (`_native.STATUS_TABLE`).

Why not Numba's `@overload`? An overload compiles a separate implementation
function for every signature, which roughly doubles the compile time of a cheap
call. `implement` registers a typing template and a lowering directly instead
(see the docstring of `_dispatch.py`).

## Code map

### Python (`rocket_fft/`)

| File | Role |
| :--- | :--- |
| `__init__.py` | Public namespace; importing it registers all implementations. Ideally, it remains backwards compatible with older Rocket-FFT versions. |
| `_dispatch.py` | `implement()`: registers a Python function as a Numba builtin backed by a kernel. |
| `_numpy_fft.py`, `_scipy_fft.py` | Which `numpy.fft`/`scipy.fft` functions exist, with their stubs and signatures. Declarative; no logic. |
| `_signatures.py` | Compile-time validation of argument types. |
| `_kernels.py` | Kernels: result dtype and layout rules, and the codegen for each kind of call. |
| `_codegen.py` | LLVM helpers: structs, argument conversion, status checking. |
| `_native.py` | Loads the native library (choosing the build, checking its ABI version) and mirrors its enums. |
| `_utils.py` | Workers and fast lengths, usable from Python and jitted code. |
| `_extras.py`, `special.py` | `numpy.roll` and the `loggamma`/`poch` ufuncs. |
| `lowlevel.py` | The low-level interface (`c2c`, `dct`, ...): thin `@intrinsic`s calling `rf_execute`. |
| `__init__.pyi` | Type stubs for the public API. |

### C++

`_rocket_fft_numba.cpp` is a single file, organized in sections (in this order):

- **Fixed-capacity arrays** (`DimArray`) for per-call bookkeeping, and `Buffer`
  (uninitialized scratch memory).
- **ABI** shared with Python: `kAbiVersion`, `Status`, dtype codes, `Kind`,
  `Flag`, `Norm`, and the `Options`, `IntSeq` and array-header structs.
- **Array allocation** through Numba's runtime (NRT), including the huge-page policy.
- **Strided copies** with dtype conversion (`copy_cast`, `copy_from`) and DUCC0 views.
- **Planning** (`make_plan` and friends) and **execution** (the `run_*` runners).
- **FHT**, **optimal lengths**, **circular shifts** (`roll`, `fftshift`).
- **Status table**, **output layouts** (`numpy_result`, `scipy_result`), and the
  **exported entry points** (`rf_*`, all `extern "C"`, all `noexcept`).

`_special_math.h` is a port of SciPy's `loggamma` and `poch`, needed by the FHT.

`ducc0/` is a vendored copy of the files of [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc)
that rocket-fft needs. Must stay byte-identical to upstream (see
CONTRIBUTING.md for how to update it).

## Rules the code relies on

**The Python side and the library share an ABI.** The enums in `_native.py`, the
struct layouts in `_codegen.py` and the signatures of the `rf_*` functions must
match the C++ file exactly. Whenever one of them changes, bump `kAbiVersion` in
the C++ file and `ABI_VERSION` in `_native.py` together. A mismatch, typically a
stale build in an editable install, then fails at import with a clear message
instead of misbehaving.

**Results match NumPy and SciPy**: same dtype, shape, values (to rounding), and
also the same memory layout (strides), including whether a SciPy result with
`overwrite_x=True` is the input or shares its memory. The layout rules exist
twice and must agree: in Python (`_numpy_layout`/`_scipy_layout`, which only
decide the Numba layout `C`/`F`/`A` at compile time) and in C++
(`numpy_result`/`scipy_result`, which compute the actual strides). The parity
tests compare all of this directly against NumPy and SciPy.

**Each library keeps its own semantics.** `numpy.fft` follows NumPy (repeated
axes are transformed once per occurrence), `scipy.fft` follows SciPy (repeated
axes are an error). The `Flag` bits passed from Python select the behaviour.

**No heap allocations for bookkeeping.** Shapes, strides and axes live in
`DimArray`s on the stack (capacity 64, NumPy's maximum number of dimensions).
DUCC0's views still need `std::vector`s; these are built from per-thread scratch
vectors that keep their capacity. A typical call now makes about seven heap
allocations in total (the result, DUCC0's views and DUCC0's internals).

**Arrays are owned by Numba.** Results are allocated through NRT with a custom
allocator (`aligned_malloc`), so Numba reference-counts and frees them like any
other array. On error, the entry point releases what it allocated before
returning (`guarded`).

**Thread safety.** The default number of workers is a global atomic; everything
else is per call or per thread. Calls from several threads, processes and
Numba's cache are tested.

## Builds and SIMD

The extension uses only CPython's stable ABI, so one wheel per platform
(`cp310-abi3`) serves every Python version. The `.so` doubles as a method-less
Python extension module only so that setuptools builds and locates it; the
functions are called through LLVM and ctypes, never through Python.

DUCC0 picks its SIMD width at compile time. On x86-64 the library is therefore
built twice (see `setup.py`): a generic build for the baseline instruction set,
and `_rocket_fft_numba_avx2` with AVX2 and FMA. At import, `_native.py` asks the
generic build whether the CPU and OS support AVX2 (`rf_cpu_supports_avx2`) and
loads the faster build it can use. `ROCKET_FFT_SIMD=generic|avx2` overrides the
choice, which is how CI tests the generic build. Floating-point contraction
(fused multiply-add) is disabled in the AVX2 build, so results stay close to
SciPy's and to the generic build.

## Design decisions

The non-obvious decisions, and why they were made:

- **DUCC0 as the backend**, vendored without modifications. It is PocketFFT's
  successor and the engine of `scipy.fft`, so parity with SciPy comes naturally.
  Not patching it keeps updates a matter of copying files.
- **One native call per Python call**, instead of building the logic in LLVM IR
  or with `@overload`: compile time is what users notice first.
- **Exact output layouts**, not just values: code that relies on contiguity
  (for example passing the result to C code) behaves the same with and without
  Numba.
- **Huge pages only for single-threaded results.** Backing large results with
  transparent huge pages (as NumPy does) cuts page faults on first touch and
  speeds up single-threaded transforms of large arrays by up to ~20%, but threads
  that first touch the same 2 MiB page wait for each other, which made
  multithreaded transforms up to 2x slower.
- **Copy into the output.** When the input must be copied anyway (dtype
  conversion, zero-padding), it is copied straight into the output and
  transformed in place, saving a scratch buffer as large as the result.
- **Runtime selection of the SIMD build** rather than compiling for the build
  machine's CPU as in rocket-fft 0.1.2 (my bad!).

## Tests and benchmarks

`tests/parity/` compares results with NumPy and SciPy (values, dtypes, layouts,
special functions); `tests/test_errors.py` checks that invalid arguments are
rejected like NumPy and SciPy reject them; the other files cover the low-level
interface, caching, threading and processes, the extras, and regressions.
`tests/common.py` holds jitted wrappers with the libraries' exact signatures, so
a test can call `NumpyFFT.fft(...)` exactly like `np.fft.fft(...)`.

`benchmarks/` holds runtime and compile-time benchmarks against SciPy and NumPy.
