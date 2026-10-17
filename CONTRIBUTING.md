# Contributing

This document covers the practical side of working on Rocket-FFT: setting up a
development environment, testing, benchmarking, updating the DUCC0 backend and
releasing. How the code is organized and why is described in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Development setup

You need Python 3.10 or newer, a C++17 compiler and pip 25.1 or newer (for
`--group`). Then:

```
$ git clone https://github.com/styfenschaer/rocket-fft.git
$ cd rocket-fft
$ python -m venv .venv && source .venv/bin/activate
$ pip install -e . --group dev
```

The `dev` dependency group (see `pyproject.toml`) contains the test tools
(`pytest`, SciPy) and the linter (`ruff`, pinned to the version CI uses).

**Rebuild after changing C++ code.** An editable install only picks up Python
changes automatically. After editing `_rocket_fft_numba.cpp` or
`_special_math.h`, run `pip install -e . --no-deps` again. If you forget, the
import fails with "does not match this version of rocket-fft" whenever the ABI
changed (see below), and otherwise silently runs the old code.

The library is built next to the sources as `*.abi3.so` for every Python
version, so editable installs in several virtual environments share one build:
rebuilding in any of them updates it for all.

## Tests

```
$ pytest
```

runs the whole suite (about 8000 tests, a few minutes). Useful variations:

```
$ pytest tests/parity/test_numpy_fft.py -k rfft    # a subset
$ ROCKET_FFT_SIMD=generic pytest                   # the generic build instead of AVX2
$ python tests/smoke.py                            # quick check of an installed package
```

How the tests are organized:

- `tests/parity/` compares results with NumPy and SciPy: values, dtypes and
  memory layouts. Most new tests belong here.
- `tests/test_errors.py`: invalid values raise the same exceptions as NumPy and
  SciPy; invalid types raise a `TypingError` that names the argument.
- `tests/test_lowlevel.py`, `test_runtime.py` (caching, threads, processes,
  workers), `test_extras.py` (`numpy.roll`, deprecated functions) and
  `test_regressions.py` (bugs that were fixed).

When writing a test, compare against the real library instead of hard-coding
expected results. `tests/common.py` provides jitted wrappers with the exact
signatures of `numpy.fft` and `scipy.fft` (`NumpyFFT.fft`, `ScipyFFT.dct`, ...)
and `assert_parity`, which checks dtype, shape, strides and values. Test file
names must be unique across directories, because pytest identifies them by name.

## Linting

```
$ ruff check .
```

The rule set is deliberately small (`pyproject.toml`): it catches bugs such as
unused or undefined names, without enforcing a formatting style. Please match
the style of the surrounding code, and comment the *why*, not the *what*.

## Benchmarks

```
$ python benchmarks/runtime.py --quick             # runtime vs. SciPy and NumPy
$ python benchmarks/runtime.py --only workers      # one section
$ python benchmarks/compile_times.py --fast-only   # JIT compile times
$ python benchmarks/helpers.py                     # fftfreq, fftshift
```

Timings on laptops are noisy (turbo boost, thermal throttling). For comparisons,
pin the process to one core (`taskset -c 3 python ...` on Linux) and compare
ratios against SciPy rather than absolute times, ideally over several runs.
Don't pin the multithreading benchmark (`--only workers`): `taskset` restricts
all threads of the process to the given cores.

## Changing the native library

- **Keep the ABI in sync.** If you change an exported `rf_*` function, the
  `Options` or `IntSeq` struct, or one of the shared enums, update the Python
  side (`_native.py`, `_codegen.py`) and bump `kAbiVersion` in the C++ file and
  `ABI_VERSION` in `_native.py` together.
- **Never throw across the boundary.** Entry points are `noexcept` and return a
  status code; wrap their bodies in `guarded`. New error conditions get a new
  `Status` and an entry in `kStatusTable` (with the exception type to raise);
  Python reads the table at import.
- **Avoid per-call heap allocations** in the common path: use `Shape`/`DimArray`
  for anything indexed by dimension, and `Buffer` for scratch memory.
- **Check both builds.** On x86-64, run the tests with and without
  `ROCKET_FFT_SIMD=generic`.

## Updating DUCC0

`rocket_fft/ducc0/` must stay an unmodified copy of upstream DUCC0, so that
updating it is just copying files. **Never patch it**; adapt the frontend
instead, or report problems upstream.

The vendored copy is upstream commit `a2d5a64` (2026-10-12) of
<https://gitlab.mpcdf.mpg.de/mtr/ducc>. To update:

1. Check out the new upstream version and copy every file that exists in
   `rocket_fft/ducc0/` from upstream's `src/ducc0/` (same relative paths). If the
   build then misses a header, copy that one too; if `setup.py`'s list of sources
   needs to change, update it.
2. Verify the copy: every file in `rocket_fft/ducc0/` must be byte-identical to
   upstream (for example with `diff -r`, limited to the vendored files).
3. Rebuild and run the tests, with both SIMD builds, and the benchmarks. Besides
   DUCC0's n-dimensional functions, the frontend's fast path uses the 1-D plan
   classes `ducc0::pocketfft_c` and `ducc0::pocketfft_r` and the half-complex
   order of the latter; `tests/test_fast_path.py` checks that it still matches
   DUCC0's general path exactly.
4. Update the commit above and, if needed, the copyright notice in
   `LICENSES/ducc0.txt`.

## Continuous integration

- **`ci.yml`** runs on pull requests, on pushes to `main`, weekly (to catch
  breakage from new Numba, NumPy or SciPy releases) and on demand. It lints and
  calls `build.yml`. Its final job, "CI passed", is the single status to require
  in branch protection.
- **`build.yml`** builds the sdist, builds the wheels *from the sdist* with
  cibuildwheel on native runners (Linux and Windows on x64 and ARM, macOS on
  Apple Silicon and Intel), and tests the installed wheels on every supported
  Python version, with the oldest supported dependencies, without SciPy, and
  with the generic SIMD build.
- **`release.yml`** publishes a release (see below).
- **Dependabot** checks monthly for new versions of the actions, which are pinned
  to commit hashes, and opens a pull request to update them.

cibuildwheel's settings live in `[tool.cibuildwheel]` in `pyproject.toml`, so a
CI build can be reproduced locally with `pipx run cibuildwheel` (Linux builds
need Docker).

## Releasing

1. Bump `rocket_fft/_version.py` and merge to `main`.
2. Tag the merge commit and push the tag:
   ```
   $ git tag v0.4.0
   $ git push origin v0.4.0
   ```
3. `release.yml` checks that the tag matches `_version.py`, builds and tests
   everything, publishes to TestPyPI, waits for approval of the `pypi`
   environment, publishes to PyPI and creates the GitHub release with generated
   notes. Tags like `v0.5.0rc1` produce pre-releases.

To rehearse a release, run `release.yml` manually on any branch: it builds,
tests and publishes to TestPyPI only. Publishing uses PyPI's Trusted Publishing,
so no API tokens are stored in the repository.
