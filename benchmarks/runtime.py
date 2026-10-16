"""Runtime of Rocket-FFT (@njit) against SciPy and NumPy, after compilation.

Covers 1D and 2D transforms from overhead-bound to throughput-bound sizes,
prime sizes (Bluestein), calls inside compiled loops, multithreading, the fast
Hankel transform and the extras (numpy.roll, rocket_fft.special).

    python benchmarks/runtime.py [--quick] [--only SECTION ...]
"""

import argparse

import numba as nb
import numpy as np
import scipy.fft
import scipy.special
from _common import print_table, time_callable, time_loop

import rocket_fft
from rocket_fft.special import loggamma, poch

rng = np.random.default_rng(0)


def _complex(*shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _row(test, rocket, scipy_ref=None, numpy_ref=None):
    return {"test": test, "Rocket-FFT": rocket, "SciPy": scipy_ref, "NumPy": numpy_ref}


# =============================================================================
# FFT
# =============================================================================


@nb.njit
def nb_fft(a):
    return scipy.fft.fft(a)


@nb.njit
def nb_rfft(a):
    return scipy.fft.rfft(a)


@nb.njit
def nb_fft2(a):
    return scipy.fft.fft2(a)


@nb.njit
def nb_fft_workers(a, workers):
    return scipy.fft.fft(a, workers=workers)


@nb.njit
def nb_fft_loop(a, iters):
    acc = 0.0
    for _ in range(iters):
        acc += scipy.fft.fft(a)[0].real
    return acc


def py_fft_loop(fft):
    def loop(a, iters):
        acc = 0.0
        for _ in range(iters):
            acc += fft(a)[0].real
        return acc

    return loop


def bench_fft(quick):
    sizes_c2c = [64, 1024, 16384, 1009] if quick else [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1009, 10007]
    sizes_r2c = [1024, 16384] if quick else [64, 1024, 16384, 65536]
    shapes_2d = [(64, 64), (256, 256)] if quick else [(64, 64), (256, 256), (512, 512)]

    rows = []
    for n in sizes_c2c:
        a = _complex(n)
        rows.append(_row(f"1D c2c (N={n})", time_callable(nb_fft, a),
                         time_callable(scipy.fft.fft, a), time_callable(np.fft.fft, a)))
    for n in sizes_r2c:
        a = rng.standard_normal(n)
        rows.append(_row(f"1D r2c (N={n})", time_callable(nb_rfft, a),
                         time_callable(scipy.fft.rfft, a), time_callable(np.fft.rfft, a)))
    for shape in shapes_2d:
        a = _complex(*shape)
        rows.append(_row(f"2D c2c ({shape[0]}x{shape[1]})", time_callable(nb_fft2, a),
                         time_callable(scipy.fft.fft2, a), time_callable(np.fft.fft2, a)))

    n, iters = 1024, 500
    a = _complex(n)
    rows.append(_row(f"In-loop FFT (N={n}, {iters} iters)", time_loop(nb_fft_loop, a, iters=iters),
                     time_loop(py_fft_loop(scipy.fft.fft), a, iters=iters),
                     time_loop(py_fft_loop(np.fft.fft), a, iters=iters)))
    print_table("FFT", rows)


@nb.njit
def nb_fft_axis0_workers(a, workers):
    return scipy.fft.fft(a, axis=0, workers=workers)


@nb.njit
def nb_fft2_workers(a, workers):
    return scipy.fft.fft2(a, workers=workers)


@nb.njit
def nb_rfftn_workers(a, workers):
    return scipy.fft.rfftn(a, workers=workers)


@nb.njit
def nb_dctn_workers(a, workers):
    return scipy.fft.dctn(a, workers=workers)


def bench_workers(quick):
    # DUCC0 parallelizes over the lanes of a transform (the other axes), so a single
    # 1-D transform does not benefit from threads; these inputs all have many lanes.
    cases = [
        ("fft 512x4096 axis=-1", nb_fft_workers, lambda a, w: scipy.fft.fft(a, workers=w), _complex(512, 4096)),
        ("fft 4096x512 axis=0", nb_fft_axis0_workers, lambda a, w: scipy.fft.fft(a, axis=0, workers=w),
         _complex(4096, 512)),
        ("fft2 1024x1024", nb_fft2_workers, lambda a, w: scipy.fft.fft2(a, workers=w), _complex(1024, 1024)),
        ("rfftn 128x128x128", nb_rfftn_workers, lambda a, w: scipy.fft.rfftn(a, workers=w),
         rng.standard_normal((128, 128, 128))),
        ("dctn 1024x1024", nb_dctn_workers, lambda a, w: scipy.fft.dctn(a, workers=w),
         rng.standard_normal((1024, 1024))),
    ]
    if quick:
        cases = cases[::2]
    rows = []
    for name, jitted, reference, a in cases:
        base_rocket = base_scipy = None
        for workers in [1, 2, 4] if quick else [1, 2, 4, 8]:
            t_rocket, t_scipy = time_callable(jitted, a, workers), time_callable(reference, a, workers)
            base_rocket, base_scipy = base_rocket or t_rocket, base_scipy or t_scipy
            rows.append(_row(f"{name} workers={workers} "
                             f"(scaling {base_rocket / t_rocket:.1f}x / {base_scipy / t_scipy:.1f}x)",
                             t_rocket, t_scipy))
    print_table("Multithreading (scaling vs. one thread: Rocket-FFT / SciPy)", rows, references=("SciPy",))


# =============================================================================
# Fast Hankel transform
# =============================================================================


@nb.njit
def nb_fht(a, dln, mu, offset, bias):
    return scipy.fft.fht(a, dln, mu, offset, bias)


@nb.njit
def nb_ifht(a, dln, mu, offset, bias):
    return scipy.fft.ifht(a, dln, mu, offset, bias)


@nb.njit
def nb_fhtoffset(dln, mu, initial, bias):
    return scipy.fft.fhtoffset(dln, mu, initial, bias)


@nb.njit
def nb_fhtoffset_loop(dln, mu, initial, bias, iters):
    acc = 0.0
    for _ in range(iters):
        acc += scipy.fft.fhtoffset(dln, mu, initial, bias)
    return acc


def py_fhtoffset_loop(dln, mu, initial, bias, iters):
    acc = 0.0
    for _ in range(iters):
        acc += scipy.fft.fhtoffset(dln, mu, initial, bias)
    return acc


@nb.njit
def nb_fht_loop(a, dln, mu, offset, bias, iters):
    acc = 0.0
    for _ in range(iters):
        acc += scipy.fft.fht(a, dln, mu, offset, bias)[0]
    return acc


def py_fht_loop(a, dln, mu, offset, bias, iters):
    acc = 0.0
    for _ in range(iters):
        acc += scipy.fft.fht(a, dln, mu, offset, bias)[0]
    return acc


def bench_fht(quick):
    sizes = [256, 4096] if quick else [64, 256, 1024, 4096, 16384, 65536]
    dln, mu, offset = 0.1, 0.5, 0.2

    rows = []
    for n in sizes:
        a = rng.standard_normal(n)
        rows.append(_row(f"fht (N={n})", time_callable(nb_fht, a, dln, mu, offset, 0.0),
                         time_callable(scipy.fft.fht, a, dln, mu, offset, 0.0)))
    for n in sizes:
        # Spans ~4 decades regardless of n, which keeps the biased transform finite.
        dln_biased = 8.0 / max(n - 1, 1)
        a = rng.standard_normal(n)
        rows.append(_row(f"fht bias=0.5 (N={n})", time_callable(nb_fht, a, dln_biased, mu, offset, 0.5),
                         time_callable(scipy.fft.fht, a, dln_biased, mu, offset, 0.5)))
    for n in sizes:
        a = rng.standard_normal(n)
        rows.append(_row(f"ifht (N={n})", time_callable(nb_ifht, a, dln, mu, offset, 0.0),
                         time_callable(scipy.fft.ifht, a, dln, mu, offset, 0.0)))
    for shape in [(16, 256)] if quick else [(16, 256), (32, 1024), (8, 4096)]:
        a = rng.standard_normal(shape)
        rows.append(_row(f"fht batched ({shape[0]}x{shape[1]})", time_callable(nb_fht, a, dln, mu, offset, 0.0),
                         time_callable(scipy.fft.fht, a, dln, mu, offset, 0.0)))

    rows.append(_row("fhtoffset", time_callable(nb_fhtoffset, dln, mu, offset, 0.0),
                     time_callable(scipy.fft.fhtoffset, dln, mu, offset, 0.0)))
    iters = 10_000
    rows.append(_row(f"In-loop fhtoffset ({iters} iters)",
                     time_loop(nb_fhtoffset_loop, dln, mu, offset, 0.0, iters=iters),
                     time_loop(py_fhtoffset_loop, dln, mu, offset, 0.0, iters=iters)))
    a, iters = rng.standard_normal(256), 500
    rows.append(_row(f"In-loop fht (N=256, {iters} iters)",
                     time_loop(nb_fht_loop, a, dln, mu, offset, 0.0, iters=iters),
                     time_loop(py_fht_loop, a, dln, mu, offset, 0.0, iters=iters)))
    print_table("Fast Hankel transform", rows, references=("SciPy",))


# =============================================================================
# Extras
# =============================================================================


@nb.njit
def nb_roll_axes(a):
    return np.roll(a, (3, -5), axis=(0, 1))


@nb.njit
def nb_roll_flat(a):
    return np.roll(a, 7, axis=None)


@nb.njit
def nb_loggamma(a):
    return loggamma(a)


@nb.njit
def nb_poch(a):
    return poch(a, 1.5)


def bench_extras(quick):
    n = 65536
    a2 = rng.standard_normal((256, 256))
    x = rng.random(n) * 50
    z = x + 1j * rng.standard_normal(n)
    rows = [
        _row("roll 256x256 axes=(0,1)", time_callable(nb_roll_axes, a2),
             numpy_ref=time_callable(np.roll, a2, (3, -5), axis=(0, 1))),
        _row("roll 256x256 axis=None", time_callable(nb_roll_flat, a2),
             numpy_ref=time_callable(np.roll, a2, 7, axis=None)),
        _row(f"loggamma real (N={n})", time_callable(nb_loggamma, x), time_callable(scipy.special.loggamma, x)),
        _row(f"loggamma complex (N={n})", time_callable(nb_loggamma, z), time_callable(scipy.special.loggamma, z)),
        _row(f"poch (N={n})", time_callable(nb_poch, x), time_callable(scipy.special.poch, x, 1.5)),
    ]
    print_table("Extras (numpy.roll, rocket_fft.special)", rows)


SECTIONS = {"fft": bench_fft, "workers": bench_workers, "fht": bench_fht, "extras": bench_extras}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="run a smaller set of sizes")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, help="run only these sections")
    args = parser.parse_args()

    print(f"rocket-fft {rocket_fft.__version__}, numba {nb.__version__}, "
          f"numpy {np.__version__}, scipy {scipy.__version__}")
    for name in args.only or SECTIONS:
        SECTIONS[name](args.quick)


if __name__ == "__main__":
    main()
