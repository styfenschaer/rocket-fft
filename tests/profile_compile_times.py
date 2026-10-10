"""
Measures pure JIT compilation times (isolating compiler warmup from transform lowering).
Covers fast-path (default arguments) and generic polymorphic paths across 1D, 2D, and ND
transforms for both SciPy and NumPy interfaces.
"""

import argparse
import os
import subprocess
import sys
from textwrap import dedent

import numpy as np

# Transform templates for isolated subprocess profiling
TESTS = {
    # 1D Transforms (Fast path)
    "scipy.fft.fft [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.fft(a)
        a = np.ones(16, dtype=np.complex128)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.ifft [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.ifft(a)
        a = np.ones(16, dtype=np.complex128)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.rfft [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.rfft(a)
        a = np.ones(16, dtype=np.float64)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.irfft [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.irfft(a)
        a = np.ones(9, dtype=np.complex128)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.dct [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.dct(a)
        a = np.ones(16, dtype=np.float64)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.dst [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.dst(a)
        a = np.ones(16, dtype=np.float64)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.fht [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.fht(a, 0.1, 0.5)
        a = np.ones(16, dtype=np.float64)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),

    # 2D & ND Transforms (Fast path)
    "scipy.fft.fft2 [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.fft2(a)
        a = np.ones((8, 8), dtype=np.complex128)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.fftn [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.fftn(a)
        a = np.ones((4, 4, 4), dtype=np.complex128)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.rfft2 [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return scipy.fft.rfft2(a)
        a = np.ones((8, 8), dtype=np.float64)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),

    # Generic Polymorphic Paths (explicit arguments)
    "scipy.fft.fft [generic]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a, n, axis, norm): return scipy.fft.fft(a, n, axis, norm)
        a = np.ones(16, dtype=np.complex128)
        tic = perf_counter(); func(a, 8, -1, 'ortho'); toc = perf_counter()
        print(toc - tic)
    """),
    "scipy.fft.fft2 [generic]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, scipy.fft, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a, s, axes, norm): return scipy.fft.fft2(a, s, axes, norm)
        a = np.ones((8, 8), dtype=np.complex128)
        tic = perf_counter(); func(a, (4, 4), (-2, -1), 'ortho'); toc = perf_counter()
        print(toc - tic)
    """),

    # NumPy equivalents (with out argument support)
    "numpy.fft.fft [fast]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a): return np.fft.fft(a)
        a = np.ones(16, dtype=np.complex128)
        tic = perf_counter(); func(a); toc = perf_counter()
        print(toc - tic)
    """),
    "numpy.fft.fft [with out]": dedent("""
        from time import perf_counter
        import numpy as np, numba as nb, rocket_fft
        nb.njit(lambda: None)()
        @nb.njit
        def func(a, out): return np.fft.fft(a, out=out)
        a = np.ones(16, dtype=np.complex128)
        out = np.empty_like(a)
        tic = perf_counter(); func(a, out); toc = perf_counter()
        print(toc - tic)
    """),
}


def run_benchmark(name, src, n_iter):
    filename = f"_prof_temp_{os.getpid()}.py"
    try:
        with open(filename, "w") as f:
            f.write(src)

        timings_ms = []
        for _ in range(n_iter):
            res = subprocess.check_output([sys.executable, filename])
            timings_ms.append(float(res.strip()) * 1000.0)

        t = np.array(timings_ms)
        return {
            "name": name,
            "min": np.amin(t),
            "mean": np.mean(t),
            "median": np.median(t),
            "max": np.amax(t),
            "std": np.std(t),
        }
    finally:
        if os.path.exists(filename):
            os.unlink(filename)


def print_summary_table(results):
    print("\n" + "=" * 80)
    print(f"{'Transform':<28} | {'Min (ms)':>9} | {'Mean (ms)':>10} | {'Median (ms)':>11} | {'Max (ms)':>9} | {'Std (ms)':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<28} | "
            f"{r['min']:9.2f} | "
            f"{r['mean']:10.2f} | "
            f"{r['median']:11.2f} | "
            f"{r['max']:9.2f} | "
            f"{r['std']:8.2f}"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Profile Rocket-FFT JIT compilation times")
    parser.add_argument("--iter", type=int, default=3, help="Number of profiling iterations per transform (default: 3)")
    parser.add_argument("--fast-only", action="store_true", help="Only profile fast-path transforms")
    parser.add_argument("--filter", type=str, default="", help="Filter transform names matching substring")
    args = parser.parse_args()

    tests = TESTS
    if args.fast_only:
        tests = {k: v for k, v in tests.items() if "[fast]" in k}
    if args.filter:
        tests = {k: v for k, v in tests.items() if args.filter in k}

    print(f"Profiling JIT compile times ({len(tests)} transforms, {args.iter} iterations each)...")

    results = []
    for name, src in tests.items():
        print(f"Profiling {name}...", end="", flush=True)
        r = run_benchmark(name, src, args.iter)
        results.append(r)
        print(f" Mean: {r['mean']:.1f} ms, Median: {r['median']:.1f} ms")

    print_summary_table(results)


if __name__ == "__main__":
    main()
