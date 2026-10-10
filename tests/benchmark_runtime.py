"""
Compares post-warmup execution times of Rocket-FFT (@njit) against SciPy and NumPy across:
- Small sizes (overhead-bound): N = 16, 64, 256
- Medium sizes (cache/vectorization): N = 1024, 4096, 16384
- Large sizes (throughput-bound): N = 65536, 262144, 1048576
- Prime sizes (Bluestein/chirp-Z): N = 1009, 10007
- 2D transforms: 64x64, 256x256, 512x512
- In-loop speedup (GIL-free @njit loop vs Python loop)
- Multithreading scaling (workers = 1, 2, 4, 8)
"""

import argparse
import time
import numba as nb
import numpy as np
import scipy.fft
import rocket_fft


# =============================================================================
# Benchmark Harness
# =============================================================================

def time_callable(fn, *args, min_seconds=0.1, max_repeats=10000, **kwargs):
    """Accurately time a callable over multiple repetitions after warmup."""
    # Warmup
    fn(*args, **kwargs)

    # Determine repeat count
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    t1 = time.perf_counter()
    single_duration = max(t1 - t0, 1e-7)

    repeats = max(1, min(int(min_seconds / single_duration), max_repeats))

    # Measurement
    t_start = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    t_end = time.perf_counter()

    avg_time_sec = (t_end - t_start) / repeats
    return avg_time_sec


def format_duration(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:6.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:6.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:6.2f} ms"
    else:
        return f"{seconds:6.2f} s "


# =============================================================================
# Benchmarks
# =============================================================================

def benchmark_1d_c2c(sizes, dtype=np.complex128):
    results = []

    # Compile jitted wrappers
    @nb.njit
    def nb_fft(a):
        return scipy.fft.fft(a)

    for n in sizes:
        a = (np.random.randn(n) + 1j * np.random.randn(n)).astype(dtype)

        # Warmup JIT
        nb_fft(a)

        t_rocket = time_callable(nb_fft, a)
        t_scipy = time_callable(scipy.fft.fft, a)
        t_numpy = time_callable(np.fft.fft, a)

        results.append({
            "test": f"1D c2c (N={n})",
            "rocket": t_rocket,
            "scipy": t_scipy,
            "numpy": t_numpy,
            "speedup_scipy": t_scipy / t_rocket,
            "speedup_numpy": t_numpy / t_rocket,
        })
    return results


def benchmark_1d_r2c(sizes, dtype=np.float64):
    results = []

    @nb.njit
    def nb_rfft(a):
        return scipy.fft.rfft(a)

    for n in sizes:
        a = np.random.randn(n).astype(dtype)
        nb_rfft(a)

        t_rocket = time_callable(nb_rfft, a)
        t_scipy = time_callable(scipy.fft.rfft, a)
        t_numpy = time_callable(np.fft.rfft, a)

        results.append({
            "test": f"1D r2c (N={n})",
            "rocket": t_rocket,
            "scipy": t_scipy,
            "numpy": t_numpy,
            "speedup_scipy": t_scipy / t_rocket,
            "speedup_numpy": t_numpy / t_rocket,
        })
    return results


def benchmark_2d(shapes, dtype=np.complex128):
    results = []

    @nb.njit
    def nb_fft2(a):
        return scipy.fft.fft2(a)

    for shape in shapes:
        a = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        nb_fft2(a)

        t_rocket = time_callable(nb_fft2, a)
        t_scipy = time_callable(scipy.fft.fft2, a)
        t_numpy = time_callable(np.fft.fft2, a)

        results.append({
            "test": f"2D c2c ({shape[0]}x{shape[1]})",
            "rocket": t_rocket,
            "scipy": t_scipy,
            "numpy": t_numpy,
            "speedup_scipy": t_scipy / t_rocket,
            "speedup_numpy": t_numpy / t_rocket,
        })
    return results


def benchmark_in_loop(n=1024, iterations=500):
    """Measure in-loop overhead: @njit loop vs Python loop calling scipy/numpy."""
    a = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128)

    @nb.njit
    def rocket_loop(a, iters):
        acc = 0.0
        for _ in range(iters):
            res = scipy.fft.fft(a)
            acc += res[0].real
        return acc

    def scipy_loop(a, iters):
        acc = 0.0
        for _ in range(iters):
            res = scipy.fft.fft(a)
            acc += res[0].real
        return acc

    def numpy_loop(a, iters):
        acc = 0.0
        for _ in range(iters):
            res = np.fft.fft(a)
            acc += res[0].real
        return acc

    rocket_loop(a, 1)

    t0 = time.perf_counter()
    rocket_loop(a, iterations)
    t_rocket = (time.perf_counter() - t0) / iterations

    t0 = time.perf_counter()
    scipy_loop(a, iterations)
    t_scipy = (time.perf_counter() - t0) / iterations

    t0 = time.perf_counter()
    numpy_loop(a, iterations)
    t_numpy = (time.perf_counter() - t0) / iterations

    return [{
        "test": f"In-Loop FFT (N={n}, {iterations} iters)",
        "rocket": t_rocket,
        "scipy": t_scipy,
        "numpy": t_numpy,
        "speedup_scipy": t_scipy / t_rocket,
        "speedup_numpy": t_numpy / t_rocket,
    }]


def benchmark_workers(n=262144, worker_counts=(1, 2, 4, 8)):
    results = []
    a = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128)

    # Dynamic workers jitted wrapper
    @nb.njit
    def nb_fft_workers(a, w):
        return scipy.fft.fft(a, workers=w)

    nb_fft_workers(a, 1)

    for w in worker_counts:
        t_rocket = time_callable(nb_fft_workers, a, w)
        t_scipy = time_callable(scipy.fft.fft, a, workers=w)

        results.append({
            "test": f"Workers={w} (N={n})",
            "rocket": t_rocket,
            "scipy": t_scipy,
            "numpy": float("nan"),
            "speedup_scipy": t_scipy / t_rocket,
            "speedup_numpy": float("nan"),
        })
    return results


# =============================================================================
# Pretty Printer
# =============================================================================

def print_table(results):
    header = f"{'Benchmark Test':<34} | {'Rocket-FFT':>11} | {'SciPy':>11} | {'NumPy':>11} | {'vs SciPy':>10} | {'vs NumPy':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        r_str = format_duration(r["rocket"])
        s_str = format_duration(r["scipy"])
        n_str = format_duration(r["numpy"]) if not np.isnan(r["numpy"]) else "   N/A    "
        sp_scipy = f"{r['speedup_scipy']:8.2f}x"
        sp_numpy = f"{r['speedup_numpy']:8.2f}x" if not np.isnan(r["speedup_numpy"]) else "   N/A   "
        print(f"{r['test']:<34} | {r_str:>11} | {s_str:>11} | {n_str:>11} | {sp_scipy:>10} | {sp_numpy:>10}")
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rocket-FFT Runtime Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark subset")
    args = parser.parse_args()

    if args.quick:
        sizes_c2c = [64, 1024, 16384, 1009]
        sizes_r2c = [1024, 16384]
        shapes_2d = [(64, 64), (256, 256)]
        worker_counts = [1, 2, 4]
        worker_size = 65536
    else:
        sizes_c2c = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1009, 10007]
        sizes_r2c = [64, 1024, 16384, 65536]
        shapes_2d = [(64, 64), (256, 256), (512, 512)]
        worker_counts = [1, 2, 4, 8]
        worker_size = 262144

    print("Running Rocket-FFT Runtime Performance Benchmarks...")
    all_results = []

    print("-> 1D Complex-to-Complex (c2c)...")
    all_results.extend(benchmark_1d_c2c(sizes_c2c))

    print("-> 1D Real-to-Complex (r2c)...")
    all_results.extend(benchmark_1d_r2c(sizes_r2c))

    print("-> 2D Complex-to-Complex (c2c)...")
    all_results.extend(benchmark_2d(shapes_2d))

    print("-> In-Loop Tight Execution Overhead...")
    all_results.extend(benchmark_in_loop())

    print("-> Multi-threading Scaling...")
    all_results.extend(benchmark_workers(n=worker_size, worker_counts=worker_counts))

    print_table(all_results)


if __name__ == "__main__":
    main()
