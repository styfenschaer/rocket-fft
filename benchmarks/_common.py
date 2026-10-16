"""Timing and reporting shared by the benchmark scripts."""

import math
import time


def time_callable(fn, *args, min_seconds=0.1, max_repeats=100_000, trials=5, **kwargs):
    """Best per-call time over several trials, measured after a warm-up call."""
    fn(*args, **kwargs)  # warm up (and compile, for jitted functions)

    t0 = time.perf_counter()
    fn(*args, **kwargs)
    single = max(time.perf_counter() - t0, 1e-7)
    repeats = max(3, min(int(min_seconds / single), max_repeats))

    best = math.inf
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - t0) / repeats)
    return best


def time_loop(fn, *args, iters):
    """Per-iteration time of a function that loops `iters` times internally."""
    fn(*args, 1)  # warm up
    t0 = time.perf_counter()
    fn(*args, iters)
    return (time.perf_counter() - t0) / iters


def format_duration(seconds):
    if seconds is None or math.isnan(seconds):
        return "N/A"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def print_table(title, rows, references=("SciPy", "NumPy")):
    """Print rows of {"test": str, "Rocket-FFT": t, <reference>: t or None}.

    For each reference, a speedup column shows reference time / Rocket-FFT time.
    """
    columns = ["Rocket-FFT", *references]
    name_width = max([len("Benchmark")] + [len(r["test"]) for r in rows])
    header = f"{'Benchmark':<{name_width}}"
    header += "".join(f" | {c:>11}" for c in columns)
    header += "".join(f" | {'vs ' + c:>9}" for c in references)
    rule = "=" * len(header)

    print(f"\n{title}\n{rule}\n{header}\n{'-' * len(header)}")
    for r in rows:
        line = f"{r['test']:<{name_width}}"
        line += "".join(f" | {format_duration(r.get(c)):>11}" for c in columns)
        for c in references:
            ref = r.get(c)
            line += f" | {ref / r['Rocket-FFT']:8.2f}x" if ref is not None else f" | {'N/A':>9}"
        print(line)
    print(rule)
