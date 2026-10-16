"""JIT compilation time of each Rocket-FFT overload.

Every measurement runs in a fresh interpreter, after Numba itself has been warmed
up by compiling an empty function, so it isolates the cost of lowering one call.
"[fast]" cases use default arguments only; "[generic]" cases pass every argument.

    python benchmarks/compile_times.py [--iter N] [--fast-only] [--category TEXT] [--filter TEXT]
"""

import argparse
import statistics
import subprocess
import sys
import tempfile

TEMPLATE = """
from time import perf_counter
import numpy as np, numba as nb, scipy.fft, rocket_fft
nb.njit(lambda: None)()
{setup}
@nb.njit
def func({params}):
    return {expr}
tic = perf_counter(); func({params}); toc = perf_counter()
print(toc - tic)
"""

C16 = "np.ones(16, dtype=np.complex128)"
R16 = "np.ones(16)"

# category -> {name: (expression, {parameter: setup expression})}
CATEGORIES = {
    "1D fast-path transforms": {
        "scipy.fft.fft [fast]": ("scipy.fft.fft(a)", {"a": C16}),
        "scipy.fft.ifft [fast]": ("scipy.fft.ifft(a)", {"a": C16}),
        "scipy.fft.rfft [fast]": ("scipy.fft.rfft(a)", {"a": R16}),
        "scipy.fft.irfft [fast]": ("scipy.fft.irfft(a)", {"a": "np.ones(9, dtype=np.complex128)"}),
        "scipy.fft.dct [fast]": ("scipy.fft.dct(a)", {"a": R16}),
        "scipy.fft.dst [fast]": ("scipy.fft.dst(a)", {"a": R16}),
        "numpy.fft.fft [fast]": ("np.fft.fft(a)", {"a": C16}),
        "numpy.fft.rfft [fast]": ("np.fft.rfft(a)", {"a": R16}),
    },
    "2D & ND fast-path transforms": {
        "scipy.fft.fft2 [fast]": ("scipy.fft.fft2(a)", {"a": "np.ones((8, 8), dtype=np.complex128)"}),
        "scipy.fft.rfft2 [fast]": ("scipy.fft.rfft2(a)", {"a": "np.ones((8, 8))"}),
        "scipy.fft.fftn [fast]": ("scipy.fft.fftn(a)", {"a": "np.ones((4, 4, 4), dtype=np.complex128)"}),
        "numpy.fft.fft2 [fast]": ("np.fft.fft2(a)", {"a": "np.ones((8, 8), dtype=np.complex128)"}),
    },
    "Generic paths": {
        "scipy.fft.fft [generic]": (
            "scipy.fft.fft(a, n, axis, norm)",
            {"a": C16, "n": "8", "axis": "-1", "norm": "'ortho'"},
        ),
        "scipy.fft.fft2 [generic]": (
            "scipy.fft.fft2(a, s, axes, norm)",
            {"a": "np.ones((8, 8), dtype=np.complex128)", "s": "(4, 4)", "axes": "(-2, -1)", "norm": "'ortho'"},
        ),
        "numpy.fft.fft [with out]": ("np.fft.fft(a, out=out)", {"a": C16, "out": f"np.empty_like({C16})"}),
    },
    "Fast Hankel transform": {
        "scipy.fft.fht": ("scipy.fft.fht(a, 0.1, 0.5)", {"a": R16}),
        "scipy.fft.ifht": ("scipy.fft.ifht(a, 0.1, 0.5)", {"a": R16}),
        "scipy.fft.fhtoffset": ("scipy.fft.fhtoffset(dln, mu)", {"dln": "0.1", "mu": "0.5"}),
    },
    "Frequency, shift and size helpers": {
        "numpy.fft.fftfreq": ("np.fft.fftfreq(n, 0.1)", {"n": "100"}),
        "numpy.fft.rfftfreq": ("np.fft.rfftfreq(n, 0.1)", {"n": "100"}),
        "numpy.fft.fftshift": ("np.fft.fftshift(a)", {"a": R16}),
        "numpy.fft.ifftshift": ("np.fft.ifftshift(a)", {"a": R16}),
        "scipy.fft.fftfreq": ("scipy.fft.fftfreq(n, 0.1)", {"n": "100"}),
        "scipy.fft.rfftfreq": ("scipy.fft.rfftfreq(n, 0.1)", {"n": "100"}),
        "scipy.fft.fftshift": ("scipy.fft.fftshift(a)", {"a": R16}),
        "scipy.fft.ifftshift": ("scipy.fft.ifftshift(a)", {"a": R16}),
        "scipy.fft.next_fast_len": ("scipy.fft.next_fast_len(n)", {"n": "100"}),
        "scipy.fft.prev_fast_len": ("scipy.fft.prev_fast_len(n)", {"n": "100"}),
        "rocket_fft.get_workers": ("rocket_fft.get_workers()", {}),
    },
    "Extras (numpy.roll, rocket_fft.special)": {
        "numpy.roll [axis]": ("np.roll(a, 3, axis=1)", {"a": "np.random.rand(16, 16)"}),
        "numpy.roll [axes tuple]": ("np.roll(a, (1, 2), axis=(0, 1))", {"a": "np.random.rand(16, 16)"}),
        "numpy.roll [axis=None]": ("np.roll(a, (1, 2), axis=None)", {"a": "np.random.rand(16, 16)"}),
        "numpy.roll [list input]": ("np.roll(a, 1, axis=0)", {"a": "[1.0, 2.0, 3.0]"}),
        "special.loggamma [array]": ("rocket_fft.special.loggamma(a)", {"a": "np.random.rand(16, 16)"}),
        "special.loggamma [complex]": ("rocket_fft.special.loggamma(z)", {"z": "1.0 + 2.0j"}),
        "special.poch [array]": ("rocket_fft.special.poch(a, 1.5)", {"a": "np.random.rand(16, 16)"}),
    },
}


def measure(expr, args, n_iter):
    """Compile time in milliseconds of each of `n_iter` fresh interpreters."""
    setup = "\n".join(f"{name} = {value}" for name, value in args.items())
    src = TEMPLATE.format(setup=setup, params=", ".join(args), expr=expr)
    # `python -c` puts the working directory first on sys.path; running from a neutral
    # directory keeps a source checkout from shadowing the installed package.
    cmd = [sys.executable, "-c", src]
    return [float(subprocess.check_output(cmd, cwd=tempfile.gettempdir())) * 1e3 for _ in range(n_iter)]


def print_summary(results):
    header = f"{'Function':<34} | {'Min (ms)':>9} | {'Mean (ms)':>10} | {'Median (ms)':>11} | {'Max (ms)':>9}"
    print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
    for category, rows in results.items():
        print(f"--- {category} ---")
        for name, t in rows:
            print(f"{name:<34} | {min(t):9.1f} | {statistics.mean(t):10.1f} | "
                  f"{statistics.median(t):11.1f} | {max(t):9.1f}")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iter", type=int, default=3, help="interpreters per function (default: 3)")
    parser.add_argument("--fast-only", action="store_true", help="only the [fast] cases")
    parser.add_argument("--category", default="", help="only categories containing this text")
    parser.add_argument("--filter", default="", help="only functions containing this text")
    args = parser.parse_args()

    selected = {}
    for category, cases in CATEGORIES.items():
        if args.category.lower() not in category.lower():
            continue
        cases = {
            name: case for name, case in cases.items()
            if args.filter.lower() in name.lower() and (not args.fast_only or "[fast]" in name)
        }
        if cases:
            selected[category] = cases

    total = sum(map(len, selected.values()))
    print(f"Profiling {total} functions, {args.iter} interpreters each...")
    results = {}
    for category, cases in selected.items():
        print(f"\n{category}")
        results[category] = []
        for name, (expr, call_args) in cases.items():
            print(f"  {name}...", end="", flush=True)
            t = measure(expr, call_args, args.iter)
            results[category].append((name, t))
            print(f" median {statistics.median(t):.1f} ms")
    print_summary(results)


if __name__ == "__main__":
    main()
