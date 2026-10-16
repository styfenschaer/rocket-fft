import os
import platform
import re

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# The extension only uses CPython's Stable ABI, so a single abi3 wheel per platform
# serves every Python version. Keep in sync with `requires-python` in pyproject.toml.
LIMITED_API_TAG = "cp310"
LIMITED_API_HEX = "0x030A0000"

SOURCES = [
    "rocket_fft/_rocket_fft_numba.cpp",
    "rocket_fft/ducc0/fft/fft_inst1.cc",
    "rocket_fft/ducc0/fft/fft_inst2.cc",
    "rocket_fft/ducc0/infra/threading.cc",
    "rocket_fft/ducc0/infra/mav.cc",
    "rocket_fft/ducc0/infra/string_utils.cc",
]

if platform.system() == "Windows":
    compile_args = ["/std:c++17", "/O2", "/W3", "/EHsc"]
    avx2_args = ["/arch:AVX2"]  # implies FMA
else:
    # -g0 drops the debug info that Python's default CFLAGS would add (~10x smaller binary).
    compile_args = ["-std=c++17", "-O3", "-g0", "-Wall", "-pthread"]
    # No fused multiply-add contraction: it would change results in the last bits
    # (away from SciPy's and the generic build's) for a tested speedup of only ~0-2%.
    # MSVC does not contract by default.
    avx2_args = ["-mavx2", "-mfma", "-ffp-contract=off"]


def extension(name, extra_args=()):
    return Extension(
        f"rocket_fft.{name}",
        sources=SOURCES,
        define_macros=[
            # Skips DUCC0's long double instantiations, which rocket-fft never uses.
            ("DUCC0_USE_NANOBIND", None),
            ("Py_LIMITED_API", LIMITED_API_HEX),
            ("RF_MODULE_NAME", name),
        ],
        extra_compile_args=compile_args + list(extra_args),
        include_dirs=["rocket_fft"],
        py_limited_api=True,
    )


# The generic build targets the platform's baseline instruction set. On x86-64,
# a second build uses AVX2 and FMA (DUCC0 picks its SIMD width at compile time,
# which makes mid-sized and large transforms up to ~2x faster); the faster one
# the CPU supports is chosen at import time (rocket_fft/_native.py).
ext_modules = [extension("_rocket_fft_numba")]
if platform.machine().lower() in ("x86_64", "amd64"):
    ext_modules.append(extension("_rocket_fft_numba_avx2", avx2_args))


class build_ext_abi3(build_ext):
    def build_extension(self, ext):
        # Both builds compile the same sources with different flags, so each gets
        # its own directory for object files (also safe for parallel builds).
        build_temp = self.build_temp
        self.build_temp = os.path.join(build_temp, ext.name)
        try:
            super().build_extension(ext)
        finally:
            self.build_temp = build_temp

    def get_ext_filename(self, fullname):
        # Python 3.15 names abi3 extensions "*.abi3-<platform>.so", a suffix that
        # Python 3.10-3.14 do not recognize, so a wheel built with 3.15 would only
        # work on 3.15+. All versions accept the plain ".abi3.so".
        return re.sub(r"\.abi3-[^.]+\.so$", ".abi3.so", super().get_ext_filename(fullname))


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_abi3},
    options={"bdist_wheel": {"py_limited_api": LIMITED_API_TAG}},
)
