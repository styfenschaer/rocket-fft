import platform
from pathlib import Path
from tempfile import NamedTemporaryFile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError


def numpy_get_include():
    import numpy as np
    return np.get_include()


def numba_get_include():
    import numba as nb
    return str(Path(nb.__file__).parent)


class build_ext_with_pthreads(build_ext):
    def build_extensions(self):
        if platform.system() != "Windows" and self._pthread_available():
            for ext in self.extensions:
                ext.define_macros.append(("POCKETFFT_PTHREADS", None))
        super().build_extensions()

    def _pthread_available(self):
        with NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write("#include <pthread.h>\nint main(void){return 0;}")
            fname = f.name
        try:
            self.compiler.compile([fname])
            return True
        except CompileError:
            return False


define_macros = [
    ("DUCC0_USE_NANOBIND", None),
]

if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17", "/Ox", "/W3"]
else:
    extra_compile_args = ["-std=c++17", "-O3", "-Wall", "-pthread"]

include_dirs = [numpy_get_include(), numba_get_include(), "rocket_fft"]

ext_modules = [
    Extension(
        "rocket_fft._pocketfft_numba",
        sources=[
            "rocket_fft/_pocketfft_numba.cpp",
            "rocket_fft/ducc0/fft/fft_inst1.cc",
            "rocket_fft/ducc0/fft/fft_inst2.cc",
            "rocket_fft/ducc0/infra/threading.cc",
            "rocket_fft/ducc0/infra/mav.cc",
            "rocket_fft/ducc0/infra/string_utils.cc",
        ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    ),
    Extension(
        "rocket_fft._special_helpers",
        sources=["rocket_fft/_special_helpers.cpp"],
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_with_pthreads},
)
