import platform
from pathlib import Path
from tempfile import NamedTemporaryFile

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError


def numpy_get_include():
    import numpy as np

    return np.get_include()


def numba_get_include():
    import numba as nb

    return Path(nb.__file__).parent


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


with open("README.md") as file:
    long_description = file.read()


define_macros = [
    ("POCKETFFT_NO_SANITYCHECK", None),
    ("POCKETFFT_CACHE_SIZE", "16"),
]

if platform.system() == "Windows":
    extra_compile_args = ["/Ox", "/W3"]
else:
    extra_compile_args = ["-std=c++11", "-O3", "-Wall"]


setup(
    name="rocket-fft",
    description="Rocket-FFT extends Numba by scipy.fft and numpy.fft",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Styfen Schär",
    author_email="styfen.schaer.blog@gmail.com",
    url="https://github.com/styfenschaer/rocket-fft",
    download_url="https://github.com/styfenschaer/rocket-fft",
    packages=find_packages(),
    include_package_data=True,
    package_data={"rocket_fft": ["*.pyi"]},
    entry_points={
        "numba_extensions": [
            "init = rocket_fft:_init_extension",
        ],
    },
    install_requires=["numba>=0.60.0"],
    license="BSD",
    ext_modules=[
        Extension(
            "rocket_fft._pocketfft_numba",
            sources=["rocket_fft/_pocketfft_numba.cpp"],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "rocket_fft._special_helpers",
            sources=["rocket_fft/_special_helpers.cpp"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    include_dirs=[
        numpy_get_include(),
        numba_get_include(),
    ],
    cmdclass={
        "build_ext": build_ext_with_pthreads,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    keywords=["FFT", "Fourier", "Numba", "SciPy", "NumPy"],
    extras_require={
        "dev": ["scipy>=1.13.1", "pytest>=8.4.2"],
    },
)
