import platform
import re
import sys
from distutils.ccompiler import new_compiler
from distutils.command.build_ext import build_ext as build_ext_distutils
from distutils.errors import CompileError
from pathlib import Path
from tempfile import NamedTemporaryFile

from setuptools import Extension, find_packages, setup

py_versions_supported = ("3.8", "3.9", "3.10", "3.11")


py_version = "{}.{}".format(*sys.version_info[:2])
if py_version not in py_versions_supported:
    sys.exit(f"Unsupported Python version {py_version};"
             f" must be one of {py_versions_supported}")


def get_version(rel_path):
    with open(Path(__file__).parent / rel_path) as file:
        return re.search(r'__version__ = "(.*?)"', file.read())[1]


def numpy_get_include():
    import numpy as np
    return np.get_include()


def numba_get_include():
    import numba as nb
    return Path(nb.__file__).parent


def pthread_available():
    with NamedTemporaryFile(mode="w", suffix=".cpp") as file:
        file.write("#include <pthread.h>\nint main(){return 0;}")
        try:
            new_compiler().compile([file.name])
            return True
        except CompileError:
            return False


class build_ext(build_ext_distutils):
    def get_export_symbols(self, ext):
        return ext.export_symbols


with open("README.md") as file:
    long_description = file.read()


define_macros = [
    ("POCKETFFT_NO_SANITYCHECK", None),
    ("POCKETFFT_CACHE_SIZE", "16"),
]
if pthread_available():
    define_macros.append(("POCKETFFT_PTHREADS", None))

if platform.system() == "Windows":
    extra_compile_args = ["/Ox", "/Wall"]
else:
    extra_compile_args = ["-std=c++11", "-O3", "-Wall"]


setup(
    name="rocket-fft",
    version=get_version("rocket_fft/_version.py"),
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
    install_requires=["numba>=0.56.0"],
    license="BSD",
    ext_modules=[
        Extension(
            "rocket_fft/_pocketfft_numba",
            sources=["rocket_fft/_pocketfft_numba.cpp"],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "rocket_fft/_special_helpers",
            sources=["rocket_fft/_special_helpers.cpp"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    include_dirs=[
        numpy_get_include(),
        numba_get_include(),
    ],
    cmdclass={
        "build_ext": build_ext,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
        "dev": ["scipy>=1.7.2", "pytest>=6.2.5"]
    }
)