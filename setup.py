"""
Partially adopted from:
https://github.com/himbeles/ctypes-example
"""

import distutils.command.build
import sys
from distutils.command.build_ext import build_ext as build_ext_orig
from pathlib import Path

from setuptools import Extension, find_packages, setup

if not ((3, 8) <= sys.version_info[:2] <= (3, 10)):
    sys.exit("Rocket-FFT requires Python 3.8 to 3.10")

    
def numpy_get_include():
    import numpy as np
    return np.get_include()


def numba_get_include():
    import numba as nb
    return Path(nb.__file__).parent


class CTypesExtension(Extension):
    pass


class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"

        return super().get_ext_filename(ext_name)


class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "rocket_fft"


with open("README.md") as f:
    long_description = f.read()


setup(
    name="rocket-fft",
    version="0.0.1",
    description="rocket-fft extends Numba by scipy.fft and numpy.fft",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Styfen Schär",
    author_email="styfen.schaer.blog@gmail.com",
    url="https://github.com/styfenschaer/rocket-fft",
    download_url="https://github.com/styfenschaer/rocket-fft",
    packages=find_packages(),
    entry_points={
        "numba_extensions": [
            "init = rocket_fft:_init_extension",
        ],
    },
    install_requires=["scipy", "numba", "numpy"],
    license="BSD",
    ext_modules=[
        CTypesExtension(
            "rocket_fft/_pocketfft_numba",
            sources=["rocket_fft/_pocketfft_numba.cpp"],
            extra_compile_args=["-std=c++11"],
        ),
    ],
    include_dirs=[
        numpy_get_include(),
        numba_get_include(),
    ],
    cmdclass={
        "build_ext": build_ext,
        "build": BuildCommand,
    },
)
