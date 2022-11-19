import distutils.command.build
import sys
#  https://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
from distutils.command.build_ext import build_ext as build_ext_orig
from pathlib import Path

import numba as nb
import numpy as np
from setuptools import Extension, find_packages, setup


def numba_get_include():
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
        if not self._ctypes:
            return super().get_ext_filename(ext_name)

        if sys.platform in ['win32', 'cygwin']:
            ext_name += '.dll'
        elif sys.platform in ['aix', 'linux', 'darwin']:
            ext_name += '.so'
        return ext_name


class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = ''


with open('README.md') as f:
    long_description = f.read()


setup(
    name='rocket-fft',
    version='1.0',
    description='rocket-fft extends Numba by scipy.fft and np.fft',
    long_description=long_description,
    author='Styfen Schär',
    author_email='styfen.schaer.blog@gmail.com',
    url='https://github.com/styfenschaer/rocket-fft',
    download_url='https://github.com/styfenschaer/rocket-fft',
    packages=find_packages(),
    entry_points={
        'numba_extensions': [
            'init = rocket_fft:_init_extension',
        ],
    },
    install_requires=['scipy', 'numba', 'numpy'],
    license='BSD',
    ext_modules=[
        CTypesExtension(
            'rocket_fft/_pocketfft',
            sources=['rocket_fft/_pocketfft_internal.cpp'],
        ),
    ],
    include_dirs=[
        np.get_include(),
        numba_get_include(),
    ],
    cmdclass={
        'build_ext': build_ext,
        'build': BuildCommand,
    },
)
