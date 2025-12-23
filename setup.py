import platform
from pathlib import Path
from tempfile import NamedTemporaryFile

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


class build_ext_with_pthreads(build_ext):
    def finalize_options(self):
        super().finalize_options()

        import numpy
        import numba

        self.include_dirs.append(numpy.get_include())
        self.include_dirs.append(Path(numba.__file__).parent)

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
    ("POCKETFFT_NO_SANITYCHECK", None),
    ("POCKETFFT_CACHE_SIZE", "16"),
]

if platform.system() == "Windows":
    extra_compile_args = ["/Ox", "/W3"]
else:
    extra_compile_args = ["-std=c++11", "-O3", "-Wall"]

setup(
    name="rocket-fft",
    version="0.3.0",
    description="Rocket-FFT extends Numba by scipy.fft and numpy.fft",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Styfen Schär",
    author_email="styfen.schaer.blog@gmail.com",
    url="https://github.com/styfenschaer/rocket-fft",
    license="BSD-3-Clause",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    keywords=["FFT", "Fourier", "Numba", "SciPy", "NumPy"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"rocket_fft": ["*.pyi"]},
    python_requires=">=3.9",
    install_requires=["numba>=0.60.0"],
    extras_require={"dev": ["scipy>=1.13.1", "pytest>=8.4.2"]},
    entry_points={"numba_extensions": ["init = rocket_fft:_init_extension"]},
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
    cmdclass={"build_ext": build_ext_with_pthreads},
)
