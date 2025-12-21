import os
from pathlib import Path

import pytest
import scipy.fft
import numpy as np
import numba as nb


def set_numba_capture_errors_new_style():
    # See: https://numba.readthedocs.io/en/latest/reference/
    # deprecation.html#deprecation-of-old-style-numba-captured-errors
    os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"


@pytest.fixture(autouse=True)
def numba_cache_cleanup():
    cache_dir = Path(__file__).parent / "__pycache__"

    for file in os.listdir(cache_dir):
        path = cache_dir / file
        if path.suffix in (".nbc", ".nbi"):
            try:
                os.remove(path)
            except (FileNotFoundError, PermissionError):
                pass


# -----------------------------------------------------------------------------
# Jitted functions
# -----------------------------------------------------------------------------


class NumpyFFT:
    @staticmethod
    @nb.njit
    def fft(a, n=None, axis=-1, norm=None, out=None):
        return np.fft.fft(a, n, axis, norm, out)

    @staticmethod
    @nb.njit
    def ifft(a, n=None, axis=-1, norm=None, out=None):
        return np.fft.ifft(a, n, axis, norm, out)

    @staticmethod
    @nb.njit
    def fft2(x, s=None, axes=(-2, -1), norm=None, out=None):
        return np.fft.fft2(x, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def ifft2(x, s=None, axes=(-2, -1), norm=None, out=None):
        return np.fft.ifft2(x, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def fftn(x, s=None, axes=None, norm=None, out=None):
        return np.fft.fftn(x, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def ifftn(x, s=None, axes=None, norm=None, out=None):
        return np.fft.ifftn(x, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def rfft(a, n=None, axis=-1, norm=None, out=None):
        return np.fft.rfft(a, n, axis, norm, out)

    @staticmethod
    @nb.njit
    def irfft(a, n=None, axis=-1, norm=None, out=None):
        return np.fft.irfft(a, n, axis, norm, out)

    @staticmethod
    @nb.njit
    def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
        return np.fft.rfft2(a, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
        return np.fft.irfft2(a, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def rfftn(a, s=None, axes=None, norm=None, out=None):
        return np.fft.rfftn(a, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def irfftn(a, s=None, axes=None, norm=None, out=None):
        return np.fft.irfftn(a, s, axes, norm, out)

    @staticmethod
    @nb.njit
    def hfft(a, n=None, axis=-1, norm=None, out=None):
        return np.fft.hfft(a, n, axis, norm, out)

    @staticmethod
    @nb.njit
    def ihfft(a, n=None, axis=-1, norm=None, out=None):
        return np.fft.ihfft(a, n, axis, norm, out)

    @staticmethod
    @nb.njit
    def fftfreq(n, d=1.0, device=None):
        return np.fft.fftfreq(n, d, device)

    @staticmethod
    @nb.njit
    def rfftfreq(n, d=1.0, device=None):
        return np.fft.rfftfreq(n, d, device)

    @staticmethod
    @nb.njit
    def fftshift(x, axes=None):
        return np.fft.fftshift(x, axes)

    @staticmethod
    @nb.njit
    def ifftshift(x, axes=None):
        return np.fft.ifftshift(x, axes)


class ScipyFFT:
    @staticmethod
    @nb.njit
    def fft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.fft(
            x,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def ifft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.ifft(
            x,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def fft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.fft2(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def ifft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.ifft2(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def fftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.fftn(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def ifftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.ifftn(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def rfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.rfft(
            x,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def irfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.irfft(
            x,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def rfft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.rfft2(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def irfft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.irfft2(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def rfftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.rfftn(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def irfftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.irfftn(
            x,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def hfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.hfft(
            x,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def ihfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        return scipy.fft.ihfft(
            x,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            plan=plan,
        )

    @staticmethod
    @nb.njit
    def fftfreq(n, d=1.0, xp=None, device=None):
        return scipy.fft.fftfreq(n, d, xp=xp, device=device)

    @staticmethod
    @nb.njit
    def rfftfreq(n, d=1.0, xp=None, device=None):
        return scipy.fft.rfftfreq(n, d, xp=xp, device=device)

    @staticmethod
    @nb.njit
    def fftshift(x, axes=None):
        return scipy.fft.fftshift(x, axes)

    @staticmethod
    @nb.njit
    def ifftshift(x, axes=None):
        return scipy.fft.ifftshift(x, axes)

    @staticmethod
    @nb.njit
    def next_fast_len(n, real=False):
        return scipy.fft.next_fast_len(n, real)

    @staticmethod
    @nb.njit
    def dct(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.dct(
            x,
            type,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def idct(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.idct(
            x,
            type,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def dctn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.dctn(
            x,
            type,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def idctn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.idctn(
            x,
            type,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def dst(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.dst(
            x,
            type,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def idst(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.idst(
            x,
            type,
            n,
            axis,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def dstn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.dstn(
            x,
            type,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def idstn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        return scipy.fft.idstn(
            x,
            type,
            s,
            axes,
            norm,
            overwrite_x,
            workers,
            orthogonalize,
        )

    @staticmethod
    @nb.njit
    def fht(x, dln, mu, offset=0.0, bias=0.0):
        return scipy.fft.fht(x, dln, mu, offset, bias)

    @staticmethod
    @nb.njit
    def ifht(x, dln, mu, offset=0.0, bias=0.0):
        return scipy.fft.ifht(x, dln, mu, offset, bias)

    @staticmethod
    @nb.njit
    def fhtoffset(dln, mu, initial=0.0, bias=0.0):
        return scipy.fft.fhtoffset(dln, mu, initial, bias)
