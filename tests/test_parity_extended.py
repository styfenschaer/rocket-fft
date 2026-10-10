import numpy as np
import pytest
import scipy.fft

from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    NumpyFFT,
    ScipyFFT,
)
from rocket_fft.overloads import set_workers

set_numba_capture_errors_new_style()


# =============================================================================
# Helpers
# =============================================================================

def assert_parity(nb_out, ref_out, rtol=1e-12, atol=1e-12):
    assert nb_out.dtype == ref_out.dtype, f"Dtype mismatch: {nb_out.dtype} vs {ref_out.dtype}"
    assert nb_out.shape == ref_out.shape, f"Shape mismatch: {nb_out.shape} vs {ref_out.shape}"
    assert np.allclose(nb_out, ref_out, rtol=rtol, atol=atol)


# =============================================================================
# 1. Single Precision Parity (float32, complex64)
# =============================================================================

class TestSinglePrecision:
    rtol = 1e-4
    atol = 1e-4

    def test_fft_ifft_1d(self):
        rng = np.random.default_rng(100)
        x = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)

        for norm in (None, "backward", "ortho", "forward"):
            # Scipy
            assert_parity(
                ScipyFFT.fft(x, norm=norm),
                scipy.fft.fft(x, norm=norm),
                rtol=self.rtol,
                atol=self.atol,
            )
            assert_parity(
                ScipyFFT.ifft(x, norm=norm),
                scipy.fft.ifft(x, norm=norm),
                rtol=self.rtol,
                atol=self.atol,
            )
            # Numpy
            assert_parity(
                NumpyFFT.fft(x, norm=norm),
                np.fft.fft(x, norm=norm),
                rtol=self.rtol,
                atol=self.atol,
            )
            assert_parity(
                NumpyFFT.ifft(x, norm=norm),
                np.fft.ifft(x, norm=norm),
                rtol=self.rtol,
                atol=self.atol,
            )

    def test_rfft_irfft_1d(self):
        rng = np.random.default_rng(101)
        x_real = rng.standard_normal(32).astype(np.float32)

        for norm in (None, "backward", "ortho", "forward"):
            # Scipy
            r_nb = ScipyFFT.rfft(x_real, norm=norm)
            r_scipy = scipy.fft.rfft(x_real, norm=norm)
            assert_parity(r_nb, r_scipy, rtol=self.rtol, atol=self.atol)
            assert r_nb.dtype == np.complex64

            inv_nb = ScipyFFT.irfft(r_nb, norm=norm)
            inv_scipy = scipy.fft.irfft(r_scipy, norm=norm)
            assert_parity(inv_nb, inv_scipy, rtol=self.rtol, atol=self.atol)
            assert inv_nb.dtype == np.float32

            # Numpy
            r_np_nb = NumpyFFT.rfft(x_real, norm=norm)
            r_np = np.fft.rfft(x_real, norm=norm)
            assert_parity(r_np_nb, r_np, rtol=self.rtol, atol=self.atol)

            inv_np_nb = NumpyFFT.irfft(r_np_nb, norm=norm)
            inv_np = np.fft.irfft(r_np, norm=norm)
            assert_parity(inv_np_nb, inv_np, rtol=self.rtol, atol=self.atol)

    def test_hfft_ihfft_1d(self):
        rng = np.random.default_rng(102)
        x_c = (rng.standard_normal(17) + 1j * rng.standard_normal(17)).astype(np.complex64)
        x_real = rng.standard_normal(32).astype(np.float32)

        # Scipy
        assert_parity(
            ScipyFFT.hfft(x_c),
            scipy.fft.hfft(x_c),
            rtol=self.rtol,
            atol=self.atol,
        )
        assert_parity(
            ScipyFFT.ihfft(x_real),
            scipy.fft.ihfft(x_real),
            rtol=self.rtol,
            atol=self.atol,
        )

        # Numpy
        assert_parity(
            NumpyFFT.hfft(x_c),
            np.fft.hfft(x_c),
            rtol=self.rtol,
            atol=self.atol,
        )
        assert_parity(
            NumpyFFT.ihfft(x_real),
            np.fft.ihfft(x_real),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_dct_dst_1d(self):
        rng = np.random.default_rng(103)
        x = rng.standard_normal(32).astype(np.float32)

        for type_ in (1, 2, 3, 4):
            for norm in (None, "backward", "ortho", "forward"):
                for ortho in (None, False, True):
                    # DCT
                    r_nb = ScipyFFT.dct(x, type=type_, norm=norm, orthogonalize=ortho)
                    r_sp = scipy.fft.dct(x, type=type_, norm=norm, orthogonalize=ortho)
                    assert_parity(r_nb, r_sp, rtol=self.rtol, atol=self.atol)
                    assert r_nb.dtype == np.float32

                    inv_nb = ScipyFFT.idct(r_nb, type=type_, norm=norm, orthogonalize=ortho)
                    inv_sp = scipy.fft.idct(r_sp, type=type_, norm=norm, orthogonalize=ortho)
                    assert_parity(inv_nb, inv_sp, rtol=self.rtol, atol=self.atol)

                    # DST
                    r_dst_nb = ScipyFFT.dst(x, type=type_, norm=norm, orthogonalize=ortho)
                    r_dst_sp = scipy.fft.dst(x, type=type_, norm=norm, orthogonalize=ortho)
                    assert_parity(r_dst_nb, r_dst_sp, rtol=self.rtol, atol=self.atol)
                    assert r_dst_nb.dtype == np.float32

                    inv_dst_nb = ScipyFFT.idst(r_dst_nb, type=type_, norm=norm, orthogonalize=ortho)
                    inv_dst_sp = scipy.fft.idst(r_dst_sp, type=type_, norm=norm, orthogonalize=ortho)
                    assert_parity(inv_dst_nb, inv_dst_sp, rtol=self.rtol, atol=self.atol)

    def test_2d_nd_transforms(self):
        rng = np.random.default_rng(104)
        x_c = (rng.standard_normal((8, 6, 4)) + 1j * rng.standard_normal((8, 6, 4))).astype(np.complex64)
        x_r = rng.standard_normal((8, 6, 4)).astype(np.float32)

        # 2D & ND complex
        assert_parity(ScipyFFT.fft2(x_c), scipy.fft.fft2(x_c), rtol=self.rtol, atol=self.atol)
        assert_parity(ScipyFFT.ifft2(x_c), scipy.fft.ifft2(x_c), rtol=self.rtol, atol=self.atol)
        assert_parity(ScipyFFT.fftn(x_c), scipy.fft.fftn(x_c), rtol=self.rtol, atol=self.atol)
        assert_parity(ScipyFFT.ifftn(x_c), scipy.fft.ifftn(x_c), rtol=self.rtol, atol=self.atol)

        # 2D & ND real
        assert_parity(ScipyFFT.rfft2(x_r), scipy.fft.rfft2(x_r), rtol=self.rtol, atol=self.atol)
        r2 = ScipyFFT.rfft2(x_r)
        assert_parity(ScipyFFT.irfft2(r2, s=(8, 6)), scipy.fft.irfft2(r2, s=(8, 6)), rtol=self.rtol, atol=self.atol)

        assert_parity(ScipyFFT.rfftn(x_r), scipy.fft.rfftn(x_r), rtol=self.rtol, atol=self.atol)
        rn = ScipyFFT.rfftn(x_r)
        assert_parity(ScipyFFT.irfftn(rn, s=(8, 6, 4)), scipy.fft.irfftn(rn, s=(8, 6, 4)), rtol=self.rtol, atol=self.atol)

        # 2D & ND DCT / DST
        assert_parity(ScipyFFT.dctn(x_r), scipy.fft.dctn(x_r), rtol=self.rtol, atol=self.atol)
        assert_parity(ScipyFFT.dstn(x_r), scipy.fft.dstn(x_r), rtol=self.rtol, atol=self.atol)

    def test_fht_single(self):
        rng = np.random.default_rng(105)
        x = rng.standard_normal(32).astype(np.float32)
        dln = np.float32(0.1)
        mu = np.float32(0.5)

        r_nb = ScipyFFT.fht(x, dln, mu)
        r_sp = scipy.fft.fht(x, dln, mu)
        assert_parity(r_nb, r_sp, rtol=self.rtol, atol=self.atol)
        assert r_nb.dtype == np.float32

        inv_nb = ScipyFFT.ifht(r_nb, dln, mu)
        inv_sp = scipy.fft.ifht(r_sp, dln, mu)
        assert_parity(inv_nb, inv_sp, rtol=self.rtol, atol=self.atol)


# =============================================================================
# 2. Array Memory Layouts & Striding (Fortran, Sliced, Reversed)
# =============================================================================

class TestMemoryLayouts:
    def test_fortran_order(self):
        rng = np.random.default_rng(200)
        a_f = np.asfortranarray(rng.standard_normal((12, 16)) + 1j * rng.standard_normal((12, 16)))
        assert a_f.flags.f_contiguous
        assert not a_f.flags.c_contiguous

        assert_parity(ScipyFFT.fft2(a_f), scipy.fft.fft2(a_f))
        assert_parity(ScipyFFT.ifft2(a_f), scipy.fft.ifft2(a_f))
        assert_parity(NumpyFFT.fft2(a_f), np.fft.fft2(a_f))
        assert_parity(NumpyFFT.ifft2(a_f), np.fft.ifft2(a_f))

        a_f_real = np.asfortranarray(rng.standard_normal((12, 16)))
        assert_parity(ScipyFFT.rfft2(a_f_real), scipy.fft.rfft2(a_f_real))
        assert_parity(ScipyFFT.dctn(a_f_real), scipy.fft.dctn(a_f_real))

    def test_strided_sliced_arrays(self):
        rng = np.random.default_rng(201)
        big = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        sliced = big[::2, ::3]  # (16, 11) non-contiguous strided
        assert not sliced.flags.c_contiguous
        assert not sliced.flags.f_contiguous

        assert_parity(ScipyFFT.fft2(sliced), scipy.fft.fft2(sliced))
        assert_parity(ScipyFFT.ifft2(sliced), scipy.fft.ifft2(sliced))
        assert_parity(NumpyFFT.fft2(sliced), np.fft.fft2(sliced))
        assert_parity(NumpyFFT.ifft2(sliced), np.fft.ifft2(sliced))

        big_r = rng.standard_normal((32, 32))
        sliced_r = big_r[1::2, ::4]
        assert_parity(ScipyFFT.rfft2(sliced_r), scipy.fft.rfft2(sliced_r))
        assert_parity(ScipyFFT.dctn(sliced_r), scipy.fft.dctn(sliced_r))

    def test_reversed_strides(self):
        rng = np.random.default_rng(202)
        x = (rng.standard_normal(32) + 1j * rng.standard_normal(32))[::-1]
        assert not x.flags.c_contiguous

        assert_parity(ScipyFFT.fft(x), scipy.fft.fft(x))
        assert_parity(NumpyFFT.fft(x), np.fft.fft(x))


# =============================================================================
# 3. Integer and Boolean Promotion
# =============================================================================

class TestIntegerPromotion:
    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.int16, np.uint8, np.bool_])
    def test_fft_integer_promotion(self, dtype):
        if dtype == np.bool_:
            x = (np.arange(16) % 2).astype(np.bool_)
        else:
            x = np.arange(16, dtype=dtype)

        res_sp_nb = ScipyFFT.fft(x)
        res_sp = scipy.fft.fft(x)
        assert_parity(res_sp_nb, res_sp)
        assert res_sp_nb.dtype == np.complex128

        res_np_nb = NumpyFFT.fft(x)
        res_np = np.fft.fft(x)
        assert_parity(res_np_nb, res_np)
        assert res_np_nb.dtype == np.complex128

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.int16, np.uint8])
    def test_rfft_integer_promotion(self, dtype):
        x = np.arange(16, dtype=dtype)

        res_sp_nb = ScipyFFT.rfft(x)
        res_sp = scipy.fft.rfft(x)
        assert_parity(res_sp_nb, res_sp)
        assert res_sp_nb.dtype == np.complex128

        res_np_nb = NumpyFFT.rfft(x)
        res_np = np.fft.rfft(x)
        assert_parity(res_np_nb, res_np)
        assert res_np_nb.dtype == np.complex128

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.int16, np.uint8])
    def test_dct_integer_promotion(self, dtype):
        x = np.arange(16, dtype=dtype)

        res_sp_nb = ScipyFFT.dct(x)
        res_sp = scipy.fft.dct(x)
        assert_parity(res_sp_nb, res_sp)
        assert res_sp_nb.dtype == np.float64


# =============================================================================
# 4. Error and Exception Parity
# =============================================================================

class TestErrorParity:
    def test_invalid_n(self):
        x = np.ones(16, dtype=np.complex128)
        with pytest.raises(Exception):
            ScipyFFT.fft(x, n=0)
        with pytest.raises(Exception):
            scipy.fft.fft(x, n=0)

        with pytest.raises(Exception):
            ScipyFFT.fft(x, n=-1)
        with pytest.raises(Exception):
            scipy.fft.fft(x, n=-1)

        with pytest.raises(Exception):
            NumpyFFT.fft(x, n=0)
        with pytest.raises(Exception):
            np.fft.fft(x, n=0)

        with pytest.raises(Exception):
            NumpyFFT.fft(x, n=-5)
        with pytest.raises(Exception):
            np.fft.fft(x, n=-5)

    def test_invalid_norm(self):
        x = np.ones(16, dtype=np.complex128)
        with pytest.raises(Exception):
            ScipyFFT.fft(x, norm="invalid")
        with pytest.raises(Exception):
            scipy.fft.fft(x, norm="invalid")

        with pytest.raises(Exception):
            NumpyFFT.fft(x, norm="invalid")
        with pytest.raises(Exception):
            np.fft.fft(x, norm="invalid")

    def test_invalid_dct_dst_type(self):
        x = np.ones(16, dtype=np.float64)
        for t in (0, 5, -1):
            with pytest.raises(Exception):
                ScipyFFT.dct(x, type=t)
            with pytest.raises(Exception):
                scipy.fft.dct(x, type=t)

            with pytest.raises(Exception):
                ScipyFFT.dst(x, type=t)
            with pytest.raises(Exception):
                scipy.fft.dst(x, type=t)

    def test_axis_out_of_bounds(self):
        x = np.ones(16, dtype=np.complex128)
        with pytest.raises(Exception):
            ScipyFFT.fft(x, axis=5)
        with pytest.raises(Exception):
            scipy.fft.fft(x, axis=5)

        with pytest.raises(Exception):
            ScipyFFT.fft(x, axis=-5)
        with pytest.raises(Exception):
            scipy.fft.fft(x, axis=-5)

        with pytest.raises(Exception):
            NumpyFFT.fft(x, axis=5)
        with pytest.raises(Exception):
            np.fft.fft(x, axis=5)

    def test_workers_errors(self):
        x = np.ones(16, dtype=np.complex128)
        with pytest.raises(Exception):
            ScipyFFT.fft(x, workers=0)
        with pytest.raises(Exception):
            scipy.fft.fft(x, workers=0)

        with pytest.raises(ValueError):
            set_workers(0)
        with pytest.raises(ValueError):
            set_workers(-1)

    def test_fast_len_errors(self):
        with pytest.raises(Exception):
            ScipyFFT.next_fast_len(-1)
        with pytest.raises(Exception):
            scipy.fft.next_fast_len(-1)

        if hasattr(scipy.fft, "prev_fast_len"):
            with pytest.raises(Exception):
                ScipyFFT.prev_fast_len(-1)
            with pytest.raises(Exception):
                scipy.fft.prev_fast_len(-1)

    def test_numpy_out_errors(self):
        x = np.ones(16, dtype=np.complex128)
        out_wrong_shape = np.zeros(8, dtype=np.complex128)
        out_wrong_dtype = np.zeros(16, dtype=np.float64)

        with pytest.raises(Exception):
            NumpyFFT.fft(x, out=out_wrong_shape)
        with pytest.raises(Exception):
            NumpyFFT.fft(x, out=out_wrong_dtype)

        x_r = np.ones(16, dtype=np.float64)
        out_r_wrong_shape = np.zeros(5, dtype=np.complex128)
        out_r_wrong_dtype = np.zeros(9, dtype=np.float64)
        with pytest.raises(Exception):
            NumpyFFT.rfft(x_r, out=out_r_wrong_shape)
        with pytest.raises(Exception):
            NumpyFFT.rfft(x_r, out=out_r_wrong_dtype)


# =============================================================================
# 5. Empty / Zero-size Arrays
# =============================================================================

class TestEmptyArrays:
    def test_empty_1d(self):
        x = np.array([], dtype=np.complex128)
        assert NumpyFFT.fft(x).shape == (0,)
        assert ScipyFFT.fft(x).shape == (0,)

    def test_empty_2d(self):
        x = np.empty((0, 5), dtype=np.complex128)
        assert NumpyFFT.fft2(x).shape == (0, 5)
        assert ScipyFFT.fft2(x).shape == (0, 5)

        x_r = np.empty((4, 0), dtype=np.float64)
        assert NumpyFFT.rfft2(x_r).shape == (4, 1)
        assert ScipyFFT.rfft2(x_r).shape == (4, 1)
