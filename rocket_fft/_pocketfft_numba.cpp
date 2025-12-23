#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "./core/runtime/nrt_external.h"
#include "_arraystruct.h"
#include "_pocketfft_hdronly.h"

#ifdef _MSC_VER
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

#define SIZEOF_COMPLEX128 16
#define SIZEOF_FLOAT64 8

using pocketfft::axes_view_t;
using pocketfft::shape_view_t;
using pocketfft::stride_view_t;
using pocketfft::detail::ndarr;
using pocketfft::detail::rev_iter;
using pocketfft::detail::util;

DLL_EXPORT uint64_t
numba_good_size(uint64_t target, bool real) {
    return real ? util::good_size_real(target)
                : util::good_size_cmplx(target);
}

DLL_EXPORT void
numba_c2c(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    shape_view_t shape(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_COMPLEX128) {
        auto data_in = reinterpret_cast<std::complex<double>*>(ain->data);
        auto data_out = reinterpret_cast<std::complex<double>*>(aout->data);
        pocketfft::c2c<double>(shape, stride_in, stride_out, axes_, forward,
            data_in, data_out, fct, nthreads);
    } else {
        auto data_in = reinterpret_cast<std::complex<float>*>(ain->data);
        auto data_out = reinterpret_cast<std::complex<float>*>(aout->data);
        pocketfft::c2c<float>(shape, stride_in, stride_out, axes_, forward,
            data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_dct(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    uint64_t type, double fct, bool ortho, uint64_t nthreads = 1) {
    shape_view_t shape(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<double*>(aout->data);
        pocketfft::dct<double>(shape, stride_in, stride_out, axes_,
            type, data_in, data_out, fct, ortho, nthreads);
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<float*>(aout->data);
        pocketfft::dct<float>(shape, stride_in, stride_out, axes_,
            type, data_in, data_out, fct, ortho, nthreads);
    }
}

DLL_EXPORT void
numba_dst(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    uint64_t type, double fct, bool ortho, uint64_t nthreads = 1) {
    shape_view_t shape(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<double*>(aout->data);
        pocketfft::dst<double>(shape, stride_in, stride_out, axes_,
            type, data_in, data_out, fct, ortho, nthreads);
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<float*>(aout->data);
        pocketfft::dst<float>(shape, stride_in, stride_out, axes_,
            type, data_in, data_out, fct, ortho, nthreads);
    }
}

DLL_EXPORT void
numba_r2c(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    shape_view_t shape_in(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<std::complex<double>*>(aout->data);
        pocketfft::r2c<double>(shape_in, stride_in, stride_out, axes_,
            forward, data_in, data_out, fct, nthreads);
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<std::complex<float>*>(aout->data);
        pocketfft::r2c<float>(shape_in, stride_in, stride_out, axes_,
            forward, data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_c2c_sym(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    shape_view_t shape_in(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<std::complex<double>*>(aout->data);
        pocketfft::r2c<double>(shape_in, stride_in, stride_out, axes_,
            forward, data_in, data_out, fct, nthreads);
        ndarr<std::complex<double>> ares(data_out, shape_in, stride_out);
        rev_iter iter(ares, axes_);
        while (iter.remaining() > 0) {
            auto v = ares[iter.ofs()];
            ares[iter.rev_ofs()] = conj(v);
            iter.advance();
        }
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<std::complex<float>*>(aout->data);
        pocketfft::r2c<float>(shape_in, stride_in, stride_out, axes_,
            forward, data_in, data_out, fct, nthreads);
        ndarr<std::complex<float>> ares(data_out, shape_in, stride_out);
        rev_iter iter(ares, axes_);
        while (iter.remaining() > 0) {
            auto v = ares[iter.ofs()];
            ares[iter.rev_ofs()] = conj(v);
            iter.advance();
        }
    }
}

DLL_EXPORT void
numba_c2r(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    shape_view_t shape_out(aout->shape_and_strides, ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_COMPLEX128) {
        auto data_in = reinterpret_cast<std::complex<double>*>(ain->data);
        auto data_out = reinterpret_cast<double*>(aout->data);
        pocketfft::c2r<double>(shape_out, stride_in, stride_out, axes_,
            forward, data_in, data_out, fct, nthreads);
    } else {
        auto data_in = reinterpret_cast<std::complex<float>*>(ain->data);
        auto data_out = reinterpret_cast<float*>(aout->data);
        pocketfft::c2r<float>(shape_out, stride_in, stride_out, axes_,
            forward, data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_r2r_fftpack(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool real2hermitian, bool forward, double fct, uint64_t nthreads = 1) {
    shape_view_t shape(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<double*>(aout->data);
        pocketfft::r2r_fftpack<double>(shape, stride_in, stride_out, axes_, real2hermitian,
            forward, data_in, data_out, fct, nthreads);
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<float*>(aout->data);
        pocketfft::r2r_fftpack<float>(shape, stride_in, stride_out, axes_, real2hermitian,
            forward, data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_r2r_separable_hartley(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout,
    arystruct_t* axes, double fct, uint64_t nthreads = 1) {
    shape_view_t shape(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<double*>(aout->data);
        pocketfft::r2r_separable_hartley<double>(shape, stride_in, stride_out, axes_,
            data_in, data_out, fct, nthreads);
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<float*>(aout->data);
        pocketfft::r2r_separable_hartley<float>(shape, stride_in, stride_out, axes_,
            data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_r2r_genuine_hartley(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout,
    arystruct_t* axes, double fct, uint64_t nthreads = 1) {
    shape_view_t shape(ain->shape_and_strides, ndim);
    stride_view_t stride_in(&ain->shape_and_strides[ndim], ndim);
    stride_view_t stride_out(&aout->shape_and_strides[ndim], ndim);
    axes_view_t axes_(axes->data, axes->nitems);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto data_in = reinterpret_cast<double*>(ain->data);
        auto data_out = reinterpret_cast<double*>(aout->data);
        pocketfft::r2r_genuine_hartley<double>(shape, stride_in, stride_out, axes_,
            data_in, data_out, fct, nthreads);
    } else {
        auto data_in = reinterpret_cast<float*>(ain->data);
        auto data_out = reinterpret_cast<float*>(aout->data);
        pocketfft::r2r_genuine_hartley<float>(shape, stride_in, stride_out, axes_,
            data_in, data_out, fct, nthreads);
    }
}

// ---- Required for setuptools (dummy Python module) ----

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pocketfft_numba",
    nullptr,
    -1,
    nullptr,
};

PyMODINIT_FUNC PyInit__pocketfft_numba(void) {
    return PyModule_Create(&moduledef);
}