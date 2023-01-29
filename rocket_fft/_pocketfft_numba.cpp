#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

#include "_pocketfft_hdronly.h"

#include "_arraystruct.h"
#include "./core/runtime/nrt_external.h"

#ifdef _MSC_VER
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

using pocketfft::shape_t;
using pocketfft::stride_t;

#define SIZEOF_COMPLEX128 16
#define SIZEOF_DOUBLE 8
// Patch for Python 3.10 
#define ssize_t Py_ssize_t 

static shape_t
copy_shape(const arystruct_t *arystruct, ssize_t ndim)
{
    shape_t res(ndim);
    for (auto i = 0; i < res.size(); i++)
        res[i] = arystruct->shape_and_strides[i];
    return res;
}

static stride_t
copy_strides(const arystruct_t *arystruct, ssize_t ndim)
{
    stride_t res(ndim);
    for (auto i = 0; i < res.size(); i++)
        res[i] = arystruct->shape_and_strides[ndim + i];
    return res;
}

static shape_t
copy_array(const arystruct_t *arystruct)
{
    auto data = reinterpret_cast<size_t *>(arystruct->data);
    shape_t res(arystruct->nitems);
    for (auto i = 0; i < res.size(); i++)
        res[i] = data[i];
    return res;
}

DLL_EXPORT size_t
numba_good_size(size_t target, bool real)
{
    using pocketfft::detail::util;
    return real ? util::good_size_real(target)
                : util::good_size_cmplx(target);
}

DLL_EXPORT void
numba_c2c(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
          bool forward, double fct, size_t nthreads = 1)
{
    auto shape = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_COMPLEX128)
    {
        auto data_in = reinterpret_cast<std::complex<double> *>(ain->data);
        auto data_out = reinterpret_cast<std::complex<double> *>(aout->data);
        pocketfft::c2c<double>(shape, stride_in, stride_out, axes_, forward,
                               data_in, data_out, fct, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<std::complex<float> *>(ain->data);
        auto data_out = reinterpret_cast<std::complex<float> *>(aout->data);
        pocketfft::c2c<float>(shape, stride_in, stride_out, axes_, forward,
                              data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_dct(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
          int64_t type, double fct, bool ortho, size_t nthreads = 1)
{
    auto shape = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<double *>(aout->data);
        pocketfft::dct<double>(shape, stride_in, stride_out, axes_,
                               type, data_in, data_out, fct, ortho, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<float *>(aout->data);
        pocketfft::dct<float>(shape, stride_in, stride_out, axes_,
                              type, data_in, data_out, fct, ortho, nthreads);
    }
}

DLL_EXPORT void
numba_dst(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
          int64_t type, double fct, bool ortho, size_t nthreads = 1)
{
    auto shape = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<double *>(aout->data);
        pocketfft::dst<double>(shape, stride_in, stride_out, axes_,
                               type, data_in, data_out, fct, ortho, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<float *>(aout->data);
        pocketfft::dst<float>(shape, stride_in, stride_out, axes_,
                              type, data_in, data_out, fct, ortho, nthreads);
    }
}

DLL_EXPORT void
numba_r2c(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
          bool forward, double fct, size_t nthreads = 1)
{
    auto shape_in = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<std::complex<double> *>(aout->data);
        pocketfft::r2c<double>(shape_in, stride_in, stride_out, axes_,
                               forward, data_in, data_out, fct, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<std::complex<float> *>(aout->data);
        pocketfft::r2c<float>(shape_in, stride_in, stride_out, axes_,
                              forward, data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_c2c_sym(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
              bool forward, double fct, size_t nthreads = 1)
{
    using namespace pocketfft::detail;
    auto shape_in = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<std::complex<double> *>(aout->data);
        pocketfft::r2c<double>(shape_in, stride_in, stride_out, axes_,
                               forward, data_in, data_out, fct, nthreads);
        ndarr<std::complex<double>> ares(data_out, shape_in, stride_out);
        rev_iter iter(ares, axes_);
        while (iter.remaining() > 0)
        {
            auto v = ares[iter.ofs()];
            ares[iter.rev_ofs()] = conj(v);
            iter.advance();
        }
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<std::complex<float> *>(aout->data);
        pocketfft::r2c<float>(shape_in, stride_in, stride_out, axes_,
                              forward, data_in, data_out, fct, nthreads);
        ndarr<std::complex<float>> ares(data_out, shape_in, stride_out);
        rev_iter iter(ares, axes_);
        while (iter.remaining() > 0)
        {
            auto v = ares[iter.ofs()];
            ares[iter.rev_ofs()] = conj(v);
            iter.advance();
        }
    }
}

DLL_EXPORT void
numba_c2r(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
          bool forward, double fct, size_t nthreads = 1)
{
    auto stride_in = copy_strides(ain, ndim);
    auto shape_out = copy_shape(aout, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_COMPLEX128)
    {
        auto data_in = reinterpret_cast<std::complex<double> *>(ain->data);
        auto data_out = reinterpret_cast<double *>(aout->data);
        pocketfft::c2r<double>(shape_out, stride_in, stride_out, axes_,
                               forward, data_in, data_out, fct, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<std::complex<float> *>(ain->data);
        auto data_out = reinterpret_cast<float *>(aout->data);
        pocketfft::c2r<float>(shape_out, stride_in, stride_out, axes_,
                              forward, data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_r2r_fftpack(size_t ndim, const arystruct_t *ain, arystruct_t *aout, arystruct_t *axes,
              bool real2hermitian, bool forward, double fct, size_t nthreads = 1)
{
    auto shape = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<double *>(aout->data);
        pocketfft::r2r_fftpack<double>(shape, stride_in, stride_out, axes_, real2hermitian,
                                       forward, data_in, data_out, fct, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<float *>(aout->data);
        pocketfft::r2r_fftpack<float>(shape, stride_in, stride_out, axes_, real2hermitian,
                                      forward, data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_r2r_separable_hartley(size_t ndim, const arystruct_t *ain, arystruct_t *aout,
                        arystruct_t *axes, double fct, size_t nthreads = 1)
{
    auto shape = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<double *>(aout->data);
        pocketfft::r2r_separable_hartley<double>(shape, stride_in, stride_out, axes_,
                                                 data_in, data_out, fct, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<float *>(aout->data);
        pocketfft::r2r_separable_hartley<float>(shape, stride_in, stride_out, axes_,
                                                data_in, data_out, fct, nthreads);
    }
}

DLL_EXPORT void
numba_r2r_genuine_hartley(size_t ndim, const arystruct_t *ain, arystruct_t *aout,
                      arystruct_t *axes, double fct, size_t nthreads = 1)
{
    auto shape = copy_shape(ain, ndim);
    auto stride_in = copy_strides(ain, ndim);
    auto stride_out = copy_strides(aout, ndim);
    auto axes_ = copy_array(axes);
    if (ain->itemsize == SIZEOF_DOUBLE)
    {
        auto data_in = reinterpret_cast<double *>(ain->data);
        auto data_out = reinterpret_cast<double *>(aout->data);
        pocketfft::r2r_genuine_hartley<double>(shape, stride_in, stride_out, axes_,
                                               data_in, data_out, fct, nthreads);
    }
    else
    {
        auto data_in = reinterpret_cast<float *>(ain->data);
        auto data_out = reinterpret_cast<float *>(aout->data);
        pocketfft::r2r_genuine_hartley<float>(shape, stride_in, stride_out, axes_,
                                              data_in, data_out, fct, nthreads);
    }
}