#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <complex>
#include <vector>
#include <limits>
#include <stdexcept>

#include "./core/runtime/nrt_external.h"
#include "_arraystruct.h"
#include "ducc0/fft/fft.h"

#ifdef _MSC_VER
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

#define SIZEOF_COMPLEX128 16
#define SIZEOF_FLOAT64 8

template<typename T>
inline ducc0::cfmav<T> make_cfmav(uint64_t ndim, const arystruct_t* a) {
    ducc0::fmav_info::shape_t shp(ndim);
    ducc0::fmav_info::stride_t str(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        shp[i] = a->shape_and_strides[i];
        str[i] = a->shape_and_strides[ndim + i] / static_cast<ptrdiff_t>(sizeof(T));
    }
    return ducc0::cfmav<T>(reinterpret_cast<const T*>(a->data), shp, str);
}

template<typename T>
inline ducc0::vfmav<T> make_vfmav(uint64_t ndim, arystruct_t* a) {
    ducc0::fmav_info::shape_t shp(ndim);
    ducc0::fmav_info::stride_t str(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        shp[i] = a->shape_and_strides[i];
        str[i] = a->shape_and_strides[ndim + i] / static_cast<ptrdiff_t>(sizeof(T));
    }
    return ducc0::vfmav<T>(reinterpret_cast<T*>(a->data), shp, str);
}

inline ducc0::fmav_info::shape_t make_axes(const arystruct_t* axes) {
    ducc0::fmav_info::shape_t ax(axes->nitems);
    auto axes_ptr = reinterpret_cast<const uint64_t*>(axes->data);
    for (uint64_t i = 0; i < axes->nitems; ++i) {
        ax[i] = axes_ptr[i];
    }
    return ax;
}

inline bool has_duplicate_axes(uint64_t ndim, const ducc0::fmav_info::shape_t& ax) {
    if (ax.size() <= 1) return false;
    std::vector<bool> seen(ndim, false);
    for (auto a : ax) {
        if (a < ndim && seen[a]) return true;
        if (a < ndim) seen[a] = true;
    }
    return false;
}

DLL_EXPORT uint64_t
numba_good_size(uint64_t target, bool real) {
    return real ? ducc0::good_size_real(target)
                : ducc0::good_size_complex(target);
}

template <typename UIntT>
static UIntT prev_good_size_cmplx_typed(UIntT n) {
    if (n <= 12) return n;
    if (n > std::numeric_limits<UIntT>::max() / 11) {
        if (sizeof(UIntT) < sizeof(std::uint64_t)) {
            auto res = prev_good_size_cmplx_typed<std::uint64_t>(n);
            if (res <= std::numeric_limits<UIntT>::max())
                return static_cast<UIntT>(res);
        }
        throw std::runtime_error("FFT size is too large.");
    }

    UIntT bestfound = 1;
    for (UIntT f11 = 1; f11 <= n; f11 *= 11)
      for (UIntT f117 = f11; f117 <= n; f117 *= 7)
        for (UIntT f1175 = f117; f1175 <= n; f1175 *= 5) {
          UIntT x = f1175;
          while (x * 2 <= n) x *= 2;
          if (x > bestfound) bestfound = x;
          while (true) {
            if (x * 3 <= n) x *= 3;
            else if (x % 2 == 0) x /= 2;
            else break;

            if (x > bestfound) bestfound = x;
          }
        }
    return bestfound;
}

template <typename UIntT>
static UIntT prev_good_size_real_typed(UIntT n) {
    if (n <= 6) return n;
    if (n > std::numeric_limits<UIntT>::max() / 5) {
        if (sizeof(UIntT) < sizeof(std::uint64_t)) {
            auto res = prev_good_size_real_typed<std::uint64_t>(n);
            if (res <= std::numeric_limits<UIntT>::max())
                return static_cast<UIntT>(res);
        }
        throw std::runtime_error("FFT size is too large.");
    }

    UIntT bestfound = 1;
    for (UIntT f5 = 1; f5 <= n; f5 *= 5) {
      UIntT x = f5;
      while (x * 2 <= n) x *= 2;
      if (x > bestfound) bestfound = x;
      while (true) {
        if (x * 3 <= n) x *= 3;
        else if (x % 2 == 0) x /= 2;
        else break;

        if (x > bestfound) bestfound = x;
      }
    }
    return bestfound;
}

DLL_EXPORT uint64_t
numba_prev_good_size(uint64_t target, bool real) {
    return real ? prev_good_size_real_typed<uint64_t>(target)
                : prev_good_size_cmplx_typed<uint64_t>(target);
}

DLL_EXPORT void
numba_c2c(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    bool has_dup = has_duplicate_axes(ndim, ax);
    if (ain->itemsize == SIZEOF_COMPLEX128) {
        auto in_mav = make_cfmav<std::complex<double>>(ndim, ain);
        auto out_mav = make_vfmav<std::complex<double>>(ndim, aout);
        if (!has_dup) {
            ducc0::c2c<double>(in_mav, out_mav, ax, forward, fct, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::c2c<double>(in_mav, out_mav, single_ax, forward, fct, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<std::complex<double>>(ndim, aout);
                ducc0::c2c<double>(inout_mav, out_mav, single_ax, forward, 1.0, nthreads);
            }
        }
    } else {
        auto in_mav = make_cfmav<std::complex<float>>(ndim, ain);
        auto out_mav = make_vfmav<std::complex<float>>(ndim, aout);
        if (!has_dup) {
            ducc0::c2c<float>(in_mav, out_mav, ax, forward, static_cast<float>(fct), nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::c2c<float>(in_mav, out_mav, single_ax, forward, static_cast<float>(fct), nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<std::complex<float>>(ndim, aout);
                ducc0::c2c<float>(inout_mav, out_mav, single_ax, forward, 1.0f, nthreads);
            }
        }
    }
}

DLL_EXPORT void
numba_dct(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    uint64_t type, double fct, bool ortho, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    bool has_dup = has_duplicate_axes(ndim, ax);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        auto out_mav = make_vfmav<double>(ndim, aout);
        if (!has_dup) {
            ducc0::dct<double>(in_mav, out_mav, ax, static_cast<int>(type), fct, ortho, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::dct<double>(in_mav, out_mav, single_ax, static_cast<int>(type), fct, ortho, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<double>(ndim, aout);
                ducc0::dct<double>(inout_mav, out_mav, single_ax, static_cast<int>(type), 1.0, ortho, nthreads);
            }
        }
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        auto out_mav = make_vfmav<float>(ndim, aout);
        if (!has_dup) {
            ducc0::dct<float>(in_mav, out_mav, ax, static_cast<int>(type), static_cast<float>(fct), ortho, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::dct<float>(in_mav, out_mav, single_ax, static_cast<int>(type), static_cast<float>(fct), ortho, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<float>(ndim, aout);
                ducc0::dct<float>(inout_mav, out_mav, single_ax, static_cast<int>(type), 1.0f, ortho, nthreads);
            }
        }
    }
}

DLL_EXPORT void
numba_dst(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    uint64_t type, double fct, bool ortho, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    bool has_dup = has_duplicate_axes(ndim, ax);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        auto out_mav = make_vfmav<double>(ndim, aout);
        if (!has_dup) {
            ducc0::dst<double>(in_mav, out_mav, ax, static_cast<int>(type), fct, ortho, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::dst<double>(in_mav, out_mav, single_ax, static_cast<int>(type), fct, ortho, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<double>(ndim, aout);
                ducc0::dst<double>(inout_mav, out_mav, single_ax, static_cast<int>(type), 1.0, ortho, nthreads);
            }
        }
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        auto out_mav = make_vfmav<float>(ndim, aout);
        if (!has_dup) {
            ducc0::dst<float>(in_mav, out_mav, ax, static_cast<int>(type), static_cast<float>(fct), ortho, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::dst<float>(in_mav, out_mav, single_ax, static_cast<int>(type), static_cast<float>(fct), ortho, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<float>(ndim, aout);
                ducc0::dst<float>(inout_mav, out_mav, single_ax, static_cast<int>(type), 1.0f, ortho, nthreads);
            }
        }
    }
}

DLL_EXPORT void
numba_r2c(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        auto out_mav = make_vfmav<std::complex<double>>(ndim, aout);
        ducc0::r2c<double>(in_mav, out_mav, ax, forward, fct, nthreads);
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        auto out_mav = make_vfmav<std::complex<float>>(ndim, aout);
        ducc0::r2c<float>(in_mav, out_mav, ax, forward, static_cast<float>(fct), nthreads);
    }
}

struct rev_arr_info {
    uint64_t ndim_;
    const int64_t* shape_;
    const int64_t* stride_;
    uint64_t ndim() const { return ndim_; }
    int64_t shape(uint64_t i) const { return shape_[i]; }
    int64_t stride(uint64_t i) const { return stride_[i]; }
};

class rev_iter {
private:
    std::vector<int64_t> pos;
    const rev_arr_info arr;
    std::vector<char> rev_axis;
    std::vector<char> rev_jump;
    uint64_t last_axis, last_size;
    std::vector<int64_t> shp;
    int64_t p, rp;
    uint64_t rem;

public:
    rev_iter(const rev_arr_info &arr_, const ducc0::fmav_info::shape_t &axes)
        : pos(arr_.ndim(), 0), arr(arr_), rev_axis(arr_.ndim(), 0),
          rev_jump(arr_.ndim(), 1), p(0), rp(0)
    {
        for (auto ax : axes)
            rev_axis[ax] = 1;
        last_axis = axes.back();
        last_size = arr.shape(last_axis) / 2 + 1;
        shp.resize(arr.ndim());
        for (uint64_t i = 0; i < arr.ndim(); ++i)
            shp[i] = arr.shape(i);
        shp[last_axis] = last_size;
        rem = 1;
        for (auto i : shp)
            rem *= i;
    }
    void advance() {
        --rem;
        for (int i_ = int(pos.size()) - 1; i_ >= 0; --i_) {
            auto i = uint64_t(i_);
            p += arr.stride(i);
            if (!rev_axis[i])
                rp += arr.stride(i);
            else {
                rp -= arr.stride(i);
                if (rev_jump[i]) {
                    rp += arr.shape(i) * arr.stride(i);
                    rev_jump[i] = 0;
                }
            }
            if (++pos[i] < shp[i])
                return;
            pos[i] = 0;
            p -= shp[i] * arr.stride(i);
            if (rev_axis[i]) {
                rp -= (arr.shape(i) - shp[i]) * arr.stride(i);
                rev_jump[i] = 1;
            } else {
                rp -= shp[i] * arr.stride(i);
            }
        }
    }
    int64_t ofs() const { return p; }
    int64_t rev_ofs() const { return rp; }
    uint64_t remaining() const { return rem; }
};

DLL_EXPORT void
numba_c2c_sym(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        ducc0::fmav_info::shape_t out_shp(ndim);
        ducc0::fmav_info::stride_t out_str(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            out_shp[i] = aout->shape_and_strides[i];
            out_str[i] = aout->shape_and_strides[ndim + i] / SIZEOF_COMPLEX128;
        }
        out_shp[ax.back()] = in_mav.shape(ax.back()) / 2 + 1;
        ducc0::vfmav<std::complex<double>> out_mav(reinterpret_cast<std::complex<double>*>(aout->data), out_shp, out_str);
        ducc0::r2c<double>(in_mav, out_mav, ax, forward, fct, nthreads);

        std::vector<int64_t> shp(ndim), str(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            shp[i] = ain->shape_and_strides[i];
            str[i] = aout->shape_and_strides[ndim + i] / SIZEOF_COMPLEX128;
        }
        rev_arr_info ainfo{ndim, shp.data(), str.data()};
        rev_iter iter(ainfo, ax);
        auto data_out = reinterpret_cast<std::complex<double>*>(aout->data);
        while (iter.remaining() > 0) {
            auto v = data_out[iter.ofs()];
            data_out[iter.rev_ofs()] = std::conj(v);
            iter.advance();
        }
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        ducc0::fmav_info::shape_t out_shp(ndim);
        ducc0::fmav_info::stride_t out_str(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            out_shp[i] = aout->shape_and_strides[i];
            out_str[i] = aout->shape_and_strides[ndim + i] / (sizeof(float) * 2);
        }
        out_shp[ax.back()] = in_mav.shape(ax.back()) / 2 + 1;
        ducc0::vfmav<std::complex<float>> out_mav(reinterpret_cast<std::complex<float>*>(aout->data), out_shp, out_str);
        ducc0::r2c<float>(in_mav, out_mav, ax, forward, static_cast<float>(fct), nthreads);

        std::vector<int64_t> shp(ndim), str(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            shp[i] = ain->shape_and_strides[i];
            str[i] = aout->shape_and_strides[ndim + i] / (sizeof(float) * 2);
        }
        rev_arr_info ainfo{ndim, shp.data(), str.data()};
        rev_iter iter(ainfo, ax);
        auto data_out = reinterpret_cast<std::complex<float>*>(aout->data);
        while (iter.remaining() > 0) {
            auto v = data_out[iter.ofs()];
            data_out[iter.rev_ofs()] = std::conj(v);
            iter.advance();
        }
    }
}

DLL_EXPORT void
numba_c2r(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool forward, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    if (ain->itemsize == SIZEOF_COMPLEX128) {
        ducc0::fmav_info::shape_t in_shp(ndim);
        ducc0::fmav_info::stride_t in_str(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            in_shp[i] = aout->shape_and_strides[i];
            in_str[i] = ain->shape_and_strides[ndim + i] / SIZEOF_COMPLEX128;
        }
        in_shp[ax.back()] = aout->shape_and_strides[ax.back()] / 2 + 1;
        ducc0::cfmav<std::complex<double>> in_mav(reinterpret_cast<const std::complex<double>*>(ain->data), in_shp, in_str);
        auto out_mav = make_vfmav<double>(ndim, aout);
        ducc0::c2r<double>(in_mav, out_mav, ax, forward, fct, nthreads);
    } else {
        ducc0::fmav_info::shape_t in_shp(ndim);
        ducc0::fmav_info::stride_t in_str(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            in_shp[i] = aout->shape_and_strides[i];
            in_str[i] = ain->shape_and_strides[ndim + i] / static_cast<ptrdiff_t>(sizeof(float) * 2);
        }
        in_shp[ax.back()] = aout->shape_and_strides[ax.back()] / 2 + 1;
        ducc0::cfmav<std::complex<float>> in_mav(reinterpret_cast<const std::complex<float>*>(ain->data), in_shp, in_str);
        auto out_mav = make_vfmav<float>(ndim, aout);
        ducc0::c2r<float>(in_mav, out_mav, ax, forward, static_cast<float>(fct), nthreads);
    }
}

DLL_EXPORT void
numba_r2r_fftpack(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout, arystruct_t* axes,
    bool real2hermitian, bool forward, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    bool has_dup = has_duplicate_axes(ndim, ax);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        auto out_mav = make_vfmav<double>(ndim, aout);
        if (!has_dup) {
            ducc0::r2r_fftpack<double>(in_mav, out_mav, ax, real2hermitian, forward, fct, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::r2r_fftpack<double>(in_mav, out_mav, single_ax, real2hermitian, forward, fct, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<double>(ndim, aout);
                ducc0::r2r_fftpack<double>(inout_mav, out_mav, single_ax, real2hermitian, forward, 1.0, nthreads);
            }
        }
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        auto out_mav = make_vfmav<float>(ndim, aout);
        if (!has_dup) {
            ducc0::r2r_fftpack<float>(in_mav, out_mav, ax, real2hermitian, forward, static_cast<float>(fct), nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::r2r_fftpack<float>(in_mav, out_mav, single_ax, real2hermitian, forward, static_cast<float>(fct), nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<float>(ndim, aout);
                ducc0::r2r_fftpack<float>(inout_mav, out_mav, single_ax, real2hermitian, forward, 1.0f, nthreads);
            }
        }
    }
}

DLL_EXPORT void
numba_r2r_separable_hartley(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout,
    arystruct_t* axes, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    bool has_dup = has_duplicate_axes(ndim, ax);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        auto out_mav = make_vfmav<double>(ndim, aout);
        if (!has_dup) {
            ducc0::r2r_separable_hartley<double>(in_mav, out_mav, ax, fct, nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::r2r_separable_hartley<double>(in_mav, out_mav, single_ax, fct, nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<double>(ndim, aout);
                ducc0::r2r_separable_hartley<double>(inout_mav, out_mav, single_ax, 1.0, nthreads);
            }
        }
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        auto out_mav = make_vfmav<float>(ndim, aout);
        if (!has_dup) {
            ducc0::r2r_separable_hartley<float>(in_mav, out_mav, ax, static_cast<float>(fct), nthreads);
        } else {
            ducc0::fmav_info::shape_t single_ax{ax[0]};
            ducc0::r2r_separable_hartley<float>(in_mav, out_mav, single_ax, static_cast<float>(fct), nthreads);
            for (size_t i = 1; i < ax.size(); ++i) {
                single_ax[0] = ax[i];
                auto inout_mav = make_cfmav<float>(ndim, aout);
                ducc0::r2r_separable_hartley<float>(inout_mav, out_mav, single_ax, 1.0f, nthreads);
            }
        }
    }
}

DLL_EXPORT void
numba_r2r_genuine_hartley(uint64_t ndim, const arystruct_t* ain, arystruct_t* aout,
    arystruct_t* axes, double fct, uint64_t nthreads = 1) {
    if (ain->nitems == 0 || aout->nitems == 0) return;
    auto ax = make_axes(axes);
    if (ain->itemsize == SIZEOF_FLOAT64) {
        auto in_mav = make_cfmav<double>(ndim, ain);
        auto out_mav = make_vfmav<double>(ndim, aout);
        ducc0::r2r_genuine_hartley<double>(in_mav, out_mav, ax, fct, nthreads);
    } else {
        auto in_mav = make_cfmav<float>(ndim, ain);
        auto out_mav = make_vfmav<float>(ndim, aout);
        ducc0::r2r_genuine_hartley<float>(in_mav, out_mav, ax, static_cast<float>(fct), nthreads);
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