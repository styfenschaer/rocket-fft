// Native frontend between Numba-compiled code and the DUCC0 FFT backend.
//
// Conventions shared with the Python side (rocket_fft/_native.py):
//   * Arrays are passed as pointers to Numba's in-memory array struct.
//   * Integer sequences (shape `s`, `axes`) are passed as `IntSeq` descriptors.
//   * Entry points never throw. They return a `Status`; the Python side turns
//     non-zero codes into exceptions using the table at the end of this file.
//   * Entry points that return arrays fill a caller-provided array struct with
//     a new reference. The memory is allocated through Numba's runtime (NRT),
//     so it is owned and released by Numba like any other array. Doing the
//     allocation here keeps the LLVM IR that Numba has to compile minimal.

#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "ducc0/fft/fft.h"
#include "_special_math.h"

#ifdef _MSC_VER
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

using ducc0::cfmav;
using ducc0::vfmav;
using std::complex;
// NumPy 2 arrays have at most 64 dimensions (NPY_MAXDIMS).
constexpr size_t kMaxDims = 64;

struct TooManyDims {};

// A vector with a fixed capacity of kMaxDims elements stored inline, for the
// per-dimension bookkeeping (shapes, strides, axes) of every call: it never
// touches the heap. Exceeding the capacity throws TooManyDims, which the entry
// points report as an error.
template <typename T>
class DimArray {
public:
    DimArray() = default;
    explicit DimArray(size_t n) { resize(n); }
    DimArray(size_t n, const T& value) {
        resize(n);
        std::fill(begin(), end(), value);
    }
    template <typename It>
    DimArray(It first, It last) { assign(first, last); }

    template <typename It>
    void assign(It first, It last) {
        size_ = 0;
        for (; first != last; ++first) push_back(*first);
    }
    void resize(size_t n) {
        if (n > kMaxDims) throw TooManyDims{};
        for (size_t i = size_; i < n; ++i) data_[i] = T();
        size_ = n;
    }
    void push_back(const T& v) {
        if (size_ == kMaxDims) throw TooManyDims{};
        data_[size_++] = v;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T* data() { return data_; }
    const T* data() const { return data_; }
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }

    friend bool operator==(const DimArray& a, const DimArray& b) {
        return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
    }
    friend bool operator!=(const DimArray& a, const DimArray& b) { return !(a == b); }

private:
    size_t size_ = 0;
    T data_[kMaxDims];
};

using Shape = DimArray<int64_t>;
using Axes = ducc0::fmav_info::shape_t;  // DUCC0's own (heap-allocated) axes type
// Uninitialized, 64-byte aligned scratch memory: callers fill every element, so
// zero-filling it first would only cost an extra pass over memory.
template <typename T>
using Buffer = ducc0::aligned_array<T>;

// -----------------------------------------------------------------------------
// ABI shared with rocket_fft/_native.py (keep both in sync)
// -----------------------------------------------------------------------------

// Bump together with ABI_VERSION in rocket_fft/_native.py whenever an exported
// function, the Options/IntSeq structs or an enum below changes. It lets the
// Python side detect a stale build of this library (e.g. in an editable install).
constexpr int64_t kAbiVersion = 1;

enum Status : int64_t {
    OK = 0,
    AXIS_OUT_OF_RANGE,
    DUPLICATE_AXES,
    NO_AXES,
    SHAPE_AXES_MISMATCH,
    SHAPE_TOO_LONG,
    INVALID_LENGTH,
    INVALID_NORM,
    WORKERS_ZERO,
    WORKERS_OUT_OF_RANGE,
    INVALID_R2R_TYPE,
    OUTPUT_SHAPE_MISMATCH,
    NEGATIVE_TARGET,
    TARGET_TOO_LARGE,
    ZERO_DIVISION,
    NEGATIVE_DIMENSION,
    OUT_OF_MEMORY,
    UNSUPPORTED_DEVICE,
    ROLL_SHAPE_MISMATCH,
    TOO_MANY_DIMS,
    INTERNAL_ERROR,
    STATUS_COUNT,
};

enum DType : int64_t {
    BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64, COMPLEX64, COMPLEX128,
};

enum Kind : int64_t {
    C2C, R2C, C2R, DCT, DST, SEPARABLE_HARTLEY, GENUINE_HARTLEY, FFTPACK,
};

enum Flag : int64_t {
    FORWARD = 1,              // forward transform (DCT/DST: dct/dst, not idct/idst)
    UNIQUE_AXES = 2,          // reject duplicate axes (SciPy semantics)
    LENGTHS_FROM_OUTPUT = 4,  // low-level API: transform lengths come from `aout`
    REAL2HERMITIAN = 8,       // FFTPACK only
    DEFAULT_LAST_AXIS = 16,   // 1-D APIs: `axes=None` means the last axis
    SCIPY = 32,               // SciPy (rather than NumPy) argument semantics
};

enum Norm : int64_t {
    INVALID_NORM_CODE = -1, BACKWARD = 0, ORTHO = 1, FORWARD_NORM = 2, EXPLICIT = 3,
};

// All fields are 8 bytes wide so that the layout is trivially reproducible as
// an LLVM literal struct.
struct Options {
    int64_t kind;
    int64_t flags;
    int64_t in_dtype;
    int64_t out_dtype;
    int64_t norm;
    double fct;          // only used with norm == EXPLICIT
    int64_t r2r_type;    // DCT/DST type (1-4)
    int64_t ortho;       // DCT/DST orthogonalization: -1 = derive from norm, 0, 1
    int64_t workers;
    int64_t workers_given;
};

// Integer sequence of arbitrary element type and stride; count < 0 means None.
struct IntSeq {
    int64_t count;
    const char* data;
    int64_t stride;  // in bytes
    int64_t dtype;
};

// The declarations of ArrayHeader and the NRT structs below mirror Numba's own
// headers (Copyright (c) 2012, Anaconda, Inc., BSD-2-Clause; see LICENSES/numba.txt),
// so that this library can exchange arrays with Numba-compiled code.

// Header of Numba's array struct (numba/_arraystruct.h). It is followed by
// `ndim` shape entries and `ndim` stride entries (in bytes).
struct ArrayHeader {
    void* meminfo;
    void* parent;
    int64_t nitems;
    int64_t itemsize;
    char* data;
};

inline const int64_t* shape_of(const ArrayHeader* a) {
    return reinterpret_cast<const int64_t*>(a + 1);
}

inline const int64_t* strides_of(const ArrayHeader* a, int64_t ndim) {
    return shape_of(a) + ndim;
}

inline int64_t* mutable_shape_of(ArrayHeader* a) {
    return reinterpret_cast<int64_t*>(a + 1);
}

// Numba's external runtime API (numba/core/runtime/nrt_external.h).
struct NRT_MemInfo;
struct NRT_ExternalAllocator {
    void* (*malloc)(size_t size, void* opaque);
    void* (*realloc)(void* ptr, size_t size, void* opaque);
    void (*free)(void* ptr, void* opaque);
    void* opaque;
};
struct NRT_api_functions {
    NRT_MemInfo* (*allocate)(size_t nbytes);
    NRT_MemInfo* (*allocate_external)(size_t nbytes, NRT_ExternalAllocator* allocator);
    NRT_MemInfo* (*manage_memory)(void* data, void (*dtor)(void*));
    void (*acquire)(NRT_MemInfo* mi);
    void (*release)(NRT_MemInfo* mi);
    void* (*get_data)(NRT_MemInfo* mi);
};

// -----------------------------------------------------------------------------
// Global settings
// -----------------------------------------------------------------------------

std::atomic<int64_t> g_default_workers{1};
const NRT_api_functions* g_nrt = nullptr;

// -----------------------------------------------------------------------------
// Array allocation through Numba's runtime
// -----------------------------------------------------------------------------

constexpr size_t kAlignment = 64;
// Large arrays can be backed by transparent huge pages where the OS allows it (as
// NumPy does), so first touching them costs ~500x fewer page faults. That speeds
// up single-threaded transforms of large arrays by up to ~20%, but threads that
// first touch the same 2 MiB page wait for each other, which made multithreaded
// transforms up to 2x slower. Huge pages are therefore only requested for arrays
// that a single thread fills (see allocate_array).
constexpr size_t kHugePageThreshold = size_t(1) << 22;  // 4 MiB
constexpr size_t kHugePageSize = size_t(1) << 21;       // 2 MiB
int g_huge_pages_tag;  // its address marks the allocator that may use huge pages

void* aligned_malloc(size_t size, void* opaque) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const bool huge = opaque == &g_huge_pages_tag && size >= kHugePageThreshold;
#else
    (void)opaque;
    const bool huge = false;
#endif
    // Only whole, aligned huge pages can be used, hence the larger alignment.
    const size_t alignment = huge ? kHugePageSize : kAlignment;
    // Over-allocate and store the original pointer just before the aligned block.
    void* raw = std::malloc(size + alignment + sizeof(void*));
    if (raw == nullptr) return nullptr;
    auto addr = reinterpret_cast<uintptr_t>(raw) + sizeof(void*);
    addr = (addr + alignment - 1) & ~(uintptr_t(alignment) - 1);
    reinterpret_cast<void**>(addr)[-1] = raw;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (huge) madvise(reinterpret_cast<void*>(addr), size, MADV_HUGEPAGE);  // a hint; failure is harmless
#endif
    return reinterpret_cast<void*>(addr);
}

void aligned_free(void* ptr, void*) {
    if (ptr != nullptr) std::free(reinterpret_cast<void**>(ptr)[-1]);
}

void* aligned_realloc(void*, size_t, void*) {
    return nullptr;  // never used for arrays
}

NRT_ExternalAllocator g_allocator = {aligned_malloc, aligned_realloc, aligned_free, nullptr};
NRT_ExternalAllocator g_allocator_huge_pages = {aligned_malloc, aligned_realloc, aligned_free, &g_huge_pages_tag};

struct BadAlloc {};

// NumPy's notion of contiguity: dimensions of length 1 are ignored, and
// empty arrays are both C- and Fortran-contiguous.
bool numpy_contiguous(int64_t ndim, const int64_t* shape, const int64_t* strides, int64_t itemsize, bool fortran) {
    for (int64_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) return true;
    }
    int64_t expected = itemsize;
    for (int64_t k = 0; k < ndim; ++k) {
        const int64_t i = fortran ? k : ndim - 1 - k;
        if (shape[i] == 1) continue;
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

// Byte strides NumPy's empty_like(prototype, shape=shape) gives an array of
// `itemsize`-byte elements (order='K'): C order if the prototype is
// C-contiguous, Fortran order if it is Fortran-contiguous, and otherwise the
// prototype's axis order sorted by decreasing stride magnitude (stable).
Shape empty_like_strides(int64_t ndim, const int64_t* proto_shape, const int64_t* proto_strides,
                         int64_t proto_itemsize, const int64_t* shape, int64_t itemsize) {
    Shape order(static_cast<size_t>(ndim));  // outermost to innermost axis
    for (int64_t i = 0; i < ndim; ++i) order[i] = i;
    if (ndim > 1 && !numpy_contiguous(ndim, proto_shape, proto_strides, proto_itemsize, false)) {
        if (numpy_contiguous(ndim, proto_shape, proto_strides, proto_itemsize, true)) {
            std::reverse(order.begin(), order.end());
        } else {
            // Stable insertion sort by decreasing stride magnitude (std::stable_sort may allocate).
            for (int64_t i = 1; i < ndim; ++i) {
                const int64_t ax = order[i];
                int64_t j = i;
                for (; j > 0 && std::llabs(proto_strides[order[j - 1]]) < std::llabs(proto_strides[ax]); --j) {
                    order[j] = order[j - 1];
                }
                order[j] = ax;
            }
        }
    }
    Shape strides(static_cast<size_t>(ndim));
    int64_t stride = itemsize;
    for (int64_t k = ndim - 1; k >= 0; --k) {
        strides[order[k]] = stride;
        stride *= shape[order[k]];
    }
    return strides;
}

Shape empty_like_strides(const ArrayHeader* proto, int64_t ndim, const int64_t* shape, int64_t itemsize) {
    return empty_like_strides(ndim, shape_of(proto), strides_of(proto, ndim), proto->itemsize, shape, itemsize);
}

// Fill `result` with a new array with the given byte strides, which must
// describe a dense layout (C order if omitted). `nthreads` is the number of
// threads that will fill it.
void allocate_array(ArrayHeader* result, int64_t ndim, const int64_t* shape, int64_t itemsize,
                    const int64_t* strides = nullptr, size_t nthreads = 1) {
    int64_t nitems = 1;
    for (int64_t i = 0; i < ndim; ++i) nitems *= shape[i];
    NRT_ExternalAllocator* allocator = nthreads == 1 ? &g_allocator_huge_pages : &g_allocator;
    NRT_MemInfo* mi = g_nrt->allocate_external(static_cast<size_t>(std::max<int64_t>(nitems * itemsize, 1)), allocator);
    if (mi == nullptr) throw BadAlloc{};
    result->meminfo = mi;
    result->parent = nullptr;
    result->nitems = nitems;
    result->itemsize = itemsize;
    result->data = static_cast<char*>(g_nrt->get_data(mi));
    int64_t* dims = mutable_shape_of(result);
    int64_t stride = itemsize;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        dims[i] = shape[i];
        dims[ndim + i] = strides != nullptr ? strides[i] : stride;
        stride *= shape[i];
    }
}

// Fill `result` with a new reference to the existing array `a`.
void new_reference(ArrayHeader* result, const ArrayHeader* a, int64_t ndim) {
    std::memcpy(result, a, sizeof(ArrayHeader) + 2 * static_cast<size_t>(ndim) * sizeof(int64_t));
    if (result->meminfo != nullptr) g_nrt->acquire(static_cast<NRT_MemInfo*>(result->meminfo));
}

// Restrict `a` (in place) to its leading `shape` elements along each axis,
// like the view a[:shape[0], :shape[1], ...].
void truncate_view(ArrayHeader* a, int64_t ndim, const int64_t* shape) {
    int64_t nitems = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        mutable_shape_of(a)[i] = shape[i];
        nitems *= shape[i];
    }
    a->nitems = nitems;
}

void release(ArrayHeader* a) {
    if (a->meminfo != nullptr) g_nrt->release(static_cast<NRT_MemInfo*>(a->meminfo));
}

int64_t itemsize_of(int64_t dtype) {
    switch (dtype) {
        case BOOL: case INT8: case UINT8: return 1;
        case INT16: case UINT16: return 2;
        case INT32: case UINT32: case FLOAT32: return 4;
        case COMPLEX128: return 16;
        default: return 8;
    }
}

int64_t cpu_count() {
    unsigned int n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int64_t>(n) : 1;
}

// -----------------------------------------------------------------------------
// Element types and strided copies
// -----------------------------------------------------------------------------

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<complex<T>> : std::true_type {};

template <typename T> constexpr int64_t dtype_code();
template <> constexpr int64_t dtype_code<float>() { return FLOAT32; }
template <> constexpr int64_t dtype_code<double>() { return FLOAT64; }
template <> constexpr int64_t dtype_code<complex<float>>() { return COMPLEX64; }
template <> constexpr int64_t dtype_code<complex<double>>() { return COMPLEX128; }

template <typename D, typename S>
inline D convert(const S& v) {
    if constexpr (is_complex<D>::value) {
        using R = typename D::value_type;
        if constexpr (is_complex<S>::value) return D(R(v.real()), R(v.imag()));
        else return D(R(v), R(0));
    } else {
        if constexpr (is_complex<S>::value) return D(v.real());
        else return D(v);
    }
}

// Copies the leading `count` sub-volume of `src` into `dst`, converting
// element types. Both strides are in bytes.
template <typename S, typename D>
void copy_cast(int64_t dim, int64_t ndim, const int64_t* count,
               const char* src, const int64_t* sstr, char* dst, const int64_t* dstr) {
    if (dim == ndim) {  // 0-d array
        *reinterpret_cast<D*>(dst) = convert<D>(*reinterpret_cast<const S*>(src));
        return;
    }
    if (dim == ndim - 1) {
        const int64_t n = count[dim], ss = sstr[dim], ds = dstr[dim];
        if (ss == int64_t(sizeof(S)) && ds == int64_t(sizeof(D))) {
            const S* s = reinterpret_cast<const S*>(src);
            D* d = reinterpret_cast<D*>(dst);
            for (int64_t i = 0; i < n; ++i) d[i] = convert<D>(s[i]);
        } else {
            for (int64_t i = 0; i < n; ++i) {
                *reinterpret_cast<D*>(dst + i * ds) = convert<D>(*reinterpret_cast<const S*>(src + i * ss));
            }
        }
        return;
    }
    for (int64_t i = 0; i < count[dim]; ++i) {
        copy_cast<S, D>(dim + 1, ndim, count, src + i * sstr[dim], sstr, dst + i * dstr[dim], dstr);
    }
}

template <typename D>
void copy_from(int64_t src_dtype, int64_t ndim, const int64_t* count,
               const char* src, const int64_t* sstr, char* dst, const int64_t* dstr) {
    switch (src_dtype) {
        case BOOL:       return copy_cast<uint8_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case INT8:       return copy_cast<int8_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case INT16:      return copy_cast<int16_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case INT32:      return copy_cast<int32_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case INT64:      return copy_cast<int64_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case UINT8:      return copy_cast<uint8_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case UINT16:     return copy_cast<uint16_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case UINT32:     return copy_cast<uint32_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case UINT64:     return copy_cast<uint64_t, D>(0, ndim, count, src, sstr, dst, dstr);
        case FLOAT32:    return copy_cast<float, D>(0, ndim, count, src, sstr, dst, dstr);
        case FLOAT64:    return copy_cast<double, D>(0, ndim, count, src, sstr, dst, dstr);
        case COMPLEX64:  return copy_cast<complex<float>, D>(0, ndim, count, src, sstr, dst, dstr);
        case COMPLEX128: return copy_cast<complex<double>, D>(0, ndim, count, src, sstr, dst, dstr);
    }
}

inline int64_t read_int(const char* p, int64_t dtype) {
    switch (dtype) {
        case BOOL:    return *reinterpret_cast<const uint8_t*>(p);
        case INT8:    return *reinterpret_cast<const int8_t*>(p);
        case INT16:   return *reinterpret_cast<const int16_t*>(p);
        case INT32:   return *reinterpret_cast<const int32_t*>(p);
        case UINT8:   return *reinterpret_cast<const uint8_t*>(p);
        case UINT16:  return *reinterpret_cast<const uint16_t*>(p);
        case UINT32:  return *reinterpret_cast<const uint32_t*>(p);
        case UINT64:  return static_cast<int64_t>(*reinterpret_cast<const uint64_t*>(p));
        case FLOAT32: return static_cast<int64_t>(*reinterpret_cast<const float*>(p));
        case FLOAT64: return static_cast<int64_t>(*reinterpret_cast<const double*>(p));
        default:      return *reinterpret_cast<const int64_t*>(p);
    }
}

inline int64_t read_int(const IntSeq& seq, int64_t i) {
    return read_int(seq.data + i * seq.stride, seq.dtype);
}

Shape read_ints(const IntSeq& seq) {
    Shape out(static_cast<size_t>(std::max<int64_t>(seq.count, 0)));
    for (size_t i = 0; i < out.size(); ++i) out[i] = read_int(seq, static_cast<int64_t>(i));
    return out;
}

// For sequences whose length is not bounded by the number of dimensions.
std::vector<int64_t> read_int_list(const IntSeq& seq) {
    std::vector<int64_t> out(static_cast<size_t>(std::max<int64_t>(seq.count, 0)));
    for (size_t i = 0; i < out.size(); ++i) out[i] = read_int(seq, static_cast<int64_t>(i));
    return out;
}

inline int64_t product(const Shape& shape) {
    int64_t n = 1;
    for (auto v : shape) n *= v;
    return n;
}

Shape contiguous_strides(const Shape& shape, int64_t itemsize) {
    Shape str(shape.size());
    int64_t s = itemsize;
    for (size_t i = shape.size(); i-- > 0;) {
        str[i] = s;
        s *= shape[i];
    }
    return str;
}

// DUCC0's views and functions take std::vectors for shapes, strides and axes. The
// views copy them, so these per-thread scratch vectors (which keep their capacity)
// only avoid allocating the temporaries. Each helper returns a reference that is
// valid until its next call on the same thread.
const Axes& to_axes_t(const int64_t* first, size_t n) {
    thread_local Axes scratch;
    scratch.assign(first, first + n);
    return scratch;
}

const Axes& to_axes_t(const Shape& shape) {
    return to_axes_t(shape.data(), shape.size());
}

// For passing a plan's axes to DUCC0.
const Axes& ducc_axes(const DimArray<size_t>& axes) {
    thread_local Axes scratch;
    scratch.assign(axes.begin(), axes.end());
    return scratch;
}

template <typename T>
const ducc0::fmav_info::stride_t& elem_strides(const int64_t* byte_strides, size_t ndim) {
    thread_local ducc0::fmav_info::stride_t scratch;
    scratch.resize(ndim);
    for (size_t i = 0; i < ndim; ++i) scratch[i] = byte_strides[i] / static_cast<int64_t>(sizeof(T));
    return scratch;
}

template <typename T>
vfmav<T> array_view(ArrayHeader* a, int64_t ndim) {
    return vfmav<T>(reinterpret_cast<T*>(a->data), to_axes_t(shape_of(a), static_cast<size_t>(ndim)),
                    elem_strides<T>(strides_of(a, ndim), ndim));
}

template <typename T>
vfmav<T> buffer_view(Buffer<T>& buf, const Shape& shape) {
    return vfmav<T>(buf.data(), to_axes_t(shape), elem_strides<T>(contiguous_strides(shape, sizeof(T)).data(), shape.size()));
}

// Returns a view of `a` with element type T and the given shape. Zero-copy if
// `a` already matches; otherwise `a` is copied into `buf`, converting its
// element type and zero-padding or truncating to `shape`.
template <typename T>
cfmav<T> materialize(const ArrayHeader* a, int64_t ndim, int64_t dtype, const Shape& shape,
                     Buffer<T>& buf, bool force_copy = false) {
    const int64_t* in_shape = shape_of(a);
    if (!force_copy && dtype == dtype_code<T>() && std::equal(shape.begin(), shape.end(), in_shape)) {
        return cfmav<T>(reinterpret_cast<const T*>(a->data), to_axes_t(shape),
                        elem_strides<T>(strides_of(a, ndim), ndim));
    }
    buf.realloc(static_cast<size_t>(product(shape)));
    Shape count(ndim);
    for (int64_t i = 0; i < ndim; ++i) count[i] = std::min(in_shape[i], shape[i]);
    if (count != shape) std::fill(buf.data(), buf.data() + buf.size(), T(0));  // zero-padding
    Shape dstr = contiguous_strides(shape, sizeof(T));
    if (product(count) > 0) {
        copy_from<T>(dtype, ndim, count.data(), a->data, strides_of(a, ndim),
                     reinterpret_cast<char*>(buf.data()), dstr.data());
    }
    return buffer_view(buf, shape);
}

// Byte range [lo, hi) covered by the elements of `a` (empty if it has none).
std::pair<uintptr_t, uintptr_t> byte_extent(const ArrayHeader* a, int64_t ndim) {
    uintptr_t lo = reinterpret_cast<uintptr_t>(a->data), hi = lo;
    const int64_t* shape = shape_of(a);
    const int64_t* strides = strides_of(a, ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) return {lo, lo};
        const int64_t span = (shape[i] - 1) * strides[i];
        if (span < 0) lo += span;
        else hi += span;
    }
    return {lo, hi + static_cast<uintptr_t>(a->itemsize)};
}

// If the input needs copying anyway (to convert its type or to change a length)
// and the copy would have the output's shape, copies it straight into the output,
// zero-padding where needed, so the caller can transform in place. That saves a
// scratch buffer as large as the output. Returns whether it did so.
template <typename T>
bool copy_into_output(const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, int64_t dtype, const Shape& shape) {
    const int64_t* in_shape = shape_of(ain);
    if (dtype == dtype_code<T>() && std::equal(shape.begin(), shape.end(), in_shape)) return false;  // no copy needed
    if (!std::equal(shape.begin(), shape.end(), shape_of(aout))) return false;
    const auto [in_lo, in_hi] = byte_extent(ain, ndim);
    const auto [out_lo, out_hi] = byte_extent(aout, ndim);
    if (in_lo < out_hi && out_lo < in_hi) return false;  // the output aliases the input

    Shape count(ndim);
    for (int64_t i = 0; i < ndim; ++i) count[i] = std::min(in_shape[i], shape[i]);
    if (count != shape) {  // zero-padding: clear the whole output first
        const T zero(0);
        const Shape zero_strides(ndim, 0);
        copy_cast<T, T>(0, ndim, shape.data(), reinterpret_cast<const char*>(&zero), zero_strides.data(),
                        aout->data, strides_of(aout, ndim));
    }
    if (product(count) > 0) {
        copy_from<T>(dtype, ndim, count.data(), ain->data, strides_of(ain, ndim), aout->data, strides_of(aout, ndim));
    }
    return true;
}

// -----------------------------------------------------------------------------
// Planning: argument validation, output shape, normalization and threads
// -----------------------------------------------------------------------------

struct Plan {
    DimArray<size_t> axes;
    Shape lengths;    // transform length for each entry of `axes`
    Shape in_shape;   // input shape the backend expects (after padding/truncation)
    Shape out_shape;
    bool duplicates = false;
    bool forward = true;
    double fct = 1.0;
    int r2r_type = 0;
    bool ortho = false;
    size_t nthreads = 1;
};

// NumPy computes a transform over several axes as a sequence of 1-D
// transforms. This is the order in which it processes the entries of `axes`
// (with the transform kind of each step). It only matters for repeated axes.
struct Step {
    int64_t kind;
    size_t index;  // into Plan::axes and Plan::lengths
};

DimArray<Step> numpy_steps(int64_t kind, size_t n) {
    DimArray<Step> steps;
    const size_t last = n - 1;
    if (kind == R2C) {
        steps.push_back({R2C, last});
        for (size_t i = last; i-- > 0;) steps.push_back({C2C, i});
    } else if (kind == C2R) {
        for (size_t i = 0; i < last; ++i) steps.push_back({C2C, i});
        steps.push_back({C2R, last});
    } else {
        for (size_t i = n; i-- > 0;) steps.push_back({kind, i});
    }
    return steps;
}

double norm_factor(int64_t norm, bool forward, double n) {
    if (norm == ORTHO) return 1.0 / std::sqrt(n);
    if (norm == (forward ? FORWARD_NORM : BACKWARD)) return 1.0 / n;
    return 1.0;
}

Status resolve_workers(const Options& o, size_t& nthreads) {
    int64_t n = o.workers;
    if (!o.workers_given) {
        n = g_default_workers.load();
    } else if (n == 0) {
        return WORKERS_ZERO;
    } else if (n < 0) {
        n += cpu_count() + 1;
        if (n < 1) return WORKERS_OUT_OF_RANGE;
    }
    nthreads = static_cast<size_t>(n);
    return OK;
}

Status resolve_axes(const Options& o, int64_t ndim, const IntSeq& s, const IntSeq& axes, Plan& p) {
    if (axes.count >= 0) {
        for (int64_t i = 0; i < axes.count; ++i) {
            const int64_t ax = read_int(axes, i);
            if (ax < -ndim || ax >= ndim) return AXIS_OUT_OF_RANGE;
            p.axes.push_back(static_cast<size_t>(ax < 0 ? ax + ndim : ax));
        }
    } else {
        int64_t n = ndim;
        if (s.count >= 0) {
            if (s.count > ndim) return SHAPE_TOO_LONG;
            n = s.count;
        } else if (o.flags & DEFAULT_LAST_AXIS) {
            n = std::min<int64_t>(1, ndim);
        }
        for (int64_t ax = ndim - n; ax < ndim; ++ax) p.axes.push_back(static_cast<size_t>(ax));
    }
    if (p.axes.empty()) return ndim == 0 ? AXIS_OUT_OF_RANGE : NO_AXES;
    if (s.count >= 0 && static_cast<size_t>(s.count) != p.axes.size()) return SHAPE_AXES_MISMATCH;

    uint64_t seen = 0;  // bit mask of axes; ndim <= kMaxDims == 64
    for (auto ax : p.axes) {
        const uint64_t bit = uint64_t(1) << ax;
        if (seen & bit) p.duplicates = true;
        seen |= bit;
    }
    // Only the complex and real FFTs define a meaning for repeated axes (NumPy's).
    const bool fft = (o.kind == C2C || o.kind == R2C || o.kind == C2R);
    if (p.duplicates && ((o.flags & UNIQUE_AXES) || !fft)) return DUPLICATE_AXES;
    return OK;
}

Status resolve_lengths(const Options& o, const ArrayHeader* ain, const ArrayHeader* aout,
                       int64_t ndim, const IntSeq& s, Plan& p) {
    const Shape s_vals = read_ints(s);
    const int64_t* in = shape_of(ain);
    const size_t last = p.axes.size() - 1;
    for (size_t i = 0; i < p.axes.size(); ++i) {
        const size_t ax = p.axes[i];
        int64_t len;
        // In the n-D APIs, -1 in `s` selects the default length (as in NumPy and SciPy).
        const bool use_default = s.count < 0 || (s_vals[i] == -1 && !(o.flags & DEFAULT_LAST_AXIS));
        if (!use_default) {
            len = s_vals[i];
        } else if ((o.flags & LENGTHS_FROM_OUTPUT) && aout != nullptr && !(o.kind == R2C && i == last)) {
            len = shape_of(aout)[ax];
        } else if (o.kind == C2R && i == last) {
            len = 2 * (in[ax] - 1);
        } else {
            len = in[ax];
        }
        // SciPy's n-D inverse real FFTs turn a computed length of 0 (input length 1) into 1.
        if (len == 0 && o.kind == C2R && (o.flags & SCIPY) && !(o.flags & DEFAULT_LAST_AXIS) && s.count < 0) len = 1;
        if (len < 1 || (o.kind == DCT && p.r2r_type == 1 && len < 2)) return INVALID_LENGTH;
        p.lengths.push_back(len);
    }

    p.in_shape.assign(in, in + ndim);
    for (size_t i = 0; i < p.axes.size(); ++i) {
        p.in_shape[p.axes[i]] = (o.kind == C2R && i == last) ? p.lengths[i] / 2 + 1 : p.lengths[i];
    }
    p.out_shape.assign(in, in + ndim);
    for (const Step& step : numpy_steps(o.kind, p.axes.size())) {
        const int64_t len = p.lengths[step.index];
        p.out_shape[p.axes[step.index]] = (step.kind == R2C) ? len / 2 + 1 : len;
    }
    return OK;
}

Status resolve_normalization(const Options& o, Plan& p) {
    if (o.norm == EXPLICIT) {
        p.fct = o.fct;
    } else {
        if (o.norm < BACKWARD || o.norm > FORWARD_NORM) return INVALID_NORM;
        double n = 1.0;
        for (auto len : p.lengths) {
            if (o.kind == DCT || o.kind == DST) {
                const double delta = (p.r2r_type != 1) ? 0.0 : (o.kind == DCT ? -1.0 : 1.0);
                n *= 2.0 * (static_cast<double>(len) + delta);
            } else {
                n *= static_cast<double>(len);
            }
        }
        p.fct = norm_factor(o.norm, p.forward, n);
    }
    p.ortho = (o.ortho < 0) ? (o.norm == ORTHO) : (o.ortho != 0);
    return OK;
}

Status make_plan(const Options& o, const ArrayHeader* ain, const ArrayHeader* aout, int64_t ndim,
                 const IntSeq& s, const IntSeq& axes, Plan& p) {
    if (ndim > static_cast<int64_t>(kMaxDims)) return TOO_MANY_DIMS;
    p.forward = (o.flags & FORWARD) != 0;
    if (o.kind == DCT || o.kind == DST) {
        if (o.r2r_type < 1 || o.r2r_type > 4) return INVALID_R2R_TYPE;
        p.r2r_type = static_cast<int>(o.r2r_type);
        // The inverse of type 2 is type 3 and vice versa; types 1 and 4 are self-inverse.
        if (!p.forward && (p.r2r_type == 2 || p.r2r_type == 3)) p.r2r_type = 5 - p.r2r_type;
    }
    if (Status st = resolve_axes(o, ndim, s, axes, p); st != OK) return st;
    if (Status st = resolve_lengths(o, ain, aout, ndim, s, p); st != OK) return st;
    if (Status st = resolve_normalization(o, p); st != OK) return st;
    return resolve_workers(o, p.nthreads);
}

// -----------------------------------------------------------------------------
// Execution
// -----------------------------------------------------------------------------

template <typename T>
void run_c2c(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    auto out = array_view<complex<T>>(aout, ndim);
    const Axes& axes = ducc_axes(p.axes);
    if (copy_into_output<complex<T>>(ain, aout, ndim, o.in_dtype, p.in_shape)) {
        ducc0::c2c(cfmav<complex<T>>(out), out, axes, p.forward, T(p.fct), p.nthreads);
        return;
    }
    Buffer<complex<T>> buf;
    auto in = materialize<complex<T>>(ain, ndim, o.in_dtype, p.in_shape, buf);
    ducc0::c2c(in, out, axes, p.forward, T(p.fct), p.nthreads);
}

// Complex transform of real input: compute the non-redundant half with an
// r2c transform and fill in the rest using Hermitian symmetry.
template <typename T>
void run_c2c_sym(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    Buffer<T> buf;
    auto in = materialize<T>(ain, ndim, o.in_dtype, p.in_shape, buf);
    auto out = array_view<complex<T>>(aout, ndim);
    Shape half_shape(shape_of(aout), shape_of(aout) + ndim);
    half_shape[p.axes.back()] = half_shape[p.axes.back()] / 2 + 1;
    vfmav<complex<T>> half(out.data(), to_axes_t(half_shape), out.stride());
    const Axes& axes = ducc_axes(p.axes);
    ducc0::r2c(in, half, axes, p.forward, T(p.fct), p.nthreads);
    ducc0::detail_fft::hermiteHelper(0, 0, 0, 0, cfmav<complex<T>>(out), out, axes,
        [](const complex<T>& c, complex<T>&, complex<T>& c1) { c1 = std::conj(c); }, p.nthreads);
}

template <typename T>
void run_r2c(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    Buffer<T> buf;
    auto in = materialize<T>(ain, ndim, o.in_dtype, p.in_shape, buf);
    ducc0::r2c(in, array_view<complex<T>>(aout, ndim), ducc_axes(p.axes), p.forward, T(p.fct), p.nthreads);
}

template <typename T>
void run_c2r(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    Buffer<complex<T>> buf;
    auto in = materialize<complex<T>>(ain, ndim, o.in_dtype, p.in_shape, buf);
    ducc0::c2r(in, array_view<T>(aout, ndim), ducc_axes(p.axes), p.forward, T(p.fct), p.nthreads);
}

// Pads or truncates axis `ax` of the contiguous array `buf` to `len`.
template <typename T>
void resize_axis(Buffer<T>& buf, Shape& shape, size_t ax, int64_t len) {
    if (shape[ax] == len) return;
    Shape new_shape = shape;
    new_shape[ax] = len;
    Shape count = shape;
    count[ax] = std::min(shape[ax], len);
    Buffer<T> resized(static_cast<size_t>(product(new_shape)));
    if (len > shape[ax]) std::fill(resized.data(), resized.data() + resized.size(), T(0));
    if (product(count) > 0) {
        const Shape sstr = contiguous_strides(shape, sizeof(T));
        const Shape dstr = contiguous_strides(new_shape, sizeof(T));
        copy_cast<T, T>(0, static_cast<int64_t>(shape.size()), count.data(),
                        reinterpret_cast<const char*>(buf.data()), sstr.data(),
                        reinterpret_cast<char*>(resized.data()), dstr.data());
    }
    buf = std::move(resized);
    shape = new_shape;
}

// Transforms over repeated axes, computed step by step exactly like NumPy.
template <typename T>
void run_sequential(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    using C = complex<T>;
    Shape shape(shape_of(ain), shape_of(ain) + ndim);
    Buffer<C> work;
    if (o.kind != R2C) materialize<C>(ain, ndim, o.in_dtype, shape, work, /*force_copy=*/true);

    for (const Step& step : numpy_steps(o.kind, p.axes.size())) {
        const size_t ax = p.axes[step.index];
        const int64_t len = p.lengths[step.index];
        const T fct = T(norm_factor(o.norm, p.forward, static_cast<double>(len)));
        if (step.kind == R2C) {
            Buffer<T> real;
            shape[ax] = len;
            auto in = materialize<T>(ain, ndim, o.in_dtype, shape, real);
            shape[ax] = len / 2 + 1;
            work.realloc(static_cast<size_t>(product(shape)));  // r2c writes every element
            ducc0::r2c(in, buffer_view(work, shape), ax, p.forward, fct, p.nthreads);
        } else if (step.kind == C2R) {
            resize_axis(work, shape, ax, len / 2 + 1);
            ducc0::c2r(cfmav<C>(buffer_view(work, shape)), array_view<T>(aout, ndim), ax, p.forward, fct, p.nthreads);
            return;
        } else {
            resize_axis(work, shape, ax, len);
            auto view = buffer_view(work, shape);
            ducc0::c2c(cfmav<C>(view), view, Axes{ax}, p.forward, fct, p.nthreads);
        }
    }
    const Shape sstr = contiguous_strides(shape, sizeof(C));
    copy_cast<C, C>(0, ndim, shape.data(), reinterpret_cast<const char*>(work.data()), sstr.data(),
                    aout->data, strides_of(aout, ndim));
}

// Real view of the real (part = 0) or imaginary (part = 1) components.
template <typename T, typename View>
auto component(const View& v, size_t part) {
    auto str = v.stride();
    for (auto& s : str) s *= 2;
    using Out = std::conditional_t<std::is_same_v<View, vfmav<complex<T>>>, vfmav<T>, cfmav<T>>;
    using Ptr = std::conditional_t<std::is_same_v<View, vfmav<complex<T>>>, T*, const T*>;
    return Out(reinterpret_cast<Ptr>(v.data()) + part, v.shape(), str);
}

template <typename T>
void run_dcst(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    const Axes& axes = ducc_axes(p.axes);
    auto transform = [&](const cfmav<T>& in, const vfmav<T>& out) {
        if (o.kind == DCT) ducc0::dct(in, out, axes, p.r2r_type, T(p.fct), p.ortho, p.nthreads);
        else ducc0::dst(in, out, axes, p.r2r_type, T(p.fct), p.ortho, p.nthreads);
    };
    if (o.out_dtype == COMPLEX64 || o.out_dtype == COMPLEX128) {
        // Complex input: transform real and imaginary parts independently.
        auto out = array_view<complex<T>>(aout, ndim);
        Buffer<complex<T>> buf;
        const cfmav<complex<T>> in = copy_into_output<complex<T>>(ain, aout, ndim, o.in_dtype, p.in_shape)
            ? cfmav<complex<T>>(out) : materialize<complex<T>>(ain, ndim, o.in_dtype, p.in_shape, buf);
        for (size_t part = 0; part < 2; ++part) transform(component<T>(in, part), component<T>(out, part));
        return;
    }
    auto out = array_view<T>(aout, ndim);
    Buffer<T> buf;
    const cfmav<T> in = copy_into_output<T>(ain, aout, ndim, o.in_dtype, p.in_shape)
        ? cfmav<T>(out) : materialize<T>(ain, ndim, o.in_dtype, p.in_shape, buf);
    transform(in, out);
}

template <typename T>
void run_real(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    auto out = array_view<T>(aout, ndim);
    Buffer<T> buf;
    const cfmav<T> in = copy_into_output<T>(ain, aout, ndim, o.in_dtype, p.in_shape)
        ? cfmav<T>(out) : materialize<T>(ain, ndim, o.in_dtype, p.in_shape, buf);
    const T fct = T(p.fct);
    const Axes& axes = ducc_axes(p.axes);
    switch (o.kind) {
        case SEPARABLE_HARTLEY:
            ducc0::r2r_separable_hartley(in, out, axes, fct, p.nthreads);
            break;
        case GENUINE_HARTLEY:
            ducc0::r2r_genuine_hartley(in, out, axes, fct, p.nthreads);
            break;
        case FFTPACK:
            ducc0::r2r_fftpack(in, out, axes, (o.flags & REAL2HERMITIAN) != 0, p.forward, fct, p.nthreads);
            break;
    }
}

template <typename T>
void dispatch(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    if (p.duplicates) return run_sequential<T>(o, ain, aout, ndim, p);
    const bool complex_input = (o.in_dtype == COMPLEX64 || o.in_dtype == COMPLEX128);
    switch (o.kind) {
        case C2C: return complex_input ? run_c2c<T>(o, ain, aout, ndim, p) : run_c2c_sym<T>(o, ain, aout, ndim, p);
        case R2C: return run_r2c<T>(o, ain, aout, ndim, p);
        case C2R: return run_c2r<T>(o, ain, aout, ndim, p);
        case DCT:
        case DST: return run_dcst<T>(o, ain, aout, ndim, p);
        default:  return run_real<T>(o, ain, aout, ndim, p);
    }
}

// -----------------------------------------------------------------------------
// Fast Hankel transform
// -----------------------------------------------------------------------------

// A port of scipy.fft's FFTLog implementation (fhtcoeff, _fhtq and fhtoffset in
// scipy/fft/_fftlog_backend.py, by Nicolas Tessore), which follows the Fortran
// FFTLog code of A. J. S. Hamilton (2000), MNRAS 312, 257. BSD-3-Clause; see
// LICENSES/scipy.txt.

void warn_singular(bool inverse) {
    PyGILState_STATE gil = PyGILState_Ensure();
    const char* msg = inverse ? "singular inverse transform; consider changing the bias"
                              : "singular transform; consider changing the bias";
    if (PyErr_WarnEx(PyExc_UserWarning, msg, 1) < 0) {
        PyErr_Clear();  // warnings turned into errors cannot propagate through Numba
    }
    PyGILState_Release(gil);
}

std::vector<complex<double>> fht_coefficients(int64_t n, double dln, double mu, double offset,
                                              double bias, bool inverse) {
    using namespace special_math;
    const double xp = (mu + 1.0 + bias) * 0.5;
    const double xm = (mu + 1.0 - bias) * 0.5;
    const int64_t m_max = n / 2;
    const double dy = kPi / (static_cast<double>(n) * dln);
    std::vector<complex<double>> u(static_cast<size_t>(m_max + 1));
    for (int64_t m = 0; m <= m_max; ++m) {
        const double y = static_cast<double>(m) * dy;
        const auto v = complex_loggamma({xm, y});
        const auto w = complex_loggamma({xp, y});
        const double re = w.real() - v.real() + kLn2 * bias;
        const double im = w.imag() + v.imag() + 2.0 * y * (kLn2 - offset);
        const double r = std::exp(re);
        u[m] = {r * std::cos(im), r * std::sin(im)};
    }
    if (n % 2 == 0) u[m_max] = u[m_max].real();
    if (!std::isfinite(u[0].real()) || !std::isfinite(u[0].imag())) {
        u[0] = std::pow(2.0, bias) * poch(xm, xp - xm);
    }
    if (std::isinf(u[0].real()) && !inverse) {
        warn_singular(false);
        u[0] = 0.0;
    } else if (u[0] == 0.0 && inverse) {
        warn_singular(true);
        u[0] = std::numeric_limits<double>::infinity();
    }
    return u;
}

// Mirrors scipy.fft.fht / scipy.fft.ifht along the last axis, computed in double precision.
// All loops run over contiguous lanes, which matters for many short transforms.
template <typename T>
void run_fht(const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, int64_t in_dtype,
             double dln, double mu, double offset, double bias, bool inverse) {
    const Shape shape(shape_of(ain), shape_of(ain) + ndim);
    const int64_t n = shape.back();
    const int64_t lanes = product(shape) / n;
    const double jc = (static_cast<double>(n) - 1.0) / 2.0;

    // Multiplies every lane elementwise by w(j), j = 0..n-1.
    auto scale_lanes = [&](double* data, auto&& w) {
        std::vector<double> weights(static_cast<size_t>(n));
        for (int64_t j = 0; j < n; ++j) weights[j] = w(j);
        for (int64_t l = 0; l < lanes; ++l) {
            double* lane = data + l * n;
            for (int64_t j = 0; j < n; ++j) lane[j] *= weights[j];
        }
    };

    Buffer<double> a;
    materialize<double>(ain, ndim, in_dtype, shape, a, /*force_copy=*/true);
    if (bias != 0.0) {
        scale_lanes(a.data(), [&](int64_t j) {
            const double t = (static_cast<double>(j) - jc) * dln;
            return inverse ? std::exp(bias * (t + offset)) : std::exp(-bias * t);
        });
    }

    const auto u = fht_coefficients(n, dln, mu, offset, bias, inverse);
    Shape spec_shape = shape;
    spec_shape.back() = n / 2 + 1;
    Buffer<complex<double>> spec(static_cast<size_t>(product(spec_shape)));
    const size_t axis = static_cast<size_t>(ndim - 1);
    const size_t nthreads = static_cast<size_t>(g_default_workers.load());
    ducc0::r2c(cfmav<double>(buffer_view(a, shape)), buffer_view(spec, spec_shape), axis, true, 1.0, nthreads);
    for (int64_t l = 0; l < lanes; ++l) {
        complex<double>* lane = spec.data() + l * static_cast<int64_t>(u.size());
        for (size_t m = 0; m < u.size(); ++m) {
            lane[m] = inverse ? lane[m] / std::conj(u[m]) : lane[m] * u[m];
        }
    }
    ducc0::c2r(cfmav<complex<double>>(buffer_view(spec, spec_shape)), buffer_view(a, shape), axis,
               false, 1.0 / static_cast<double>(n), nthreads);

    // Output element j is a[n - 1 - j] times the output bias at j: apply the bias
    // in place (indexed by source position), then copy `a` reversed into the output.
    if (bias != 0.0) {
        scale_lanes(a.data(), [&](int64_t k) {
            const double t = (static_cast<double>(n - 1 - k) - jc) * dln;
            return inverse ? std::exp(bias * t) : std::exp(-bias * (t + offset));
        });
    }
    Shape sstr = contiguous_strides(shape, sizeof(double));
    sstr.back() = -sstr.back();
    copy_cast<double, T>(0, ndim, shape.data(), reinterpret_cast<const char*>(a.data() + (n - 1)), sstr.data(),
                         aout->data, strides_of(aout, ndim));
}

// -----------------------------------------------------------------------------
// Optimal transform lengths
// -----------------------------------------------------------------------------

// DUCC0 only provides the next good size (ducc0::good_size_complex/_real, used
// by rf_good_size). These two find the previous one, for scipy.fft.prev_fast_len.
// Adapted from prev_good_size_cmplx and prev_good_size_real in SciPy's copy of
// pocketfft (pocketfft_hdronly.h), BSD-3-Clause; see LICENSES/scipy.txt.

// Returns the largest composite of 2, 3, 5, 7 and 11 that is <= n.
template <typename UInt>
UInt prev_good_size_complex(UInt n) {
    if (n <= 12) return n;
    UInt best = 1;
    for (UInt f11 = 1; f11 <= n; f11 *= 11)
        for (UInt f117 = f11; f117 <= n; f117 *= 7)
            for (UInt f1175 = f117; f1175 <= n; f1175 *= 5) {
                UInt x = f1175;
                while (x * 2 <= n) x *= 2;
                best = std::max(best, x);
                while (true) {
                    if (x * 3 <= n) x *= 3;
                    else if (x % 2 == 0) x /= 2;
                    else break;
                    best = std::max(best, x);
                }
            }
    return best;
}

// Returns the largest composite of 2, 3 and 5 that is <= n.
template <typename UInt>
UInt prev_good_size_real(UInt n) {
    if (n <= 6) return n;
    UInt best = 1;
    for (UInt f5 = 1; f5 <= n; f5 *= 5) {
        UInt x = f5;
        while (x * 2 <= n) x *= 2;
        best = std::max(best, x);
        while (true) {
            if (x * 3 <= n) x *= 3;
            else if (x % 2 == 0) x /= 2;
            else break;
            best = std::max(best, x);
        }
    }
    return best;
}

// -----------------------------------------------------------------------------
// Circular shifts (numpy.roll, fftshift, ifftshift)
// -----------------------------------------------------------------------------

// Copies `n` elements of `Size` bytes between strided locations.
template <size_t Size>
void copy_elements(char* dst, int64_t dstr, const char* src, int64_t sstr, int64_t n) {
    struct Element { unsigned char bytes[Size]; };
    for (int64_t i = 0; i < n; ++i) {
        *reinterpret_cast<Element*>(dst + i * dstr) = *reinterpret_cast<const Element*>(src + i * sstr);
    }
}

void copy_strided(char* dst, int64_t dstr, const char* src, int64_t sstr, int64_t n, int64_t itemsize) {
    if (dstr == itemsize && sstr == itemsize) {
        std::memcpy(dst, src, static_cast<size_t>(n * itemsize));
        return;
    }
    switch (itemsize) {
        case 1: return copy_elements<1>(dst, dstr, src, sstr, n);
        case 2: return copy_elements<2>(dst, dstr, src, sstr, n);
        case 4: return copy_elements<4>(dst, dstr, src, sstr, n);
        case 8: return copy_elements<8>(dst, dstr, src, sstr, n);
        case 16: return copy_elements<16>(dst, dstr, src, sstr, n);
        default:
            for (int64_t i = 0; i < n; ++i) {
                std::memcpy(dst + i * dstr, src + i * sstr, static_cast<size_t>(itemsize));
            }
    }
}

struct RollGeometry {
    int64_t ndim;
    const int64_t* shape;
    const int64_t* shifts;  // in [0, shape[d])
    const int64_t* sstr;
    const int64_t* dstr;
    int64_t itemsize;
    int64_t block_dim;      // from here on: no shifts and both arrays contiguous
    int64_t block_bytes;
};

void roll_dim(const RollGeometry& g, int64_t dim, const char* src, char* dst) {
    if (dim == g.block_dim) {
        std::memcpy(dst, src, static_cast<size_t>(g.block_bytes));
        return;
    }
    const int64_t n = g.shape[dim];
    const int64_t s = g.shifts[dim];
    const int64_t ss = g.sstr[dim], ds = g.dstr[dim];
    if (dim + 1 == g.ndim) {
        // Element i goes to (i + s) mod n: two runs without wrap-around.
        copy_strided(dst + s * ds, ds, src, ss, n - s, g.itemsize);
        copy_strided(dst, ds, src + (n - s) * ss, ss, s, g.itemsize);
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        const int64_t j = (i + s < n) ? i + s : i + s - n;
        roll_dim(g, dim + 1, src + i * ss, dst + j * ds);
    }
}

// Circularly shifts `src` into `dst` (same shape) by `shifts[d]` along each
// dimension `d`, with 0 <= shifts[d] < shape[d]. Strides are in bytes.
void roll(int64_t ndim, const int64_t* shape, const int64_t* shifts,
          const char* src, const int64_t* sstr, char* dst, const int64_t* dstr, int64_t itemsize) {
    RollGeometry g{ndim, shape, shifts, sstr, dstr, itemsize, ndim, itemsize};
    // Find the trailing dimensions that can be copied as one block.
    int64_t bytes = itemsize;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (shifts[d] != 0 || sstr[d] != bytes || dstr[d] != bytes) break;
        bytes *= shape[d];
        g.block_dim = d;
        g.block_bytes = bytes;
    }
    if (g.block_dim == ndim) g.block_bytes = itemsize;
    roll_dim(g, 0, src, dst);
}

// -----------------------------------------------------------------------------
// Status table
// -----------------------------------------------------------------------------

enum ExceptionKind : int64_t { VALUE_ERROR = 0, ZERO_DIVISION_ERROR = 1, MEMORY_ERROR = 2 };

struct StatusInfo {
    const char* message;
    int64_t exception;
};

constexpr StatusInfo kStatusTable[STATUS_COUNT] = {
    {"ok", VALUE_ERROR},
    {"axis exceeds dimensionality of input", VALUE_ERROR},
    {"duplicate axes are not allowed", VALUE_ERROR},
    {"at least one axis must be specified", VALUE_ERROR},
    {"when given, axes and shape arguments have to be of the same length", VALUE_ERROR},
    {"shape requires more axes than are present", VALUE_ERROR},
    {"invalid number of data points specified", VALUE_ERROR},
    {"invalid norm value; should be 'backward', 'ortho' or 'forward'", VALUE_ERROR},
    {"workers must not be zero", VALUE_ERROR},
    {"workers value out of range", VALUE_ERROR},
    {"invalid DCT/DST type; should be 1, 2, 3 or 4", VALUE_ERROR},
    {"output array has wrong shape", VALUE_ERROR},
    {"target cannot be negative", VALUE_ERROR},
    {"target is too large", VALUE_ERROR},
    {"division by zero", ZERO_DIVISION_ERROR},
    {"negative dimensions are not allowed", VALUE_ERROR},
    {"unable to allocate memory", MEMORY_ERROR},
    {"unsupported device; only 'cpu' is supported", VALUE_ERROR},
    {"'shift' and 'axis' must be scalars or sequences of the same length", VALUE_ERROR},
    {"at most 64 dimensions and axes are supported", VALUE_ERROR},
    {"internal error in the DUCC0 backend", VALUE_ERROR},
};

// -----------------------------------------------------------------------------
// Shared implementation of the entry points
// -----------------------------------------------------------------------------

// Runs the planned transform; `aout` must have the planned output shape.
Status execute(const Options& o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim, const Plan& p) {
    if (!std::equal(p.out_shape.begin(), p.out_shape.end(), shape_of(aout))) return OUTPUT_SHAPE_MISMATCH;
    if (product(p.out_shape) == 0) return OK;
    if (o.out_dtype == FLOAT64 || o.out_dtype == COMPLEX128) dispatch<double>(o, ain, aout, ndim, p);
    else dispatch<float>(o, ain, aout, ndim, p);
    return OK;
}

// -----------------------------------------------------------------------------
// Output arrays with exactly NumPy's and SciPy's memory layout
// -----------------------------------------------------------------------------

enum Api : int64_t { API_NUMPY = 0, API_SCIPY = 1 };

// numpy.fft applies one 1-D transform per axis entry, and each allocates its
// output with empty_like(input). Replaying those steps on the layout alone
// gives the strides of NumPy's final result.
void numpy_result(const Options& o, const ArrayHeader* ain, int64_t ndim, const Plan& p, ArrayHeader* result) {
    const bool complex_out = (o.out_dtype == COMPLEX64 || o.out_dtype == COMPLEX128);
    const int64_t complex_size = complex_out ? itemsize_of(o.out_dtype) : 2 * itemsize_of(o.out_dtype);
    Shape shape(shape_of(ain), shape_of(ain) + ndim);
    Shape strides(strides_of(ain, ndim), strides_of(ain, ndim) + ndim);
    int64_t itemsize = ain->itemsize;
    for (const Step& step : numpy_steps(o.kind, p.axes.size())) {
        Shape next = shape;
        const int64_t len = p.lengths[step.index];
        next[p.axes[step.index]] = (step.kind == R2C) ? len / 2 + 1 : len;
        const int64_t next_itemsize = (step.kind == C2R) ? complex_size / 2 : complex_size;
        strides = empty_like_strides(ndim, shape.data(), strides.data(), itemsize, next.data(), next_itemsize);
        shape = next;
        itemsize = next_itemsize;
    }
    allocate_array(result, ndim, shape.data(), itemsize, strides.data(), p.nthreads);
}

// scipy.fft works on `tmp`: the input, converted to floating point if it is
// not (a copy in the input's layout), then truncated (a view) or zero-padded
// (a new C-contiguous array) to the transform lengths. If it may overwrite
// `tmp` (overwrite_x, or `tmp` is already a copy), it returns `tmp` itself.
// Otherwise complex DCT/DST results are allocated with empty_like(tmp), and
// all other results are new C-contiguous arrays.
void scipy_result(const Options& o, const ArrayHeader* ain, int64_t ndim, const Plan& p, bool overwrite,
                  ArrayHeader* result) {
    const int64_t out_size = itemsize_of(o.out_dtype);
    const bool complex_in = (o.in_dtype == COMPLEX64 || o.in_dtype == COMPLEX128);
    const bool converted = !(complex_in || o.in_dtype == FLOAT32 || o.in_dtype == FLOAT64);
    const bool r2r = (o.kind == DCT || o.kind == DST);
    const int64_t* in = shape_of(ain);
    bool padded = false;
    for (int64_t i = 0; i < ndim; ++i) padded = padded || p.in_shape[i] > in[i];
    const bool reuse = overwrite || converted || padded;

    if (!((o.kind == C2C && complex_in) || r2r) || padded) {
        allocate_array(result, ndim, p.out_shape.data(), out_size, nullptr, p.nthreads);
    } else if (reuse && converted) {
        // The converted copy has the input's full shape; the result views its leading part.
        const Shape strides = empty_like_strides(ain, ndim, in, out_size);
        allocate_array(result, ndim, in, out_size, strides.data(), p.nthreads);
        truncate_view(result, ndim, p.out_shape.data());
    } else if (reuse) {
        // A new array object viewing the input's memory, as SciPy returns.
        new_reference(result, ain, ndim);
        result->parent = nullptr;
        truncate_view(result, ndim, p.out_shape.data());
    } else if (r2r && complex_in) {
        const Shape strides = empty_like_strides(ndim, p.out_shape.data(), strides_of(ain, ndim), ain->itemsize,
                                                 p.out_shape.data(), out_size);
        allocate_array(result, ndim, p.out_shape.data(), out_size, strides.data(), p.nthreads);
    } else {
        allocate_array(result, ndim, p.out_shape.data(), out_size, nullptr, p.nthreads);
    }
}

// Runs `f` (which fills `result`), translating C++ exceptions into status
// codes and releasing `result` if anything fails after it was filled.
template <typename F>
int64_t guarded(ArrayHeader* result, F&& f) noexcept {
    bool filled = false;
    int64_t status;
    try {
        status = f(filled);
    } catch (const TooManyDims&) {
        status = TOO_MANY_DIMS;
    } catch (const BadAlloc&) {
        status = OUT_OF_MEMORY;
    } catch (const std::bad_alloc&) {
        status = OUT_OF_MEMORY;
    } catch (...) {
        status = INTERNAL_ERROR;
    }
    if (status != OK && filled && result != nullptr) release(result);
    return status;
}

}  // namespace

// =============================================================================
// Exported entry points
// =============================================================================

DLL_EXPORT int64_t rf_abi_version() noexcept { return kAbiVersion; }

DLL_EXPORT int64_t rf_status_count() noexcept { return STATUS_COUNT; }

DLL_EXPORT const char* rf_status_message(int64_t code) noexcept {
    return (code >= 0 && code < STATUS_COUNT) ? kStatusTable[code].message : kStatusTable[INTERNAL_ERROR].message;
}

DLL_EXPORT int64_t rf_status_exception(int64_t code) noexcept {
    return (code >= 0 && code < STATUS_COUNT) ? kStatusTable[code].exception : VALUE_ERROR;
}

DLL_EXPORT void rf_set_nrt_api(const NRT_api_functions* api) noexcept { g_nrt = api; }

DLL_EXPORT void rf_set_default_workers(int64_t n) noexcept { g_default_workers.store(n); }

DLL_EXPORT int64_t rf_get_default_workers() noexcept { return g_default_workers.load(); }

DLL_EXPORT int64_t rf_cpu_count() noexcept { return cpu_count(); }

// Whether the CPU and the OS support AVX2 and FMA, which the "_avx2" build of
// this library requires (see setup.py and rocket_fft/_native.py).
DLL_EXPORT int64_t rf_cpu_supports_avx2() noexcept {
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return 0;
    __cpuid(r, 1);
    const bool fma = r[2] & (1 << 12), osxsave = r[2] & (1 << 27), avx = r[2] & (1 << 28);
    if (!(fma && osxsave && avx)) return 0;
    if ((_xgetbv(0) & 0x6) != 0x6) return 0;  // the OS saves the YMM registers
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    // Also checks that the OS saves the YMM registers.
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

// Checks the "device" argument in e.g. numpy.fft.fftfreq; not related device/CPU/OS analysis above.
DLL_EXPORT int64_t rf_check_device(const char* data, int64_t length, int32_t kind) noexcept {
    const bool cpu = kind == 1 && data != nullptr && std::string_view(data, static_cast<size_t>(length)) == "cpu";
    return cpu ? OK : UNSUPPORTED_DEVICE;
}

DLL_EXPORT int64_t rf_parse_norm(const char* data, int64_t length, int32_t kind) noexcept {
    if (kind != 1 || data == nullptr) return INVALID_NORM_CODE;
    const std::string_view s(data, static_cast<size_t>(length));
    if (s == "backward") return BACKWARD;
    if (s == "ortho") return ORTHO;
    if (s == "forward") return FORWARD_NORM;
    return INVALID_NORM_CODE;
}

// High-level API. The result is a new reference to `out` if given; otherwise
// it has exactly the memory layout the corresponding NumPy or SciPy function
// returns (see numpy_result and scipy_result).
DLL_EXPORT int64_t rf_transform(const Options* o, const ArrayHeader* ain, int64_t ndim, const IntSeq* s,
                                const IntSeq* axes, const ArrayHeader* out, int64_t overwrite, int64_t api,
                                ArrayHeader* result) noexcept {
    return guarded(result, [&](bool& filled) -> int64_t {
        Plan p;
        if (Status st = make_plan(*o, ain, out, ndim, *s, *axes, p); st != OK) return st;
        if (out != nullptr) new_reference(result, out, ndim);
        else if (api == API_NUMPY) numpy_result(*o, ain, ndim, p, result);
        else scipy_result(*o, ain, ndim, p, overwrite != 0, result);
        filled = true;
        return execute(*o, ain, result, ndim, p);
    });
}

// Low-level API: transform `ain` into the caller-provided `aout`.
DLL_EXPORT int64_t rf_execute(const Options* o, const ArrayHeader* ain, ArrayHeader* aout, int64_t ndim,
                              const IntSeq* s, const IntSeq* axes) noexcept {
    return guarded(nullptr, [&](bool&) -> int64_t {
        Plan p;
        if (Status st = make_plan(*o, ain, aout, ndim, *s, *axes, p); st != OK) return st;
        return execute(*o, ain, aout, ndim, p);
    });
}

DLL_EXPORT int64_t rf_fht(const ArrayHeader* ain, int64_t ndim, int64_t in_dtype, int64_t out_dtype,
                          double dln, double mu, double offset, double bias, int64_t inverse,
                          ArrayHeader* result) noexcept {
    return guarded(result, [&](bool& filled) -> int64_t {
        if (ndim == 0) return AXIS_OUT_OF_RANGE;
        allocate_array(result, ndim, shape_of(ain), itemsize_of(out_dtype));
        filled = true;
        // SciPy returns the reversal of a C-contiguous array along the last axis.
        const int64_t n = shape_of(ain)[ndim - 1];
        if (n > 0) result->data += (n - 1) * result->itemsize;
        mutable_shape_of(result)[2 * ndim - 1] = -result->itemsize;
        if (result->nitems == 0) return OK;
        if (out_dtype == FLOAT64) run_fht<double>(ain, result, ndim, in_dtype, dln, mu, offset, bias, inverse != 0);
        else run_fht<float>(ain, result, ndim, in_dtype, dln, mu, offset, bias, inverse != 0);
        return OK;
    });
}

DLL_EXPORT double rf_fhtoffset(double dln, double mu, double initial, double bias) noexcept {
    using namespace special_math;
    const double xp = (mu + 1.0 + bias) * 0.5;
    const double xm = (mu + 1.0 - bias) * 0.5;
    const double y = kPi / (2.0 * dln);
    const auto zp = complex_loggamma({xp, y});
    const auto zm = complex_loggamma({xm, y});
    const double arg = (kLn2 - initial) / dln + (zp.imag() + zm.imag()) / kPi;
    return initial + (arg - std::round(arg)) * dln;
}

// Returns the length on success and a negated Status on failure.
DLL_EXPORT int64_t rf_good_size(int64_t target, int64_t real) noexcept {
    if (target < 0) return -NEGATIVE_TARGET;
    try {
        const size_t n = static_cast<size_t>(target);
        return static_cast<int64_t>(real ? ducc0::good_size_real(n) : ducc0::good_size_complex(n));
    } catch (...) {
        return -TARGET_TOO_LARGE;
    }
}

DLL_EXPORT int64_t rf_prev_good_size(int64_t target, int64_t real) noexcept {
    if (target < 0) return -NEGATIVE_TARGET;
    const uint64_t n = static_cast<uint64_t>(target);
    if (n > std::numeric_limits<uint64_t>::max() / 11) return -TARGET_TOO_LARGE;
    return static_cast<int64_t>(real ? prev_good_size_real(n) : prev_good_size_complex(n));
}

// Sample frequencies of fftfreq (rfft = 0) or rfftfreq (rfft = 1).
DLL_EXPORT int64_t rf_freq(int64_t n, double d_re, double d_im, int64_t is_complex, int64_t rfft,
                           ArrayHeader* result) noexcept {
    return guarded(result, [&](bool& filled) -> int64_t {
        if (n == 0 || (d_re == 0.0 && d_im == 0.0)) return ZERO_DIVISION;
        if (n < 0) return NEGATIVE_DIMENSION;
        const int64_t length = rfft ? n / 2 + 1 : n;
        allocate_array(result, 1, &length, is_complex ? 16 : 8);
        filled = true;
        const int64_t positive = rfft ? length : (n - 1) / 2 + 1;
        const complex<double> val = 1.0 / (static_cast<double>(n) * complex<double>(d_re, d_im));
        for (int64_t i = 0; i < length; ++i) {
            const double k = static_cast<double>(i < positive ? i : i - n);
            if (is_complex) reinterpret_cast<complex<double>*>(result->data)[i] = k * val;
            else reinterpret_cast<double*>(result->data)[i] = k * val.real();
        }
        return OK;
    });
}

DLL_EXPORT int64_t rf_fftshift(const ArrayHeader* ain, int64_t ndim, const IntSeq* axes, int64_t inverse,
                               ArrayHeader* result) noexcept {
    return guarded(result, [&](bool& filled) -> int64_t {
        const int64_t* shape = shape_of(ain);
        Shape shifts(static_cast<size_t>(ndim), 0);
        auto add_shift = [&](int64_t ax) {
            const int64_t n = shape[ax];
            if (n > 0) shifts[ax] = (shifts[ax] + (inverse ? n - n / 2 : n / 2)) % n;
        };
        if (axes->count < 0) {
            for (int64_t ax = 0; ax < ndim; ++ax) add_shift(ax);
        } else {
            for (int64_t i = 0; i < axes->count; ++i) {
                const int64_t ax = read_int(*axes, i);
                if (ax < -ndim || ax >= ndim) return AXIS_OUT_OF_RANGE;
                add_shift(ax < 0 ? ax + ndim : ax);
            }
        }
        // Like NumPy: the result is empty_like(x).
        const Shape strides = empty_like_strides(ain, ndim, shape, ain->itemsize);
        allocate_array(result, ndim, shape, ain->itemsize, strides.data());
        filled = true;
        if (ain->nitems == 0) return OK;
        if (ndim == 0) {
            std::memcpy(result->data, ain->data, static_cast<size_t>(ain->itemsize));
            return OK;
        }
        roll(ndim, shape, shifts.data(), ain->data, strides_of(ain, ndim), result->data,
             strides_of(result, ndim), ain->itemsize);
        return OK;
    });
}

// =============================================================================
// Extras: functionality beyond numpy.fft/scipy.fft kept from earlier releases
// (numpy.roll and rocket_fft.special). Python side: rocket_fft/_extras.py.
// =============================================================================

// numpy.roll. With `axes` = None, the flattened array is rolled by the sum of
// all shifts. Otherwise shifts and axes are broadcast against each other and
// accumulated per axis, as in NumPy.
DLL_EXPORT int64_t rf_roll(const ArrayHeader* ain, int64_t ndim, const IntSeq* shift, const IntSeq* axes,
                           ArrayHeader* result) noexcept {
    return guarded(result, [&](bool& filled) -> int64_t {
        const int64_t* shape = shape_of(ain);
        const std::vector<int64_t> shift_vals = read_int_list(*shift);
        Shape shifts(static_cast<size_t>(ndim), 0);
        int64_t flat_shift = 0;
        if (axes->count < 0) {
            for (int64_t sh : shift_vals) flat_shift += sh;
        } else {
            const std::vector<int64_t> axis_vals = read_int_list(*axes);
            const size_t ns = shift_vals.size(), na = axis_vals.size();
            if (ns != na && ns != 1 && na != 1) return ROLL_SHAPE_MISMATCH;
            const size_t pairs = (ns == 0 || na == 0) ? 0 : std::max(ns, na);
            for (size_t i = 0; i < pairs; ++i) {
                int64_t ax = axis_vals[na == 1 ? 0 : i];
                if (ax < -ndim || ax >= ndim) return AXIS_OUT_OF_RANGE;
                if (ax < 0) ax += ndim;
                shifts[ax] += shift_vals[ns == 1 ? 0 : i];
            }
            for (int64_t d = 0; d < ndim; ++d) {
                shifts[d] = shape[d] > 0 ? ((shifts[d] % shape[d]) + shape[d]) % shape[d] : 0;
            }
        }
        // Like NumPy: the result is empty_like(a), or C order when rolling the flattened array.
        const Shape strides = (axes->count >= 0) ? empty_like_strides(ain, ndim, shape, ain->itemsize)
                                                 : contiguous_strides(Shape(shape, shape + ndim), ain->itemsize);
        allocate_array(result, ndim, shape, ain->itemsize, strides.data());
        filled = true;
        const int64_t n = result->nitems;
        if (n == 0) return OK;
        if (ndim == 0) {
            std::memcpy(result->data, ain->data, static_cast<size_t>(ain->itemsize));
            return OK;
        }
        if (axes->count >= 0) {
            roll(ndim, shape, shifts.data(), ain->data, strides_of(ain, ndim), result->data,
                 strides_of(result, ndim), ain->itemsize);
            return OK;
        }
        // Roll the C-order flattened array, copying to a contiguous buffer first if needed.
        const Shape cstr = contiguous_strides(Shape(shape, shape + ndim), ain->itemsize);
        const char* flat = ain->data;
        std::vector<char> buf;
        if (!std::equal(cstr.begin(), cstr.end(), strides_of(ain, ndim))) {
            buf.resize(static_cast<size_t>(n * ain->itemsize));
            const Shape zero(static_cast<size_t>(ndim), 0);
            roll(ndim, shape, zero.data(), ain->data, strides_of(ain, ndim), buf.data(), cstr.data(), ain->itemsize);
            flat = buf.data();
        }
        const int64_t k = ((flat_shift % n) + n) % n;
        roll(1, &n, &k, flat, &ain->itemsize, result->data, &ain->itemsize, ain->itemsize);
        return OK;
    });
}

// scipy.special.loggamma for real arguments: NaN for negative arguments.
DLL_EXPORT double rf_loggamma(double x) noexcept {
    if (std::isnan(x) || x < 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return std::numeric_limits<double>::infinity();
    return std::lgamma(x);
}

DLL_EXPORT void rf_loggamma_complex(double re, double im, double* out) noexcept {
    const complex<double> z = special_math::complex_loggamma({re, im});
    out[0] = z.real();
    out[1] = z.imag();
}

DLL_EXPORT double rf_poch(double a, double m) noexcept {
    return special_math::poch(a, m);
}

// The shared library doubles as a (method-less) Python extension module so
// that it is built and located like any other extension. On x86-64 it is built
// twice (see setup.py), the second time as _rocket_fft_numba_avx2.
#ifndef RF_MODULE_NAME
#define RF_MODULE_NAME _rocket_fft_numba
#endif
#define RF_STR_(x) #x
#define RF_STR(x) RF_STR_(x)
#define RF_CAT_(a, b) a##b
#define RF_CAT(a, b) RF_CAT_(a, b)

static PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, RF_STR(RF_MODULE_NAME), nullptr, -1,
                                 nullptr, nullptr, nullptr, nullptr, nullptr};

PyMODINIT_FUNC RF_CAT(PyInit_, RF_MODULE_NAME)(void) {
    return PyModule_Create(&moduledef);
}
