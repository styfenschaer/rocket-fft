#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

typedef struct
{
    double real;
    double imag;
} complex;

typedef complex (*func_type)(complex);

DLL_EXPORT double *
__pyx_fuse_0loggamma_call_by_address(long long addr, double real, double imag)
{
    complex zin = {real, imag};
    complex zout = ((func_type)addr)(zin);
    return &zout.real;
}
