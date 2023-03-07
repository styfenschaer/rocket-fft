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

DLL_EXPORT void
__pyx_fuse_0loggamma_call_by_address2(long long addr, double real, double imag, double *real_out, double *imag_out)
{
    complex zin = {real, imag};
    complex zout = ((func_type)addr)(zin);
    real_out[0] = zout.real;
    imag_out[0] = zout.imag;
}
