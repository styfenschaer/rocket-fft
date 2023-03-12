#include <stddef.h>
#include <mutex>

#include <Python.h>

#ifdef _MSC_VER
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

typedef struct
{
    double real;
    double imag;
} complex;

typedef complex (*complex_loggamma_type)(complex);
typedef double (*real_loggamma_type)(double);
typedef double (*poch_type)(double, double);

void *complex_loggamma_ptr = NULL;
void *real_loggamma_ptr = NULL;
void *poch_ptr = NULL;

std::mutex import_mutex;

// Copied from: https://github.com/numba/numba/blob/main/numba/_helperlib.c
static void *
import_cython_function(const char *module_name, const char *function_name)
{
    PyObject *module, *capi, *cobj;
    void *res = NULL;
    const char *capsule_name;

    module = PyImport_ImportModule(module_name);
    if (module == NULL)
        return NULL;
    capi = PyObject_GetAttrString(module, "__pyx_capi__");
    Py_DECREF(module);
    if (capi == NULL)
        return NULL;
    cobj = PyMapping_GetItemString(capi, (char *)function_name);
    Py_DECREF(capi);
    if (cobj == NULL)
    {
        PyErr_Clear();
        PyErr_Format(PyExc_ValueError,
                     "No function '%s' found in __pyx_capi__ of '%s'",
                     function_name, module_name);
        return NULL;
    }
    /* 2.7+ => Cython exports a PyCapsule */
    capsule_name = PyCapsule_GetName(cobj);
    if (capsule_name != NULL)
    {
        res = PyCapsule_GetPointer(cobj, capsule_name);
    }
    Py_DECREF(cobj);
    return res;
}

DLL_EXPORT void
init_special_functions()
{
    std::lock_guard<std::mutex> lock(import_mutex);
    char *module_name = "scipy.special.cython_special";
    complex_loggamma_ptr = import_cython_function(module_name, "__pyx_fuse_0loggamma");
    real_loggamma_ptr = import_cython_function(module_name, "__pyx_fuse_1loggamma");
    poch_ptr = import_cython_function(module_name, "poch");
}

DLL_EXPORT void
numba_complex_loggamma(double real, double imag, double *real_out, double *imag_out)
{
    complex zin = {real, imag};
    complex zout = ((complex_loggamma_type)complex_loggamma_ptr)(zin);
    real_out[0] = zout.real;
    imag_out[0] = zout.imag;
}

DLL_EXPORT double
numba_real_loggamma(double z)
{
    return ((real_loggamma_type)real_loggamma_ptr)(z);
}

DLL_EXPORT double
numba_poch(double z, double m)
{
    return ((poch_type)poch_ptr)(z, m);
}