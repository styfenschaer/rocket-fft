cimport numpy 
cimport scipy.special.cython_special


cdef api void __pyx_fuse_0loggamma(double real, double imag, double *real_out, double *imag_out):
    cdef double complex zin, zout

    zin.real = real
    zin.imag = imag
    zout = scipy.special.cython_special.loggamma(zin)
    real_out[0] = zout.real
    imag_out[0] = zout.imag