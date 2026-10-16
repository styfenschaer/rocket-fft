// Special functions needed by the Fast Hankel Transform (loggamma for complex
// arguments and the Pochhammer symbol). Ported from SciPy's special module.
#pragma once

#include <cmath>
#include <complex>
#include <limits>

namespace special_math {

constexpr double kLn2 = 0.693147180559945309417232121458176568;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double loggamma_SMALLX = 7.0;
constexpr double loggamma_SMALLY = 7.0;
constexpr double loggamma_HLOG2PI = 0.918938533204672741780329736;      // log(2*pi)/2
constexpr double loggamma_LOGPI = 1.1447298858494001741434262;          // log(pi)
constexpr double loggamma_TAYLOR_RADIUS = 0.2;

inline std::complex<double> evalpoly(const double *coeffs, int degree, std::complex<double> z) noexcept {
    if (degree == 0) {
        return coeffs[0];
    }
    double a = coeffs[0];
    double b = coeffs[1];
    double r = 2.0 * z.real();
    double s = std::norm(z); // real^2 + imag^2
    for (int j = 2; j <= degree; ++j) {
        double tmp = b;
        b = std::fma(-s, a, coeffs[j]);
        a = std::fma(r, a, tmp);
    }
    return z * a + b;
}

inline std::complex<double> loggamma_stirling(std::complex<double> z) noexcept {
    static const double coeffs[] = {
        -2.955065359477124183E-2,  6.4102564102564102564E-3, -1.9175269175269175269E-3,
         8.4175084175084175084E-4, -5.952380952380952381E-4,  7.9365079365079365079E-4,
        -2.7777777777777777778E-3, 8.3333333333333333333E-2
    };
    std::complex<double> rz = 1.0 / z;
    std::complex<double> rzz = rz / z;
    return (z - 0.5) * std::log(z) - z + loggamma_HLOG2PI + rz * evalpoly(coeffs, 7, rzz);
}

inline std::complex<double> loggamma_taylor(std::complex<double> z) noexcept {
    static const double coeffs[] = {
        -4.3478266053040259361E-2,  4.5454556293204669442E-2, -4.7619070330142227991E-2,
         5.000004769810169364E-2,  -5.2631679379616660734E-2,  5.5555767627403611102E-2,
        -5.8823978658684582339E-2,  6.2500955141213040742E-2, -6.6668705882420468033E-2,
         7.1432946295361336059E-2, -7.6932516411352191473E-2,  8.3353840546109004025E-2,
        -9.0954017145829042233E-2,  1.0009945751278180853E-1, -1.1133426586956469049E-1,
         1.2550966952474304242E-1, -1.4404989676884611812E-1,  1.6955717699740818995E-1,
        -2.0738555102867398527E-1,  2.7058080842778454788E-1, -4.0068563438653142847E-1,
         8.2246703342411321824E-1, -5.7721566490153286061E-1
    };
    z -= 1.0;
    return z * evalpoly(coeffs, 22, z);
}

inline std::complex<double> zlog1(std::complex<double> z) noexcept {
    if (std::abs(z - 1.0) > 0.1) {
        return std::log(z);
    }
    if (z == 1.0) {
        return 0.0;
    }
    std::complex<double> coeff = -1.0;
    std::complex<double> res = 0.0;
    z -= 1.0;
    for (int n = 1; n < 17; ++n) {
        coeff *= -z;
        res += coeff / static_cast<double>(n);
        if (coeff == 0.0 || std::abs(res / coeff) < std::numeric_limits<double>::epsilon()) {
            break;
        }
    }
    return res;
}

inline std::complex<double> sinpi(std::complex<double> z) noexcept {
    double x = z.real();
    double piy = kPi * z.imag();
    double abspiy = std::abs(piy);
    double sinpix = std::sin(kPi * x);
    double cospix = std::cos(kPi * x);

    if (abspiy < 700.0) {
        return std::complex<double>(sinpix * std::cosh(piy), cospix * std::sinh(piy));
    }
    double exphpiy = std::exp(abspiy * 0.5);
    double coshfac = 0.5 * sinpix * exphpiy;
    double sinhfac = 0.5 * cospix * exphpiy;
    return std::complex<double>(coshfac * exphpiy, sinhfac * exphpiy);
}

inline std::complex<double> loggamma_recurrence(std::complex<double> z) noexcept {
    int signflips = 0;
    int sb = 0;
    std::complex<double> shiftprod = z;
    z += 1.0;
    while (z.real() <= loggamma_SMALLX) {
        shiftprod *= z;
        int nsb = (shiftprod.imag() < 0.0) ? 1 : 0;
        signflips += (nsb != 0 && sb == 0) ? 1 : 0;
        sb = nsb;
        z += 1.0;
    }
    return loggamma_stirling(z) - std::log(shiftprod) - static_cast<double>(signflips) * 2.0 * kPi * std::complex<double>(0.0, 1.0);
}

inline std::complex<double> complex_loggamma(std::complex<double> z) noexcept {
    if (std::isnan(z.real()) || std::isnan(z.imag())) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    if (z.real() <= 0.0 && z.real() == std::floor(z.real()) && z.imag() == 0.0) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    if (z.real() > loggamma_SMALLX || std::abs(z.imag()) > loggamma_SMALLY) {
        return loggamma_stirling(z);
    }
    if (std::abs(z - 1.0) < loggamma_TAYLOR_RADIUS) {
        return loggamma_taylor(z);
    }
    if (std::abs(z - 2.0) < loggamma_TAYLOR_RADIUS) {
        return zlog1(z - 1.0) + loggamma_taylor(z - 1.0);
    }
    if (z.real() < 0.1) {
        double tmp = std::copysign(2.0 * kPi, z.imag()) * std::floor(0.5 * z.real() + 0.25);
        return std::complex<double>(loggamma_LOGPI, tmp) - std::log(sinpi(z)) - complex_loggamma(1.0 - z);
    }
    if (z.imag() > 0.0 || (z.imag() == 0.0 && !std::signbit(z.imag()))) {
        return loggamma_recurrence(z);
    }
    return std::conj(loggamma_recurrence(std::conj(z)));
}

inline bool is_nonpos_int(double x) noexcept {
    return x <= 0.0 && x == std::floor(x) && std::abs(x) < 1e13;
}

inline double gammasgn(double x) noexcept {
    if (x > 0.0) return 1.0;
    if (x == std::floor(x)) return 0.0;
    double fx = std::floor(x);
    return (std::fmod(-fx, 2.0) != 0.0) ? -1.0 : 1.0;
}

inline double poch(double a, double m) noexcept {
    double r = 1.0;
    while (m >= 1.0) {
        if (a + m == 1.0) break;
        m -= 1.0;
        r *= (a + m);
        if (!std::isfinite(r) || r == 0.0) break;
    }
    while (m <= -1.0) {
        if (a + m == 0.0) break;
        r /= (a + m);
        m += 1.0;
        if (!std::isfinite(r) || r == 0.0) break;
    }
    if (m == 0.0) {
        return r;
    } else if (a > 1e4 && std::abs(m) <= 1.0) {
        return r * std::pow(a, m) *
               (1.0 + m * (m - 1.0) / (2.0 * a) +
                m * (m - 1.0) * (m - 2.0) * (3.0 * m - 1.0) / (24.0 * a * a) +
                m * m * (m - 1.0) * (m - 1.0) * (m - 2.0) * (m - 3.0) / (48.0 * a * a * a));
    }
    if (is_nonpos_int(a + m) && !is_nonpos_int(a) && a + m != m) {
        return std::numeric_limits<double>::infinity();
    }
    if (!is_nonpos_int(a + m) && is_nonpos_int(a)) {
        return 0.0;
    }
    return r * std::exp(std::lgamma(a + m) - std::lgamma(a)) * gammasgn(a + m) * gammasgn(a);
}

} // namespace special_math
