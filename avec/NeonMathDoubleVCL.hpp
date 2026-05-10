/*
Copyright 2026 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Port of the double-precision math functions from the Vector Class
Library (VCL) by Agner Fog (Apache License 2.0,
https://www.agner.org/optimize/) to AArch64 NEON.

Algorithms (VDT/Cephes/Moshier-derived) are unchanged from VCL's
vectormath_exp.h and vectormath_trig.h:
  - exp:     range reduction by ln(2), 13-term Taylor polynomial,
             reconstruction via 2^n bit trick.
  - log:     mantissa/exponent split, rational form (5/5),
             reconstruction via fe*ln(2).
  - sincos:  range reduction to [-pi/4, pi/4] in three steps
             (DP1+DP2+DP3 = pi/2), separate degree-5 polynomials
             for sin and cos in x^2, sign tracking from quadrant.
  - tan:     computed as sin/cos.
Only the SIMD plumbing is rewritten in NEON.

Precision is full double (~15-16 decimal digits) on normal inputs.
*/

#pragma once
#include <arm_neon.h>
#include <limits>

#if defined(__aarch64__)

namespace avec {
namespace detail {

// ---- helpers --------------------------------------------------------------

// FMA wrappers matching VCL's mul_add / nmul_add semantics.
//   vcl_mul_add(a, b, c)  = a*b + c
//   vcl_nmul_add(a, b, c) = c - a*b
inline float64x2_t
vcl_mul_add_pd(float64x2_t a, float64x2_t b, float64x2_t c)
{
  return vfmaq_f64(c, a, b);
}
inline float64x2_t
vcl_nmul_add_pd(float64x2_t a, float64x2_t b, float64x2_t c)
{
  return vfmsq_f64(c, a, b);
}

// 2^n where n is a double whose value is a small integer.
// Magic-number trick: adding (bias + 2^52) puts (n + bias) in the low
// 52 bits of the mantissa; the subsequent left-shift by 52 places that
// value in the exponent field.
inline float64x2_t
vcl_pow2n_pd(float64x2_t n)
{
  float64x2_t a = vaddq_f64(n, vdupq_n_f64(4503599627371519.0)); // 2^52 + 1023
  int64x2_t b = vshlq_n_s64(vreinterpretq_s64_f64(a), 52);
  return vreinterpretq_f64_s64(b);
}

// Mantissa of x mapped to [0.5, 1) by forcing the exponent to -1.
// For positive normal x. Sign bit is cleared by the mantissa mask.
inline float64x2_t
vcl_fraction_2_pd(float64x2_t x)
{
  uint64x2_t bits = vreinterpretq_u64_f64(x);
  bits = vandq_u64(bits, vdupq_n_u64(0x000FFFFFFFFFFFFFULL));
  bits = vorrq_u64(bits, vdupq_n_u64(0x3FE0000000000000ULL));
  return vreinterpretq_f64_u64(bits);
}

// Unbiased exponent of x as a double.
// Magic-number trick: shift the exponent down to bit 0, OR with the bit
// pattern of 2^52 to form (2^52 + raw_exp), subtract (2^52 + 1023).
inline float64x2_t
vcl_exponent_f_pd(float64x2_t x)
{
  uint64x2_t a = vshrq_n_u64(vreinterpretq_u64_f64(x), 52);
  a = vorrq_u64(a, vdupq_n_u64(0x4330000000000000ULL)); // 2^52
  return vsubq_f64(vreinterpretq_f64_u64(a),
                   vdupq_n_f64(4503599627371519.0));    // 2^52 + 1023
}

// Copy the sign bit of b onto a (preserving |a|).
inline float64x2_t
vcl_sign_combine_pd(float64x2_t a, float64x2_t b)
{
  return vbslq_f64(vdupq_n_u64(0x8000000000000000ULL), b, a);
}

// Polynomial helpers in Estrin's scheme (matching vectormath_common.h).

// c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
inline float64x2_t
vcl_poly_5_pd(float64x2_t x,
              double c0, double c1, double c2,
              double c3, double c4, double c5)
{
  float64x2_t x2 = vmulq_f64(x, x);
  float64x2_t x4 = vmulq_f64(x2, x2);
  float64x2_t a = vcl_mul_add_pd(vdupq_n_f64(c3), x, vdupq_n_f64(c2));
  float64x2_t b = vcl_mul_add_pd(vdupq_n_f64(c5), x, vdupq_n_f64(c4));
  float64x2_t c = vcl_mul_add_pd(vdupq_n_f64(c1), x, vdupq_n_f64(c0));
  return vcl_mul_add_pd(a, x2, vcl_mul_add_pd(b, x4, c));
}

// 1*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0  (monic in x^5)
inline float64x2_t
vcl_poly_5n_pd(float64x2_t x,
               double c0, double c1, double c2, double c3, double c4)
{
  float64x2_t x2 = vmulq_f64(x, x);
  float64x2_t x4 = vmulq_f64(x2, x2);
  float64x2_t a = vcl_mul_add_pd(vdupq_n_f64(c3), x, vdupq_n_f64(c2));
  float64x2_t b = vaddq_f64(vdupq_n_f64(c4), x);
  float64x2_t c = vcl_mul_add_pd(vdupq_n_f64(c1), x, vdupq_n_f64(c0));
  return vcl_mul_add_pd(a, x2, vcl_mul_add_pd(b, x4, c));
}

// c13*x^13 + ... + c2*x^2 + x  (note implicit 1*x term, no c0/c1)
inline float64x2_t
vcl_poly_13m_pd(float64x2_t x,
                double c2, double c3, double c4, double c5,
                double c6, double c7, double c8, double c9,
                double c10, double c11, double c12, double c13)
{
  float64x2_t x2 = vmulq_f64(x, x);
  float64x2_t x4 = vmulq_f64(x2, x2);
  float64x2_t x8 = vmulq_f64(x4, x4);

  float64x2_t t13_12 = vcl_mul_add_pd(vdupq_n_f64(c13), x, vdupq_n_f64(c12));
  float64x2_t t11_10 = vcl_mul_add_pd(vdupq_n_f64(c11), x, vdupq_n_f64(c10));
  float64x2_t t9_8   = vcl_mul_add_pd(vdupq_n_f64(c9),  x, vdupq_n_f64(c8));
  float64x2_t hi = vcl_mul_add_pd(t13_12, x4,
                                  vcl_mul_add_pd(t11_10, x2, t9_8));

  float64x2_t t7_6 = vcl_mul_add_pd(vdupq_n_f64(c7), x, vdupq_n_f64(c6));
  float64x2_t t5_4 = vcl_mul_add_pd(vdupq_n_f64(c5), x, vdupq_n_f64(c4));
  float64x2_t t3_2 = vcl_mul_add_pd(vdupq_n_f64(c3), x, vdupq_n_f64(c2));
  float64x2_t mid = vcl_mul_add_pd(vcl_mul_add_pd(t7_6, x2, t5_4), x4,
                                   vcl_mul_add_pd(t3_2, x2, x));
  return vcl_mul_add_pd(hi, x8, mid);
}

// ---- exp ------------------------------------------------------------------

inline float64x2_t
vcl_exp_pd(float64x2_t initial_x)
{
  // 1/n! Taylor coefficients
  constexpr double p2  = 1. / 2.;
  constexpr double p3  = 1. / 6.;
  constexpr double p4  = 1. / 24.;
  constexpr double p5  = 1. / 120.;
  constexpr double p6  = 1. / 720.;
  constexpr double p7  = 1. / 5040.;
  constexpr double p8  = 1. / 40320.;
  constexpr double p9  = 1. / 362880.;
  constexpr double p10 = 1. / 3628800.;
  constexpr double p11 = 1. / 39916800.;
  constexpr double p12 = 1. / 479001600.;
  constexpr double p13 = 1. / 6227020800.;

  constexpr double max_x    = 708.39;
  constexpr double ln2_hi   = 0.693145751953125;
  constexpr double ln2_lo   = 1.42860682030941723212E-6;
  constexpr double VM_LOG2E = 1.44269504088896340736;

  float64x2_t r = vrndnq_f64(vmulq_f64(initial_x, vdupq_n_f64(VM_LOG2E)));
  float64x2_t x = vcl_nmul_add_pd(r, vdupq_n_f64(ln2_hi), initial_x);
  x = vcl_nmul_add_pd(r, vdupq_n_f64(ln2_lo), x);

  float64x2_t z = vcl_poly_13m_pd(
    x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
  float64x2_t n2 = vcl_pow2n_pd(r);
  z = vmulq_f64(vaddq_f64(z, vdupq_n_f64(1.0)), n2);

  // Range / NaN handling: out-of-range → 0 if x<0 else +Inf, NaN passes through.
  float64x2_t inf  = vdupq_n_f64(std::numeric_limits<double>::infinity());
  float64x2_t zero = vdupq_n_f64(0.0);
  uint64x2_t inrange  = vcltq_f64(vabsq_f64(initial_x), vdupq_n_f64(max_x));
  uint64x2_t neg_in   = vcltq_f64(initial_x, zero);
  float64x2_t ovr_val = vbslq_f64(neg_in, zero, inf);
  z = vbslq_f64(inrange, z, ovr_val);

  uint64x2_t is_not_nan = vceqq_f64(initial_x, initial_x);
  z = vbslq_f64(is_not_nan, z, initial_x);
  return z;
}

// ---- log ------------------------------------------------------------------

inline float64x2_t
vcl_log_pd(float64x2_t initial_x)
{
  constexpr double ln2_hi = 0.693359375;
  constexpr double ln2_lo = -2.121944400546905827679E-4;

  constexpr double P0 = 7.70838733755885391666E0;
  constexpr double P1 = 1.79368678507819816313E1;
  constexpr double P2 = 1.44989225341610930846E1;
  constexpr double P3 = 4.70579119878881725854E0;
  constexpr double P4 = 4.97494994976747001425E-1;
  constexpr double P5 = 1.01875663804580931796E-4;

  constexpr double Q0 = 2.31251620126765340583E1;
  constexpr double Q1 = 7.11544750618563894466E1;
  constexpr double Q2 = 8.29875266912776603211E1;
  constexpr double Q3 = 4.52279145837532221105E1;
  constexpr double Q4 = 1.12873587189167450590E1;

  constexpr double SQRTH = 0.70710678118654752440; // sqrt(2)/2
  constexpr double smallest_normal = 2.2250738585072014E-308;

  float64x2_t x1 = initial_x;
  float64x2_t x  = vcl_fraction_2_pd(x1); // mantissa in [0.5, 1)
  float64x2_t fe = vcl_exponent_f_pd(x1); // unbiased exponent

  // if (x > sqrt(2)/2) { fe += 1; x -= 1 } else { x = 2x - 1 }
  uint64x2_t blend = vcgtq_f64(x, vdupq_n_f64(SQRTH));
  x  = vbslq_f64(blend, x, vaddq_f64(x, x));
  fe = vbslq_f64(blend, vaddq_f64(fe, vdupq_n_f64(1.0)), fe);
  x  = vsubq_f64(x, vdupq_n_f64(1.0));

  float64x2_t px = vcl_poly_5_pd(x, P0, P1, P2, P3, P4, P5);
  float64x2_t x2 = vmulq_f64(x, x);
  px = vmulq_f64(px, vmulq_f64(x, x2));
  float64x2_t qx  = vcl_poly_5n_pd(x, Q0, Q1, Q2, Q3, Q4);
  float64x2_t res = vdivq_f64(px, qx);

  res = vcl_mul_add_pd(fe, vdupq_n_f64(ln2_lo), res);
  res = vaddq_f64(res, vcl_nmul_add_pd(x2, vdupq_n_f64(0.5), x));
  res = vcl_mul_add_pd(fe, vdupq_n_f64(ln2_hi), res);

  // Special-case handling, applied as a chain of selects (later ones override
  // earlier ones for the lanes they target):
  //   subnormal-or-zero: -Inf  (overridden below for negative inputs)
  //   negative:          NaN
  //   +Inf:              +Inf  (mostly already correct from the polynomial)
  //   NaN:               passes through
  float64x2_t inf     = vdupq_n_f64(std::numeric_limits<double>::infinity());
  float64x2_t ninf    = vnegq_f64(inf);
  float64x2_t nan_val = vdupq_n_f64(std::numeric_limits<double>::quiet_NaN());

  uint64x2_t underflow = vcltq_f64(x1, vdupq_n_f64(smallest_normal));
  res = vbslq_f64(underflow, ninf, res);

  uint64x2_t negative = vcltq_f64(x1, vdupq_n_f64(0.0));
  res = vbslq_f64(negative, nan_val, res);

  uint64x2_t pos_inf = vceqq_f64(x1, inf);
  res = vbslq_f64(pos_inf, inf, res);

  uint64x2_t is_not_nan = vceqq_f64(x1, x1);
  res = vbslq_f64(is_not_nan, res, x1);
  return res;
}

// ---- sincos ---------------------------------------------------------------

inline void
vcl_sincos_pd(float64x2_t xx, float64x2_t* sin_out, float64x2_t* cos_out)
{
  constexpr double P0sin = -1.66666666666666307295E-1;
  constexpr double P1sin =  8.33333333332211858878E-3;
  constexpr double P2sin = -1.98412698295895385996E-4;
  constexpr double P3sin =  2.75573136213857245213E-6;
  constexpr double P4sin = -2.50507477628578072866E-8;
  constexpr double P5sin =  1.58962301576546568060E-10;

  constexpr double P0cos =  4.16666666666665929218E-2;
  constexpr double P1cos = -1.38888888888730564116E-3;
  constexpr double P2cos =  2.48015872888517045348E-5;
  constexpr double P3cos = -2.75573141792967388112E-7;
  constexpr double P4cos =  2.08757008419747316778E-9;
  constexpr double P5cos = -1.13585365213876817300E-11;

  // pi/2 split into three doubles for extended-precision argument reduction.
  constexpr double DP1 = 7.853981554508209228515625E-1 * 2.;
  constexpr double DP2 = 7.94662735614792836714E-9    * 2.;
  constexpr double DP3 = 3.06161699786838294307E-17   * 2.;

  constexpr double TWO_OVER_PI = 2. / 3.14159265358979323846;

  float64x2_t xa = vabsq_f64(xx);
  float64x2_t y  = vrndnq_f64(vmulq_f64(xa, vdupq_n_f64(TWO_OVER_PI)));
  int64x2_t   q  = vcvtq_s64_f64(y); // y is already an integer-valued double

  // x = ((xa - y*DP1) - y*DP2) - y*DP3
  float64x2_t x = vcl_nmul_add_pd(y, vdupq_n_f64(DP1), xa);
  x = vcl_nmul_add_pd(y, vdupq_n_f64(DP2), x);
  x = vcl_nmul_add_pd(y, vdupq_n_f64(DP3), x);

  float64x2_t x2 = vmulq_f64(x, x);
  float64x2_t s = vcl_poly_5_pd(x2, P0sin, P1sin, P2sin, P3sin, P4sin, P5sin);
  float64x2_t c = vcl_poly_5_pd(x2, P0cos, P1cos, P2cos, P3cos, P4cos, P5cos);
  s = vcl_mul_add_pd(vmulq_f64(x, x2), s, x);                    // s = x + (x*x2)*s
  float64x2_t c_const = vcl_nmul_add_pd(x2, vdupq_n_f64(0.5),
                                        vdupq_n_f64(1.0));        // 1 - x2/2
  c = vcl_mul_add_pd(vmulq_f64(x2, x2), c, c_const);              // c = (1 - x2/2) + (x2*x2)*c

  // Quadrant-based swap and sign.
  uint64x2_t swap = vceqq_s64(vandq_s64(q, vdupq_n_s64(1)),
                              vdupq_n_s64(1));
  float64x2_t sin_v = vbslq_f64(swap, c, s);
  float64x2_t cos_v = vbslq_f64(swap, s, c);

  // Overflow when q is huge (input way outside the reduction range): force
  // sin → 0, cos → 1. Skip when input is non-finite (NaN/Inf handled below).
  uint64x2_t overflow = vcgtq_u64(vreinterpretq_u64_s64(q),
                                  vdupq_n_u64(0x80000000000000ULL));
  uint64x2_t finite_m = vcltq_f64(xa,
                                  vdupq_n_f64(std::numeric_limits<double>::infinity()));
  overflow = vandq_u64(overflow, finite_m);
  sin_v = vbslq_f64(overflow, vdupq_n_f64(0.0), sin_v);
  cos_v = vbslq_f64(overflow, vdupq_n_f64(1.0), cos_v);

  // sin sign: bit 1 of q (placed at bit 63 by <<62) XORed with sign(xx).
  int64x2_t signsin = veorq_s64(vshlq_n_s64(q, 62),
                                vreinterpretq_s64_f64(xx));
  sin_v = vcl_sign_combine_pd(sin_v, vreinterpretq_f64_s64(signsin));

  // cos sign: bit 1 of (q+1), placed at bit 63 by <<62.
  int64x2_t cos_sign = vshlq_n_s64(
    vandq_s64(vaddq_s64(q, vdupq_n_s64(1)), vdupq_n_s64(2)), 62);
  cos_v = vreinterpretq_f64_s64(
    veorq_s64(vreinterpretq_s64_f64(cos_v), cos_sign));

  *sin_out = sin_v;
  *cos_out = cos_v;
}

inline float64x2_t
vcl_sin_pd(float64x2_t x)
{
  float64x2_t s, c;
  vcl_sincos_pd(x, &s, &c);
  return s;
}

inline float64x2_t
vcl_cos_pd(float64x2_t x)
{
  float64x2_t s, c;
  vcl_sincos_pd(x, &s, &c);
  return c;
}

inline float64x2_t
vcl_tan_pd(float64x2_t x)
{
  float64x2_t s, c;
  vcl_sincos_pd(x, &s, &c);
  return vdivq_f64(s, c);
}

} // namespace detail
} // namespace avec

#endif // __aarch64__
