// Port of http://gruntthepeon.free.fr/ssemath/neon_mathfun.html
// to double precision for https://github.com/unevens/avec

/* NEON implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#pragma once
#include "avec/NeonMathFloat.hpp"

#if defined(__aarch64__)

namespace avec {
namespace detail {

typedef float64x2_t v2sd; // vector of 2 double
typedef uint64x2_t v2su;  // vector of 2 uint64
typedef int64x2_t v2si;   // vector of 2 uint64

v2sd
log_pd(v2sd x)
{
  v2sd one = vdupq_n_f64(1);

  x = vmaxq_f64(x, vdupq_n_f64(0)); /* force flush to zero on denormal values */
  v2su invalid_mask = vcleq_f64(x, vdupq_n_f64(0));

  v2si ux = vreinterpretq_s64_f64(x);

  v2si emm0 = vshrq_n_s64(ux, 23);

  /* keep only the fractional part */
  ux = vandq_s64(ux, vdupq_n_s64(c_inv_mant_mask));
  ux = vorrq_s64(ux, vreinterpretq_s64_f64(vdupq_n_f64(0.5)));
  x = vreinterpretq_f64_s64(ux);

  emm0 = vsubq_s64(emm0, vdupq_n_s64(0x7f));
  v2sd e = vcvtq_f64_s64(emm0);

  e = vaddq_f64(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  v2su mask = vcltq_f64(x, vdupq_n_f64(c_cephes_SQRTHF));
  v2sd tmp = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(x), mask));
  x = vsubq_f64(x, one);
  e = vsubq_f64(
    e, vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(one), mask)));
  x = vaddq_f64(x, tmp);

  v2sd z = vmulq_f64(x, x);

  v2sd y = vdupq_n_f64(c_cephes_log_p0);
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p1));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p2));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p3));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p4));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p5));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p6));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p7));
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, vdupq_n_f64(c_cephes_log_p8));
  y = vmulq_f64(y, x);

  y = vmulq_f64(y, z);

  tmp = vmulq_f64(e, vdupq_n_f64(c_cephes_log_q1));
  y = vaddq_f64(y, tmp);

  tmp = vmulq_f64(z, vdupq_n_f64(0.5));
  y = vsubq_f64(y, tmp);

  tmp = vmulq_f64(e, vdupq_n_f64(c_cephes_log_q2));
  x = vaddq_f64(x, y);
  x = vaddq_f64(x, tmp);
  x = vreinterpretq_f64_u64(vorrq_u64(
    vreinterpretq_u64_f64(x), invalid_mask)); // negative arg will be NAN
  return x;
}

v2sd
exp_pd(v2sd x)
{
  v2sd tmp, fx;

  v2sd one = vdupq_n_f64(1);
  x = vminq_f64(x, vdupq_n_f64(c_exp_hi));
  x = vmaxq_f64(x, vdupq_n_f64(c_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f64(vdupq_n_f64(0.5), x, vdupq_n_f64(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f64_s64(vcvtq_s64_f64(fx));

  /* if greater, substract 1 */
  v2su mask = vcgtq_f64(tmp, fx);
  mask = vandq_u64(mask, vreinterpretq_u64_f64(one));

  fx = vsubq_f64(tmp, vreinterpretq_f64_u64(mask));

  tmp = vmulq_f64(fx, vdupq_n_f64(c_cephes_exp_C1));
  v2sd z = vmulq_f64(fx, vdupq_n_f64(c_cephes_exp_C2));
  x = vsubq_f64(x, tmp);
  x = vsubq_f64(x, z);

  static const double cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1,
                                         c_cephes_exp_p2, c_cephes_exp_p3,
                                         c_cephes_exp_p4, c_cephes_exp_p5 };
  v2sd y = vld1q_dup_f64(cephes_exp_p + 0);
  v2sd c1 = vld1q_dup_f64(cephes_exp_p + 1);
  v2sd c2 = vld1q_dup_f64(cephes_exp_p + 2);
  v2sd c3 = vld1q_dup_f64(cephes_exp_p + 3);
  v2sd c4 = vld1q_dup_f64(cephes_exp_p + 4);
  v2sd c5 = vld1q_dup_f64(cephes_exp_p + 5);

  y = vmulq_f64(y, x);
  z = vmulq_f64(x, x);
  y = vaddq_f64(y, c1);
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, c2);
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, c3);
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, c4);
  y = vmulq_f64(y, x);
  y = vaddq_f64(y, c5);

  y = vmulq_f64(y, z);
  y = vaddq_f64(y, x);
  y = vaddq_f64(y, one);

  /* build 2^n */
  int64x2_t mm;
  mm = vcvtq_s64_f64(fx);
  mm = vaddq_s64(mm, vdupq_n_s64(0x7f));
  mm = vshlq_n_s64(mm, 23);
  v2sd pow2n = vreinterpretq_f64_s64(mm);

  y = vmulq_f64(y, pow2n);
  return y;
}

void
sincos_pd(v2sd x, v2sd* ysin, v2sd* ycos)
{ // any x
  v2sd xmm1, xmm2, xmm3, y;

  v2su emm2;

  v2su sign_mask_sin, sign_mask_cos;
  sign_mask_sin = vcltq_f64(x, vdupq_n_f64(0));
  x = vabsq_f64(x);

  /* scale by 4/Pi */
  y = vmulq_f64(x, vdupq_n_f64(c_cephes_FOPI));

  /* store the integer part of y in mm0 */
  emm2 = vcvtq_u64_f64(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = vaddq_u64(emm2, vdupq_n_u64(1));
  emm2 = vandq_u64(emm2, vdupq_n_u64(~1));
  y = vcvtq_f64_u64(emm2);

  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  v2su poly_mask = vtstq_u64(emm2, vdupq_n_u64(2));

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = vmulq_n_f64(y, c_minus_cephes_DP1);
  xmm2 = vmulq_n_f64(y, c_minus_cephes_DP2);
  xmm3 = vmulq_n_f64(y, c_minus_cephes_DP3);
  x = vaddq_f64(x, xmm1);
  x = vaddq_f64(x, xmm2);
  x = vaddq_f64(x, xmm3);

  sign_mask_sin = veorq_u64(sign_mask_sin, vtstq_u64(emm2, vdupq_n_u64(4)));
  sign_mask_cos = vtstq_u64(vsubq_u64(emm2, vdupq_n_u64(2)), vdupq_n_u64(4));

  /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */
  v2sd z = vmulq_f64(x, x);
  v2sd y1, y2;

  y1 = vmulq_n_f64(z, c_coscof_p0);
  y2 = vmulq_n_f64(z, c_sincof_p0);
  y1 = vaddq_f64(y1, vdupq_n_f64(c_coscof_p1));
  y2 = vaddq_f64(y2, vdupq_n_f64(c_sincof_p1));
  y1 = vmulq_f64(y1, z);
  y2 = vmulq_f64(y2, z);
  y1 = vaddq_f64(y1, vdupq_n_f64(c_coscof_p2));
  y2 = vaddq_f64(y2, vdupq_n_f64(c_sincof_p2));
  y1 = vmulq_f64(y1, z);
  y2 = vmulq_f64(y2, z);
  y1 = vmulq_f64(y1, z);
  y2 = vmulq_f64(y2, x);
  y1 = vsubq_f64(y1, vmulq_f64(z, vdupq_n_f64(0.5)));
  y2 = vaddq_f64(y2, x);
  y1 = vaddq_f64(y1, vdupq_n_f64(1));

  /* select the correct result from the two polynoms */
  v2sd ys = vbslq_f64(poly_mask, y1, y2);
  v2sd yc = vbslq_f64(poly_mask, y2, y1);
  *ysin = vbslq_f64(sign_mask_sin, vnegq_f64(ys), ys);
  *ycos = vbslq_f64(sign_mask_cos, yc, vnegq_f64(yc));
}

v2sd
sin_pd(v2sd x)
{
  v2sd ysin, ycos;
  sincos_pd(x, &ysin, &ycos);
  return ysin;
}

v2sd
cos_pd(v2sd x)
{
  v2sd ysin, ycos;
  sincos_pd(x, &ysin, &ycos);
  return ycos;
}

} // namespace detail
} // namespace avec

#endif