/*
Copyright 2021 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
 * This file implements some of the functionality of  Agner Fog's Vectorclass
 * for NEON.
 * The classes Vec2d and Vec4f are implemented with most of their operators and
 * functions, as in the file vectorf128.h.
 * Overloads for exp, log, sin and cos are implemented in the NeonMath* files
 * using Julien Pommier's neon_mathfun.
 * Some code has been adapted from https://github.com/DLTcollab/sse2neon
 * */

/****************************  vectorf128.h   *******************************
 * (c) Copyright 2012-2020 Agner Fog.
 * Apache License version 2.0 or later.
 *****************************************************************************/

/*
 * sse2neon is freely redistributable under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// sse2neon by
//   John W. Ratcliff <jratcliffscarab@gmail.com>
//   Brandon Rowlett <browlett@nvidia.com>
//   Ken Fast <kfast@gdeb.com>
//   Eric van Beurden <evanbeurden@nvidia.com>
//   Alexander Potylitsin <apotylitsin@nvidia.com>
//   Hasindu Gamaarachchi <hasindu2008@gmail.com>
//   Jim Huang <jserv@biilabs.io>
//   Mark Cheng <marktwtn@biilabs.io>
//   Malcolm James MacLeod <malcolm@gulden.com>
//   Devin Hussey (easyaspi314) <husseydevin@gmail.com>
//   Sebastian Pop <spop@amazon.com>
//   Developer Ecosystem Engineering <DeveloperEcosystemEngineering@apple.com>
//   Danila Kutenin <danilak@google.com>
//   François Turban (JishinMaster) <francois.turban@gmail.com>
//   Pei-Hsuan Hung <afcidk@gmail.com>
//   Yang-Hao Yuan <yanghau@biilabs.io>

#pragma once

#include <arm_neon.h>

namespace avec {

class Vec4f
{
protected:
  float32x4_t vec; // Float vector
public:
  // Default constructor:
  Vec4f() {}

  // Constructor to broadcast the same value into all elements:
  Vec4f(float f) { vec = vdupq_n_f32(f); }

  // Constructor to build from all elements:
  Vec4f(float f0, float f1, float f2, float f3)
  {
    float __attribute__((aligned(16))) data[4] = { f0, f1, f2, f3 };
    vec = vld1q_f32(data);
  }

  // Constructor to convert from type float32x4_t used in intrinsics:
  Vec4f(float32x4_t const x) { vec = x; }

  // Assignment operator to convert from type float32x4_t used in intrinsics:
  Vec4f& operator=(float32x4_t const x)
  {
    vec = x;
    return *this;
  }

  // Type cast operator to convert to float32x4_t used in intrinsics
  operator float32x4_t() const { return vec; }

  // Member function to load from array
  Vec4f& load(float const* p)
  {
    vec = vld1q_f32(p);
    return *this;
  }

  Vec4f& load_a(float const* p)
  {
    vec = vld1q_f32(p);
    return *this;
  }

  // Member function to store into array
  void store(float* p) const { vst1q_f32(p, vec); }

  void store_a(float* p) const { vst1q_f32(p, vec); }

  // Member function extract a single element from vector
  float extract(int index) const
  {
    float x[4];
    store(x);
    return x[index & 3];
  }

  // Extract a single element. Use store function if extracting more than one
  // element. Operator [] can only read an element, not write.
  float operator[](int index) const { return extract(index); }

  static constexpr int size() { return 4; }

  static constexpr int elementtype() { return 16; }

  typedef float32x4_t registertype;
};

using Vec4fb = Vec4f;

#if defined(__aarch64__)

class Vec2d
{
protected:
  float64x2_t vec; // double vector
public:
  // Default constructor:
  Vec2d() {}

  // Constructor to broadcast the same value into all elements:
  Vec2d(double d) { vec = vdupq_n_f64(d); }

  // Constructor to build from all elements:
  Vec2d(double d0, double d1)
  {
    double __attribute__((aligned(16))) data[2] = { d0, d1 };
    vec = vld1q_f64(data);
  }

  // Constructor to convert from type float64x2_t used in intrinsics:
  Vec2d(float64x2_t const x) { vec = x; }

  // Assignment operator to convert from type float64x2_t used in intrinsics:
  Vec2d& operator=(float64x2_t const x)
  {
    vec = x;
    return *this;
  }

  // Type cast operator to convert to float64x2_t used in intrinsics
  operator float64x2_t() const { return vec; }

  // Member function to load from array (unaligned)
  Vec2d& load(double const* p)
  {
    vec = vld1q_f64(p);
    return *this;
  }

  // Member function to load from array
  Vec2d const load_a(double const* p)
  {
    vec = vld1q_f64((const double*)p);
    return *this;
  }

  // Member function to store into array
  void store(double* p) const { vst1q_f64((float64_t*)p, vec); }

  void store_a(double* p) const { vst1q_f64((float64_t*)p, vec); }

  // Member function extract a single element from vector
  double extract(int index) const
  {
    double x[2];
    store(x);
    return x[index & 1];
  }

  // Extract a single element. Use store function if extracting more than one
  // element. Operator [] can only read an element, not write.
  double operator[](int index) const { return extract(index); }

  static constexpr int size() { return 2; }

  static constexpr int elementtype() { return 17; }

  typedef float64x2_t registertype;
};

#else

class Vec2d
{
  Vec2d() = default;

public:
  static constexpr int size() { return 2; }
};

#endif // defined(__aarch64__)

using Vec2db = Vec2d;

// sse2neon stuff and some other implementation details
namespace {
inline float32x4_t
selectf(float32x4_t const s, float32x4_t const a, float32x4_t const b)
{
  return vbslq_f32(vreinterpretq_u32_f32(s), a, b);
}
#if defined(__aarch64__)
inline float64x2_t
selectd(float64x2_t const s, float64x2_t const a, float64x2_t const b)
{
  return vbslq_f64(vreinterpretq_u64_f64(s), a, b);
}
#endif

inline int32x4_t
_mm_set1_epi32(int _i)
{
  return vdupq_n_s32(_i);
}

inline float32x4_t
_mm_castsi128_ps(int32x4_t a)
{
  return vreinterpretq_f32_s32(a);
}

inline float32x4_t
_mm_xor_ps(float32x4_t a, float32x4_t b)
{
  return vreinterpretq_f32_s32(
    veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

inline int32x4_t
_mm_cvtps_epi32(float32x4_t a)
{
#if defined(__aarch64__)
  return vcvtnq_s32_f32(a);
#else
  uint32x4_t signmask = vdupq_n_u32(0x80000000);
  float32x4_t half = vbslq_f32(signmask, (a), vdupq_n_f32(0.5f)); /* +/- 0.5 */
  int32x4_t r_normal =
    vcvtq_s32_f32(vaddq_f32((a), half));  /* round to integer: [a + 0.5]*/
  int32x4_t r_trunc = vcvtq_s32_f32((a)); /* truncate to integer: [a] */
  int32x4_t plusone = vreinterpretq_s32_u32(
    vshrq_n_u32(vreinterpretq_u32_s32(vnegq_s32(r_trunc)), 31)); /* 1 or 0 */
  int32x4_t r_even = vbicq_s32(vaddq_s32(r_trunc, plusone),
                               vdupq_n_s32(1)); /* ([a] + {0,1}) & ~1 */
  float32x4_t delta = vsubq_f32(
    (a), vcvtq_f32_s32(r_trunc)); /* compute delta: delta = (a - [a]) */
  uint32x4_t is_delta_half = vceqq_f32(delta, half); /* delta == +/- 0.5 */
  return vbslq_s32(is_delta_half, r_even, r_normal);
#endif
}

inline float32x4_t
_mm_cvtepi32_ps(int32x4_t a)
{
  return vcvtq_f32_s32(a);
}

inline int32x4_t
_mm_cvttps_epi32(float32x4_t a)
{
  return vcvtq_s32_f32(a);
}

inline float32x4_t
_mm_blendv_ps(float32x4_t a, float32x4_t b, float32x4_t mask)
{
  return vbslq_f32(vreinterpretq_u32_f32(mask), b, a);
}

inline int32x4_t
_mm_setr_epi32(int i3, int i2, int i1, int i0)
{
  int32_t __attribute__((aligned(16))) data[4] = { i3, i2, i1, i0 };
  return vld1q_s32(data);
}

constexpr auto _MM_FROUND_TO_NEAREST_INT = 0x00;
constexpr auto _MM_FROUND_TO_NEG_INF = 0x01;
constexpr auto _MM_FROUND_TO_POS_INF = 0x02;
constexpr auto _MM_FROUND_TO_ZERO = 0x03;
constexpr auto _MM_FROUND_CUR_DIRECTION = 0x04;
constexpr auto _MM_FROUND_NO_EXC = 0x08;

constexpr auto _MM_FROUND_TO_NEAREST_INT_MM_FROUND_NO_EXC =
  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
constexpr auto _MM_FROUND_TO_NEG_INF_MM_FROUND_NO_EXC =
  _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
constexpr auto __MM_FROUND_TO_POS_INF_MM_FROUND_NO_EXC =
  _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
constexpr auto _MM_FROUND_TO_ZERO_MM_FROUND_NO_EXC =
  _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;

template<int rounding>
inline float32x4_t
_mm_round_ps(float32x4_t a);

template<>
inline float32x4_t
_mm_round_ps<_MM_FROUND_TO_NEAREST_INT_MM_FROUND_NO_EXC>(float32x4_t a)
{
#if defined(__aarch64__)
  return vrndnq_f32(a);
#else
  return _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
#endif
}
template<>
inline float32x4_t
_mm_round_ps<_MM_FROUND_TO_NEG_INF_MM_FROUND_NO_EXC>(float32x4_t a)
{
#if defined(__aarch64__)
  return vrndmq_f32(a);
#else
  return (
    float32x4_t){ floorf(a[0]), floorf(a[1]), floorf(a[2]), floorf(a[3]) };
#endif
}
template<>
inline float32x4_t
_mm_round_ps<__MM_FROUND_TO_POS_INF_MM_FROUND_NO_EXC>(float32x4_t a)
{
#if defined(__aarch64__)
  return vrndpq_f32(a);
#else
  return (float32x4_t){ ceilf(a[0]), ceilf(a[1]), ceilf(a[2]), ceilf(a[3]) };
#endif
}
template<>
inline float32x4_t
_mm_round_ps<_MM_FROUND_TO_ZERO_MM_FROUND_NO_EXC>(float32x4_t a)
{
#if defined(__aarch64__)
  return vrndq_f32(a);
#else
  auto zero = Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
  auto neg_inf = Vec4f(floorf(a[0]), floorf(a[1]), floorf(a[2]), floorf(a[3]));
  auto pos_inf = Vec4f(ceilf(a[0]), ceilf(a[1]), ceilf(a[2]), ceilf(a[3]));
  return selectf(vreinterpretq_f32_u32(vcleq_f32(a, zero)), pos_inf, neg_inf);
#endif
}
template<>
inline float32x4_t
_mm_round_ps<_MM_FROUND_CUR_DIRECTION>(float32x4_t a)
{
#if defined(__aarch64__)
  return vrndiq_f32(a);
#else
  return (
    float32x4_t){ roundf(a[0]), roundf(a[1]), roundf(a[2]), roundf(a[3]) };
#endif
}

#if defined(__aarch64__)
inline float64x2_t
_mm_xor_pd(float64x2_t a, float64x2_t b)
{
  return vreinterpretq_f64_s64(
    veorq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)));
}

inline int64x2_t
_mm_castpd_si128(float64x2_t a)
{
  return vreinterpretq_s64_f64(a);
}

template<int rounding>
inline float64x2_t
_mm_round_pd(float64x2_t a);

template<>
inline float64x2_t
_mm_round_pd<_MM_FROUND_TO_NEAREST_INT_MM_FROUND_NO_EXC>(float64x2_t a)
{
  return vrndnq_f64(a);
}
template<>
inline float64x2_t
_mm_round_pd<_MM_FROUND_TO_NEG_INF_MM_FROUND_NO_EXC>(float64x2_t a)
{
  return vrndmq_f64(a);
}
template<>
inline float64x2_t
_mm_round_pd<__MM_FROUND_TO_POS_INF_MM_FROUND_NO_EXC>(float64x2_t a)
{
  return vrndpq_f64(a);
}
template<>
inline float64x2_t
_mm_round_pd<_MM_FROUND_TO_ZERO_MM_FROUND_NO_EXC>(float64x2_t a)
{
  return vrndq_f64(a);
}
template<>
inline float64x2_t
_mm_round_pd<_MM_FROUND_CUR_DIRECTION>(float64x2_t a)
{
  return vrndiq_f64(a);
}

#endif

} // namespace

// Generate a constant vector of 4 integers stored in memory.
template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3>
static inline constexpr int32x4_t
constant4ui()
{
  return _mm_setr_epi32(i0, i1, i2, i3);
}

/*****************************************************************************
 *
 *          Operators for Vec4f
 *
 *****************************************************************************/

// vector operator + : add element by element
static inline Vec4f
operator+(Vec4f const a, Vec4f const b)
{
  return vaddq_f32(a, b);
}

// vector operator + : add vector and scalar
static inline Vec4f
operator+(Vec4f const a, float b)
{
  return a + Vec4f(b);
}

static inline Vec4f
operator+(float a, Vec4f const b)
{
  return Vec4f(a) + b;
}

// vector operator += : add
static inline Vec4f&
operator+=(Vec4f& a, Vec4f const b)
{
  a = a + b;
  return a;
}

// postfix operator ++
static inline Vec4f
operator++(Vec4f& a, int)
{
  Vec4f a0 = a;
  a = a + 1.0f;
  return a0;
}

// prefix operator ++
static inline Vec4f&
operator++(Vec4f& a)
{
  a = a + 1.0f;
  return a;
}

// vector operator - : subtract element by element
static inline Vec4f
operator-(Vec4f const a, Vec4f const b)
{
  return vsubq_f32(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec4f
operator-(Vec4f const a, float b)
{
  return a - Vec4f(b);
}

static inline Vec4f
operator-(float a, Vec4f const b)
{
  return Vec4f(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec4f
operator-(Vec4f const a)
{
  return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
}

// vector operator -= : subtract
static inline Vec4f&
operator-=(Vec4f& a, Vec4f const b)
{
  a = a - b;
  return a;
}

// postfix operator --
static inline Vec4f
operator--(Vec4f& a, int)
{
  Vec4f a0 = a;
  a = a - 1.0f;
  return a0;
}

// prefix operator --
static inline Vec4f&
operator--(Vec4f& a)
{
  a = a - 1.0f;
  return a;
}

// vector operator * : multiply element by element
static inline Vec4f
operator*(Vec4f const a, Vec4f const b)
{
  return vmulq_f32(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec4f
operator*(Vec4f const a, float b)
{
  return a * Vec4f(b);
}

static inline Vec4f
operator*(float a, Vec4f const b)
{
  return Vec4f(a) * b;
}

// vector operator *= : multiply
static inline Vec4f&
operator*=(Vec4f& a, Vec4f const b)
{
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
static inline Vec4f
operator/(Vec4f const a, Vec4f const b)
{
#if defined(__aarch64__)
  return vdivq_f32(a, b);
#else
  float32x4_t recip0 = vrecpeq_f32((b));
  float32x4_t recip1 = vmulq_f32(recip0, vrecpsq_f32(recip0, (b)));
  return vmulq_f32((a), recip1);
#endif
}

// vector operator / : divide vector and scalar
static inline Vec4f
operator/(Vec4f const a, float b)
{
  return a / Vec4f(b);
}

static inline Vec4f
operator/(float a, Vec4f const b)
{
  return Vec4f(a) / b;
}

// vector operator /= : divide
static inline Vec4f&
operator/=(Vec4f& a, Vec4f const b)
{
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec4fb
operator==(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_u32(vceqq_f32(a, b));
}

// vector operator != : returns true for elements for which a != b
static inline Vec4fb
operator!=(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a, b)));
}

// vector operator < : returns true for elements for which a < b
static inline Vec4fb
operator<(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_u32(vcltq_f32(a, b));
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec4fb
operator<=(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_u32(vcleq_f32(a, b));
}

// vector operator > : returns true for elements for which a > b
static inline Vec4fb
operator>(Vec4f const a, Vec4f const b)
{
  return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec4fb
operator>=(Vec4f const a, Vec4f const b)
{
  return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec4f
operator&(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_s32(
    vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

// vector operator &= : bitwise and
static inline Vec4f&
operator&=(Vec4f& a, Vec4f const b)
{
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline Vec4f
operator|(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_s32(
    vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

// vector operator |= : bitwise or
static inline Vec4f&
operator|=(Vec4f& a, Vec4f const b)
{
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline Vec4f
operator^(Vec4f const a, Vec4f const b)
{
  return vreinterpretq_f32_s32(
    veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

// vector operator ^= : bitwise xor
static inline Vec4f&
operator^=(Vec4f& a, Vec4f const b)
{
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec4fb
operator!(Vec4f const a)
{
  return a == Vec4f(0.0f);
}

/*****************************************************************************
 *
 *          Functions for Vec4f
 *
 *****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec4f
select(Vec4fb const s, Vec4f const a, Vec4f const b)
{
  return selectf(s, a, b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i])
// : a[i]
static inline Vec4f
if_add(Vec4fb const f, Vec4f const a, Vec4f const b)
{
  return a + (Vec4f(f) & b);
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] -
// b[i]) : a[i]
static inline Vec4f
if_sub(Vec4fb const f, Vec4f const a, Vec4f const b)
{
  return a - (Vec4f(f) & b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] *
// b[i]) : a[i]
static inline Vec4f
if_mul(Vec4fb const f, Vec4f const a, Vec4f const b)
{
  return a * select(f, b, 1.f);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] /
// b[i]) : a[i]
static inline Vec4f
if_div(Vec4fb const f, Vec4f const a, Vec4f const b)
{
  return a / select(f, b, 1.f);
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec4f
sign_combine(Vec4f const a, Vec4f const b)
{
  return a ^ (b & Vec4f(-0.0f));
}

// General arithmetic functions, etc.

// function max: a > b ? a : b
static inline Vec4f
max(Vec4f const a, Vec4f const b)
{
#if SSE2NEON_PRECISE_MINMAX
  return vbslq_f32(vcltq_f32(b, a), a, b);
#else
  return vmaxq_f32(a, b);
#endif
}

// function min: a < b ? a : b
static inline Vec4f
min(Vec4f const a, Vec4f const b)
{
#if SSE2NEON_PRECISE_MINMAX
  return vbslq_f32(vcltq_f32(a, b), a, b);
#else
  return vminq_f32(a, b);
#endif
}

// function abs: absolute value
static inline Vec4f
abs(Vec4f const a)
{
  int32x4_t mask = _mm_set1_epi32(0x7FFFFFFF);
  return vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(a), mask));
}

// function sqrt: square root
static inline Vec4f
sqrt(Vec4f const a)
{
#if defined(__aarch64__)
  return vsqrtq_f32((a));
#else
  float32x4_t recipsq = vrsqrteq_f32((a));
  float32x4_t sq = vrecpeq_f32(recipsq);
  return (sq);
#endif
}

// function square: a * a
static inline Vec4f
square(Vec4f const a)
{
  return a * a;
}

static inline Vec4f
round(Vec4f const a)
{
  return _mm_round_ps<8>(a);
}

// function truncate: round towards zero. (result as float vector)
static inline Vec4f
truncate(Vec4f const a)
{
  return _mm_round_ps<3 + 8>(a);
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec4f
floor(Vec4f const a)
{
  return _mm_round_ps<1 + 8>(a);
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec4f
ceil(Vec4f const a)
{
  return _mm_round_ps<2 + 8>(a);
}

// Approximate math functions

// approximate reciprocal (Faster than 1.f / a. relative accuracy better than
// 2^-11)
static inline Vec4f
approx_recipr(Vec4f const a)
{
#if defined(__aarch64__)
  return vdivq_f32(vdupq_n_f32(1.0f), a);
#else
  float32x4_t recip = vrecpeq_f32(a);
  recip = vmulq_f32(recip, vrecpsq_f32(recip, a));
  return recip;
#endif
}

// approximate reciprocal squareroot (Faster than 1.f / sqrt(a). Relative
// accuracy better than 2^-11)
static inline Vec4f
approx_rsqrt(Vec4f const a)
{
  return vrsqrteq_f32(a);
}

// Fused multiply and add functions

// Multiply and add
static inline Vec4f
mul_add(Vec4f const a, Vec4f const b, Vec4f const c)
{
#if defined(__aarch64__)
  return vfmaq_f32(c, b, a);
#else
  return a * b + c;
#endif
}

// Multiply and subtract
static inline Vec4f
mul_sub(Vec4f const a, Vec4f const b, Vec4f const c)
{
  return mul_add(a, b, -c);
}

// Multiply and inverse subtract
static inline Vec4f
nmul_add(Vec4f const a, Vec4f const b, Vec4f const c)
{
  return -mul_sub(a, b, c);
}

// change signs on vectors Vec4f
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for
// no change
template<int i0, int i1, int i2, int i3>
static inline Vec4f
change_sign(Vec4f const a)
{
  if ((i0 | i1 | i2 | i3) == 0)
    return a;
  auto mask = constant4ui < i0 ? 0x80000000 : 0, i1 ? 0x80000000 : 0,
       i2 ? 0x80000000 : 0, i3 ? 0x80000000 : 0 > ();
  return vreinterpretq_f32_s32(veorq_s32(
    vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(_mm_castsi128_ps(mask))));
}

#if defined(__aarch64__)

/*****************************************************************************
 *
 *          Operators for Vec2d
 *
 *****************************************************************************/

// vector operator + : add element by element
static inline Vec2d
operator+(Vec2d const a, Vec2d const b)
{
  return vaddq_f64(a, b);
}

// vector operator + : add vector and scalar
static inline Vec2d
operator+(Vec2d const a, double b)
{
  return a + Vec2d(b);
}
static inline Vec2d
operator+(double a, Vec2d const b)
{
  return Vec2d(a) + b;
}

// vector operator += : add
static inline Vec2d&
operator+=(Vec2d& a, Vec2d const b)
{
  a = a + b;
  return a;
}

// postfix operator ++
static inline Vec2d
operator++(Vec2d& a, int)
{
  Vec2d a0 = a;
  a = a + 1.0;
  return a0;
}

// prefix operator ++
static inline Vec2d&
operator++(Vec2d& a)
{
  a = a + 1.0;
  return a;
}

// vector operator - : subtract element by element
static inline Vec2d
operator-(Vec2d const a, Vec2d const b)
{
  return vsubq_f64(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec2d
operator-(Vec2d const a, double b)
{
  return a - Vec2d(b);
}
static inline Vec2d
operator-(double a, Vec2d const b)
{
  return Vec2d(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec2d
operator-(Vec2d const a)
{
  return _mm_xor_pd(
    a, vreinterpretq_f64_s32(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000)));
}

// vector operator -= : subtract
static inline Vec2d&
operator-=(Vec2d& a, Vec2d const b)
{
  a = a - b;
  return a;
}

// postfix operator --
static inline Vec2d
operator--(Vec2d& a, int)
{
  Vec2d a0 = a;
  a = a - 1.0;
  return a0;
}

// prefix operator --
static inline Vec2d&
operator--(Vec2d& a)
{
  a = a - 1.0;
  return a;
}

// vector operator * : multiply element by element
static inline Vec2d
operator*(Vec2d const a, Vec2d const b)
{
  return vmulq_f64(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec2d
operator*(Vec2d const a, double b)
{
  return a * Vec2d(b);
}
static inline Vec2d
operator*(double a, Vec2d const b)
{
  return Vec2d(a) * b;
}

// vector operator *= : multiply
static inline Vec2d&
operator*=(Vec2d& a, Vec2d const b)
{
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
static inline Vec2d
operator/(Vec2d const a, Vec2d const b)
{
  return vdivq_f64(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec2d
operator/(Vec2d const a, double b)
{
  return a / Vec2d(b);
}
static inline Vec2d
operator/(double a, Vec2d const b)
{
  return Vec2d(a) / b;
}

// vector operator /= : divide
static inline Vec2d&
operator/=(Vec2d& a, Vec2d const b)
{
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec2db
operator==(Vec2d const a, Vec2d const b)
{
  return vreinterpretq_f64_u64(vceqq_f64(a, b));
}

// vector operator != : returns true for elements for which a != b
static inline Vec2db
operator!=(Vec2d const a, Vec2d const b)
{
  return vreinterpretq_f64_u32(
    vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(a, b))));
}

// vector operator < : returns true for elements for which a < b
static inline Vec2db
operator<(Vec2d const a, Vec2d const b)
{
  return vreinterpretq_f64_u64(vcltq_f64(a, b));
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec2db
operator<=(Vec2d const a, Vec2d const b)
{
  return vreinterpretq_f64_u64(vcleq_f64(a, b));
}

// vector operator > : returns true for elements for which a > b
static inline Vec2db
operator>(Vec2d const a, Vec2d const b)
{
  return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec2db
operator>=(Vec2d const a, Vec2d const b)
{
  return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec2d
operator&(Vec2d const a, Vec2d const b)
{
  return vreinterpretq_f64_s64(
    vandq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)));
}

// vector operator &= : bitwise and
static inline Vec2d&
operator&=(Vec2d& a, Vec2d const b)
{
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline Vec2d
operator|(Vec2d const a, Vec2d const b)
{

  return vreinterpretq_f64_s64(
    vorrq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)));
}

// vector operator |= : bitwise or
static inline Vec2d&
operator|=(Vec2d& a, Vec2d const b)
{
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline Vec2d
operator^(Vec2d const a, Vec2d const b)
{
  return _mm_xor_pd(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec2d&
operator^=(Vec2d& a, Vec2d const b)
{
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec2db
operator!(Vec2d const a)
{
  return a == Vec2d(0.0);
}

/*****************************************************************************
 *
 *          Functions for Vec2d
 *
 *****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true).
// No other values are allowed.
static inline Vec2d
select(Vec2db const s, Vec2d const a, Vec2d const b)
{
  return vbslq_f64(vreinterpretq_u64_f64(s), a, b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i])
// : a[i]
static inline Vec2d
if_add(Vec2db const f, Vec2d const a, Vec2d const b)
{
  return a + (Vec2d(f) & b);
}

// Conditional subtract
static inline Vec2d
if_sub(Vec2db const f, Vec2d const a, Vec2d const b)
{
  return a - (Vec2d(f) & b);
}

// Conditional multiply
static inline Vec2d
if_mul(Vec2db const f, Vec2d const a, Vec2d const b)
{
  return a * select(f, b, 1.);
}

// Conditional divide
static inline Vec2d
if_div(Vec2db const f, Vec2d const a, Vec2d const b)
{
  return a / select(f, b, 1.);
}

// Sign functions

// change signs on vectors Vec2d
// Each index i0 - i1 is 1 for changing sign on the corresponding element, 0 for
// no change
template<int i0, int i1>
static inline Vec2d
change_sign(Vec2d const a)
{
  if ((i0 | i1) == 0)
    return a;
  auto mask = constant4ui < 0, i0 ? 0x80000000 : 0, 0, i1 ? 0x80000000 : 0 > ();
  return _mm_xor_pd(a, _mm_castsi128_pd(mask)); // flip sign bits
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec2d
sign_combine(Vec2d const a, Vec2d const b)
{
  return a ^ (b & Vec2d(-0.0));
}

// General arithmetic functions, etc.

// function max: a > b ? a : b
static inline Vec2d
max(Vec2d const a, Vec2d const b)
{
#if SSE2NEON_PRECISE_MINMAX
  return vbslq_f64(vcltq_f64(b, a), a, b);
#else
  return vmaxq_f64(a, b);
#endif
}

// function min: a < b ? a : b
static inline Vec2d
min(Vec2d const a, Vec2d const b)
{
#if SSE2NEON_PRECISE_MINMAX
  return vbslq_f64(vcltq_f64(a, b), a, b);
#else
  return vminq_f64(a, b);
#endif
}

// function abs: absolute value
static inline Vec2d
abs(Vec2d const a)
{
  auto mask = _mm_setr_epi32(-1, 0x7FFFFFFF, -1, 0x7FFFFFFF);
  return vreinterpretq_f64_s64(
    vandq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_s32(mask)));
}

// function sqrt: square root
static inline Vec2d
sqrt(Vec2d const a)
{
  return vsqrtq_f64(a);
}

// function square: a * a
static inline Vec2d
square(Vec2d const a)
{
  return a * a;
}

// function round: round to nearest integer (even). (result as double vector)
static inline Vec2d
round(Vec2d const a)
{
  return _mm_round_pd<8>(a);
}

// function truncate: round towards zero. (result as double vector)
static inline Vec2d
truncate(Vec2d const a)
{
  return _mm_round_pd<3 + 8>(a);
}

// function floor: round towards minus infinity. (result as double vector)
static inline Vec2d
floor(Vec2d const a)
{
  return _mm_round_pd<1 + 8>(a);
}

// function ceil: round towards plus infinity. (result as double vector)
static inline Vec2d
ceil(Vec2d const a)
{
  return _mm_round_pd<2 + 8>(a);
}

// Multiply and add
static inline Vec2d
mul_add(Vec2d const a, Vec2d const b, Vec2d const c)
{
  return vfmaq_f64(c, b, a);
}

// Multiply and subtract
static inline Vec2d
mul_sub(Vec2d const a, Vec2d const b, Vec2d const c)
{
  return mul_add(a, b, -c);
}

// Multiply and inverse subtract
static inline Vec2d
nmul_add(Vec2d const a, Vec2d const b, Vec2d const c)
{
  return -mul_sub(a, b, c);
}

#endif // defined(__aarch64__)

//
/*****************************************************************************
 *
 *  NOT IMPLEMENTED, these are just here for compatibility with vectorclass
 *
 *****************************************************************************/
class Vec4d final
{
  Vec4d() = default;

public:
  static constexpr int size() { return 4; }
};

using Vec4db = Vec4d;

class Vec8f final
{
  Vec8f() = default;

public:
  static constexpr int size() { return 8; }
};

using Vec8fb = Vec8f;

class Vec8d final
{
  Vec8d() = default;

public:
  static constexpr int size() { return 8; }
};

using Vec8db = Vec8d;

class Vec16f final
{
  Vec16f() = default;

public:
  static constexpr int size() { return 16; }
};

using Vec16fb = Vec16f;

} // namespace avec