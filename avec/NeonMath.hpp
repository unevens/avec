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

#pragma once
#include "NeonMathDouble.hpp"
#include "NeonMathDoubleVCL.hpp"
#include "NeonMathFloat.hpp"
#include "NeonVec.hpp"
#include <cmath>
#include <utility>

#if defined(AVEC_USE_NEON_PD) && defined(AVEC_USE_SCALAR_PD)
#error "AVEC_USE_NEON_PD and AVEC_USE_SCALAR_PD are mutually exclusive"
#endif

namespace avec {

inline Vec4f
sin(Vec4f const x)
{
  return detail::sin_ps(x);
}

inline Vec4f
cos(Vec4f const x)
{
  return detail::cos_ps(x);
}

inline Vec4f
log(Vec4f const x)
{
  return detail::log_ps(x);
}

inline Vec4f
exp(Vec4f const x)
{
  return detail::exp_ps(x);
}

inline std::pair<Vec4f, Vec4f>
sincos(Vec4f const x)
{
  detail::v4sf s, c;
  detail::sincos_ps(x, &s, &c);
  return { s, c };
}

inline Vec4f
tan(Vec4f const x)
{
  detail::v4sf s, c;
  detail::sincos_ps(x, &s, &c);
  return Vec4f(s) / Vec4f(c);
}

#if defined(__aarch64__)

// Three implementations of the Vec2d math overloads:
//
//   default:              vectorclass-derived detail::vcl_*_pd intrinsics
//                         from NeonMathDoubleVCL.hpp — vectorized, full
//                         f64 precision.
//   AVEC_USE_NEON_PD:     Pommier-style detail::*_pd intrinsics from
//                         NeonMathDouble.hpp — vectorized, ~f32-grade
//                         precision (~7 decimal digits). See the TODO
//                         at the top of that file.
//   AVEC_USE_SCALAR_PD:   per-lane scalar libm — full f64 precision,
//                         ~2× the per-call cost of vectorized.

#if defined(AVEC_USE_NEON_PD)

inline Vec2d sin(Vec2d const x) { return detail::sin_pd(x); }
inline Vec2d cos(Vec2d const x) { return detail::cos_pd(x); }
inline Vec2d log(Vec2d const x) { return detail::log_pd(x); }
inline Vec2d exp(Vec2d const x) { return detail::exp_pd(x); }

inline std::pair<Vec2d, Vec2d>
sincos(Vec2d const x)
{
  detail::v2sd s, c;
  detail::sincos_pd(x, &s, &c);
  return { s, c };
}

inline Vec2d
tan(Vec2d const x)
{
  detail::v2sd s, c;
  detail::sincos_pd(x, &s, &c);
  return Vec2d(s) / Vec2d(c);
}

#elif defined(AVEC_USE_SCALAR_PD)

inline Vec2d
sin(Vec2d const x)
{
  float64x2_t v = x;
  double const a = std::sin(vgetq_lane_f64(v, 0));
  double const b = std::sin(vgetq_lane_f64(v, 1));
  v = vsetq_lane_f64(a, v, 0);
  v = vsetq_lane_f64(b, v, 1);
  return v;
}

inline Vec2d
cos(Vec2d const x)
{
  float64x2_t v = x;
  double const a = std::cos(vgetq_lane_f64(v, 0));
  double const b = std::cos(vgetq_lane_f64(v, 1));
  v = vsetq_lane_f64(a, v, 0);
  v = vsetq_lane_f64(b, v, 1);
  return v;
}

inline Vec2d
log(Vec2d const x)
{
  float64x2_t v = x;
  double const a = std::log(vgetq_lane_f64(v, 0));
  double const b = std::log(vgetq_lane_f64(v, 1));
  v = vsetq_lane_f64(a, v, 0);
  v = vsetq_lane_f64(b, v, 1);
  return v;
}

inline Vec2d
exp(Vec2d const x)
{
  float64x2_t v = x;
  double const a = std::exp(vgetq_lane_f64(v, 0));
  double const b = std::exp(vgetq_lane_f64(v, 1));
  v = vsetq_lane_f64(a, v, 0);
  v = vsetq_lane_f64(b, v, 1);
  return v;
}

inline std::pair<Vec2d, Vec2d>
sincos(Vec2d const x)
{
  return { sin(x), cos(x) };
}

inline Vec2d
tan(Vec2d const x)
{
  float64x2_t v = x;
  double const a = std::tan(vgetq_lane_f64(v, 0));
  double const b = std::tan(vgetq_lane_f64(v, 1));
  v = vsetq_lane_f64(a, v, 0);
  v = vsetq_lane_f64(b, v, 1);
  return v;
}

#else // default: vectorclass-derived full f64

inline Vec2d sin(Vec2d const x) { return detail::vcl_sin_pd(x); }
inline Vec2d cos(Vec2d const x) { return detail::vcl_cos_pd(x); }
inline Vec2d log(Vec2d const x) { return detail::vcl_log_pd(x); }
inline Vec2d exp(Vec2d const x) { return detail::vcl_exp_pd(x); }
inline Vec2d tan(Vec2d const x) { return detail::vcl_tan_pd(x); }

inline std::pair<Vec2d, Vec2d>
sincos(Vec2d const x)
{
  float64x2_t s, c;
  detail::vcl_sincos_pd(x, &s, &c);
  return { s, c };
}

#endif // AVEC_USE_NEON_PD / AVEC_USE_SCALAR_PD

#endif

} // namespace avec

// Mirror Agner Fog's vectorclass convention: math overloads for Vec types
// must be findable by unqualified lookup, since the Vec types themselves
// live in the global namespace and ADL therefore won't reach `namespace avec`.
using avec::cos;
using avec::exp;
using avec::log;
using avec::sin;
using avec::sincos;
using avec::tan;
