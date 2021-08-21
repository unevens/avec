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
#include "NeonMathFloat.hpp"
#include "NeonVec.hpp"
#include <utility>


inline Vec4f
sin(Vec4f const x)
{
  return avec::detail::sin_ps(x);
}

inline Vec4f
cos(Vec4f const x)
{
  return avec::detail::cos_ps(x);
}

inline Vec4f
log(Vec4f const x)
{
  return avec::detail::log_ps(x);
}

inline Vec4f
exp(Vec4f const x)
{
  return avec::detail::exp_ps(x);
}

inline std::pair<Vec4f, Vec4f>
sincos(Vec4f const x)
{
  avec::detail::v4sf s, c;
  avec::detail::sincos_ps(x, &s, &c);
  return { s, c };
}

inline Vec4f
tan(Vec4f const x)
{
  avec::detail::v4sf s, c;
  avec::detail::sincos_ps(x, &s, &c);
  return Vec4f(s) / Vec4f(c);
}

#if defined(__aarch64__)

inline Vec2d
sin(Vec2d const x)
{
  return avec::detail::sin_pd(x);
}

inline Vec2d
cos(Vec2d const x)
{
  return avec::detail::cos_pd(x);
}

inline Vec2d
log(Vec2d const x)
{
  return avec::detail::log_pd(x);
}

inline Vec2d
exp(Vec2d const x)
{
  return avec::detail::exp_pd(x);
}

inline std::pair<Vec2d, Vec2d>
sincos(Vec2d const x)
{
  avec::detail::v2sd s, c;
  avec::detail::sincos_pd(x, &s, &c);
  return { s, c };
}

inline Vec2d
tan(Vec2d const x)
{
  avec::detail::v2sd s, c;
  avec::detail::sincos_pd(x, &s, &c);
  return Vec2d(s) / Vec2d(c);
}

#endif

