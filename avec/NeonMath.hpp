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

namespace avec {

Vec4f
sin(Vec4f const x)
{
  return detail::sin_ps(x);
}
Vec4f
cos(Vec4f const x)
{
  return detail::cos_ps(x);
}
Vec4f
log(Vec4f const x)
{
  return detail::log_ps(x);
}
Vec4f
exp(Vec4f const x)
{
  return detail::exp_ps(x);
}

#if defined(__aarch64__)

Vec2d
sin(Vec2d const x)
{
  return detail::sin_pd(x);
}
Vec2d
cos(Vec2d const x)
{
  return detail::cos_pd(x);
}
Vec2d
log(Vec2d const x)
{
  return detail::log_pd(x);
}
Vec2d
exp(Vec2d const x)
{
  return detail::exp_pd(x);
}
#endif

} // namespace avec
