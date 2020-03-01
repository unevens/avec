/*
Copyright 2020 Dario Mambro

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
#include "Spline.hpp"

#define LOAD_SPLINE_STATE(spline, numActiveKnots, Vec, maxNumKnots)            \
  Scalar* spline##_knots = spline->getKnots();                                 \
  Vec spline##_x[maxNumKnots];                                                 \
  Vec spline##_y[maxNumKnots];                                                 \
  Vec spline##_t[maxNumKnots];                                                 \
  Vec spline##_s[maxNumKnots];                                                 \
  for (int n = 0; n < numActiveKnots; ++n) {                                   \
    spline##_x[n] = Vec().load_a(spline##_knots[n].x);                         \
    spline##_y[n] = Vec().load_a(spline##_knots[n].y);                         \
    spline##_t[n] = Vec().load_a(spline##_knots[n].t);                         \
    spline##_s[n] = Vec().load_a(spline##_knots[n].s);                         \
  }

#define LOAD_SPLINE_AUTOMATOR(automator, numActiveKnots, Vec, maxNumKnots)     \
  Vec const automator##_alpha =                                                \
    Vec().load_a(automator->getSmoothingAlpha()[0]);                           \
  Scalar* automator##_knots = automator->getKnots();                           \
  Vec automator##_x_a[maxNumKnots];                                            \
  Vec automator##_y_a[maxNumKnots];                                            \
  Vec automator##_t_a[maxNumKnots];                                            \
  Vec automator##_s_a[maxNumKnots];                                            \
  for (int n = 0; n < numActiveKnots; ++n) {                                   \
    automator##_x_a[n] = Vec().load_a(automator##_knots[n].x);                 \
    automator##_y_a[n] = Vec().load_a(automator##_knots[n].y);                 \
    automator##_t_a[n] = Vec().load_a(automator##_knots[n].t);                 \
    automator##_s_a[n] = Vec().load_a(automator##_knots[n].s);                 \
  }

#define STORE_SPLINE_STATE(spline, numActiveKnots)                             \
  for (int n = 0; n < numActiveKnots; ++n) {                                   \
    spline##_x[n].store_a(spline##_knots[n].x);                                \
    spline##_y[n].store_a(spline##_knots[n].y);                                \
    spline##_t[n].store_a(spline##_knots[n].t);                                \
    spline##_s[n].store_a(spline##_knots[n].s);                                \
  }

#define SPILINE_AUTOMATION(spline, automator, numActiveKnots, Vec)             \
  for (int n = 0; n < numActiveKnots; ++n) {                                   \
    spline##_x[n] = spline##_alpha * (spline##_x[n] - automator##_x_a[n]) +    \
                    automator##_x_a[n];                                        \
    spline##_y[n] = spline##_alpha * (spline##_y[n] - automator##_y_a[n]) +    \
                    automator##_y_a[n];                                        \
    spline##_t[n] = spline##_alpha * (spline##_t[n] - automator##_t_a[n]) +    \
                    automator##_t_a[n];                                        \
    spline##_s[n] = spline##_alpha * (spline##_s[n] - automator##_s_a[n]) +    \
                    automator##_s_a[n];                                        \
  }

#define LOAD_SPLINE_SYMMETRY(spline, Vec)                                      \
  auto const spline##_symm = Vec().load_a(spline->getIsSymmetric()) != 0.0;

#define COMPUTE_SPLINE(spline, numActiveKnots, Vec, in, out)                   \
  {                                                                            \
    Vec x0 = std::numeric_limits<float>::lowest();                             \
    Vec y0 = 0.f;                                                              \
    Vec t0 = 0.f;                                                              \
    Vec s0 = 0.f;                                                              \
                                                                               \
    Vec x1 = std::numeric_limits<float>::max();                                \
    Vec y1 = 0.f;                                                              \
    Vec t1 = 0.f;                                                              \
    Vec s1 = 0.f;                                                              \
                                                                               \
    Vec x_low = spline##_x[0];                                                 \
    Vec y_low = spline##_y[0];                                                 \
    Vec t_low = spline##_t[0];                                                 \
                                                                               \
    Vec x_high = spline##_x[0];                                                \
    Vec y_high = spline##_y[0];                                                \
    Vec t_high = spline##_t[0];                                                \
                                                                               \
    for (int n = 1; n < numActiveKnots; ++n) {                                 \
      auto const is_left = (in > spline##_x[n]) && (spline##_x[n] > x0);       \
      x0 = select(is_left, spline##_x[n], x0);                                 \
      y0 = select(is_left, spline##_y[n], y0);                                 \
      t0 = select(is_left, spline##_t[n], t0);                                 \
      s0 = select(is_left, spline##_s[n], s0);                                 \
                                                                               \
      auto const is_right = (in <= spline##_x[n]) && (spline##_x[n] < x1);     \
      x1 = select(is_right, spline##_x[n], x1);                                \
      y1 = select(is_right, spline##_y[n], y1);                                \
      t1 = select(is_right, spline##_t[n], t1);                                \
      s1 = select(is_right, spline##_s[n], s1);                                \
                                                                               \
      auto const is_lowest = spline##_x[n] < x_low;                            \
      x_low = select(is_lowest, spline##_x[n], x_low);                         \
      y_low = select(is_lowest, spline##_y[n], y_low);                         \
      t_low = select(is_lowest, spline##_t[n], t_low);                         \
                                                                               \
      auto const is_highest = spline##_x[n] > x_high;                          \
      x_high = select(is_highest, spline##_x[n], x_high);                      \
      y_high = select(is_highest, spline##_y[n], y_high);                      \
      t_high = select(is_highest, spline##_t[n], t_high);                      \
    }                                                                          \
                                                                               \
    auto const is_high = x1 == std::numeric_limits<float>::max();              \
    auto const is_low = x0 == std::numeric_limits<float>::lowest();            \
                                                                               \
    Vec const dx = max(x1 - x0, std::numeric_limits<float>::min());            \
    Vec const dy = y1 - y0;                                                    \
    Vec const a = t0 * dx - dy;                                                \
    Vec const b = -t1 * dx + dy;                                               \
    Vec const ix = 1.0 / dx;                                                   \
    Vec const m = dy * ix;                                                     \
    Vec const o = y0 - m * x0;                                                 \
                                                                               \
    Vec const j = (in - x0) * ix;                                              \
    Vec const k = 1.0 - j;                                                     \
                                                                               \
    Vec const hermite = k * y0 + j * y1 + j * k * (a * k + b * j);             \
    Vec const segment = m * in + o;                                            \
    Vec const smoothness = s1 + k * (s0 - s1);                                 \
    Vec const curve = segment + smoothness * (hermite - segment);              \
                                                                               \
    Vec const low = y_low + (in - x_low) * t_low;                              \
    Vec const high = y_high + (in - x_high) * t_high;                          \
                                                                               \
    out = select(is_high, high, select(is_low, low, curve));                   \
  }

#define COMPUTE_SPLINE_WITH_SYMMETRY(spline, numActiveKnots, Vec, in_, out)    \
  {                                                                            \
    Vec const in = select(spline##_symm, abs(in_), in_);                       \
    COMPUTE_SPLINE(spline, numActiveKnots, Vec, in, out);                      \
    out = select(spline##_symm, sign_combine(out, in_), out);                  \
  }

#define COMPUTE_SPLINE_WITH_DERIVATIVE(                                        \
  spline, numActiveKnots, Vec, in, out, delta)                                 \
  {                                                                            \
    Vec x0 = std::numeric_limits<float>::lowest();                             \
    Vec y0 = 0.f;                                                              \
    Vec t0 = 0.f;                                                              \
    Vec s0 = 0.f;                                                              \
                                                                               \
    Vec x1 = std::numeric_limits<float>::max();                                \
    Vec y1 = 0.f;                                                              \
    Vec t1 = 0.f;                                                              \
    Vec s1 = 0.f;                                                              \
                                                                               \
    Vec x_low = spline##_x[0];                                                 \
    Vec y_low = spline##_y[0];                                                 \
    Vec t_low = spline##_t[0];                                                 \
                                                                               \
    Vec x_high = spline##_x[0];                                                \
    Vec y_high = spline##_y[0];                                                \
    Vec t_high = spline##_t[0];                                                \
                                                                               \
    for (int n = 1; n < numActiveKnots; ++n) {                                 \
      auto const is_left = (in > spline##_x[n]) && (spline##_x[n] > x0);       \
      x0 = select(is_left, spline##_x[n], x0);                                 \
      y0 = select(is_left, spline##_y[n], y0);                                 \
      t0 = select(is_left, spline##_t[n], t0);                                 \
      s0 = select(is_left, spline##_s[n], s0);                                 \
                                                                               \
      auto const is_right = (in <= spline##_x[n]) && (spline##_x[n] < x1);     \
      x1 = select(is_right, spline##_x[n], x1);                                \
      y1 = select(is_right, spline##_y[n], y1);                                \
      t1 = select(is_right, spline##_t[n], t1);                                \
      s1 = select(is_right, spline##_s[n], s1);                                \
                                                                               \
      auto const is_lowest = spline##_x[n] < x_low;                            \
      x_low = select(is_lowest, spline##_x[n], x_low);                         \
      y_low = select(is_lowest, spline##_y[n], y_low);                         \
      t_low = select(is_lowest, spline##_t[n], t_low);                         \
                                                                               \
      auto const is_highest = spline##_x[n] > x_high;                          \
      x_high = select(is_highest, spline##_x[n], x_high);                      \
      y_high = select(is_highest, spline##_y[n], y_high);                      \
      t_high = select(is_highest, spline##_t[n], t_high);                      \
    }                                                                          \
                                                                               \
    auto const is_high = x1 == std::numeric_limits<float>::max();              \
    auto const is_low = x0 == std::numeric_limits<float>::lowest();            \
                                                                               \
    Vec const dx = max(x1 - x0, std::numeric_limits<float>::min());            \
    Vec const dy = y1 - y0;                                                    \
    Vec const a = t0 * dx - dy;                                                \
    Vec const b = -t1 * dx + dy;                                               \
    Vec const ix = 1.0 / dx;                                                   \
    Vec const m = dy * ix;                                                     \
    Vec const o = y0 - m * x0;                                                 \
                                                                               \
    Vec const j = (in - x0) * ix;                                              \
    Vec const k = 1.0 - j;                                                     \
                                                                               \
    Vec const akbj = a * k + b * j;                                            \
    Vec const hermite = k * y0 + j * y1 + j * k * akbj;                        \
    Vec const hermite_delta = ix * (dy + (k + j) * akbj + j * k * (b - a));    \
    Vec const segment = m * in + o;                                            \
    Vec const smoothness = s1 + k * (s0 - s1);                                 \
    Vec const curve = segment + smoothness * (hermite - segment);              \
    Vec const curve_delta = m + smoothness * (hermite_delta - m);              \
                                                                               \
    Vec const low = y_low + (in - x_low) * t_low;                              \
    Vec const high = y_high + (in - x_high) * t_high;                          \
                                                                               \
    out = select(is_high, high, select(is_low, low, curve));                   \
    delta = select(is_high, t_high, select(is_low, x_low, curve_delta));       \
  }

#define COMPUTE_SPLINE_WITH_SYMMETRY_WITH_DERIVATIVE(                          \
  spline, numActiveKnots, Vec, in_, out, delta)                                \
  {                                                                            \
    Vec const in = select(spline##_symm, abs(in_), in_);                       \
    COMPUTE_SPLINE_WITH_DERIVATIVE(                                            \
      spline, numActiveKnots, Vec, in, out, delta);                            \
    /*delta *= select(spline##_symm, sign_combine(1.0, in_), 1.0);*/           \
    out = select(spline##_symm, sign_combine(out, in_), out);                  \
    /*delta *= select(spline##_symm, sign_combine(1.0, in_), 1.0);*/           \
  }
