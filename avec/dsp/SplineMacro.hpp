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

#define LOAD_SPLINE_STATE(spline, numActiveNodes, Vec, maxNumNodes)            \
  Vec const spline##_alpha = spline->GetSmoothingAlpha()[0];                   \
  Vec spline##_xs[maxNumNodes];                                                \
  Vec spline##_ys[maxNumNodes];                                                \
  Vec spline##_ts[maxNumNodes];                                                \
  Vec spline##_ss[maxNumNodes];                                                \
  Vec spline##_xt[maxNumNodes];                                                \
  Vec spline##_yt[maxNumNodes];                                                \
  Vec spline##_tt[maxNumNodes];                                                \
  Vec spline##_st[maxNumNodes];                                                \
  for (int n = 0; n < numActiveNodes; ++n) {                                   \
    spline##_xs[n] = Vec().load_a(spline->GetNodes()[n].state.x);              \
    spline##_ys[n] = Vec().load_a(spline->GetNodes()[n].state.y);              \
    spline##_ts[n] = Vec().load_a(spline->GetNodes()[n].state.t);              \
    spline##_ss[n] = Vec().load_a(spline->GetNodes()[n].state.s);              \
    spline##_xt[n] = Vec().load_a(spline->GetNodes()[n].target.x);             \
    spline##_yt[n] = Vec().load_a(spline->GetNodes()[n].target.y);             \
    spline##_tt[n] = Vec().load_a(spline->GetNodes()[n].target.t);             \
    spline##_st[n] = Vec().load_a(spline->GetNodes()[n].target.s);             \
  }

#define STORE_SPLINE_STATE(spline, numActiveNodes)                             \
  for (int n = 0; n < numActiveNodes; ++n) {                                   \
    spline##_xs[n].store_a(spline->GetNodes()[n].state.x);                     \
    spline##_ys[n].store_a(spline->GetNodes()[n].state.y);                     \
    spline##_ts[n].store_a(spline->GetNodes()[n].state.t);                     \
    spline##_ss[n].store_a(spline->GetNodes()[n].state.s);                     \
  }

#define SPILINE_AUTOMATION(spline, numActiveNodes, Vec)                        \
  for (int n = 0; n < numActiveNodes; ++n) {                                   \
    spline##_xs[n] =                                                           \
      spline##_alpha * (spline##_xs[n] - spline##_xt[n]) + spline##_xt[n];     \
    spline##_ys[n] =                                                           \
      spline##_alpha * (spline##_ys[n] - spline##_yt[n]) + spline##_yt[n];     \
    spline##_ts[n] =                                                           \
      spline##_alpha * (spline##_ts[n] - spline##_tt[n]) + spline##_tt[n];     \
    spline##_ss[n] =                                                           \
      spline##_alpha * (spline##_ss[n] - spline##_st[n]) + spline##_st[n];     \
  }

#define COMPUTE_SPLINE(spline, numActiveNodes, Vec, in, out)                   \
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
    Vec x_low = spline##_xs[0];                                                \
    Vec y_low = spline##_ys[0];                                                \
    Vec t_low = spline##_ts[0];                                                \
                                                                               \
    Vec x_high = spline##_xs[0];                                               \
    Vec y_high = spline##_ys[0];                                               \
    Vec t_high = spline##_ts[0];                                               \
                                                                               \
    for (int n = 1; n < numActiveNodes; ++n) {                                 \
      auto const is_left = (in > spline##_xs[n]) && (spline##_xs[n] > x0);     \
      x0 = select(is_left, spline##_xs[n], x0);                                \
      y0 = select(is_left, spline##_ys[n], y0);                                \
      t0 = select(is_left, spline##_ts[n], t0);                                \
      s0 = select(is_left, spline##_ss[n], s0);                                \
                                                                               \
      auto const is_right = (in <= spline##_xs[n]) && (spline##_xs[n] < x1);   \
      x1 = select(is_right, spline##_xs[n], x1);                               \
      y1 = select(is_right, spline##_ys[n], y1);                               \
      t1 = select(is_right, spline##_ts[n], t1);                               \
      s1 = select(is_right, spline##_ss[n], s1);                               \
                                                                               \
      auto const is_lowest = spline##_xs[n] < x_low;                           \
      x_low = select(is_lowest, spline##_xs[n], x_low);                        \
      y_low = select(is_lowest, spline##_ys[n], y_low);                        \
      t_low = select(is_lowest, spline##_ts[n], t_low);                        \
                                                                               \
      auto const is_highest = spline##_xs[n] > x_high;                         \
      x_high = select(is_highest, spline##_xs[n], x_high);                     \
      y_high = select(is_highest, spline##_ys[n], y_high);                     \
      t_high = select(is_highest, spline##_ts[n], t_high);                     \
    }                                                                          \
                                                                               \
    auto const is_high = x1 == std::numeric_limits<float>::max();              \
    auto const is_low = x0 == std::numeric_limits<float>::lowest();            \
                                                                               \
    Vec const dx = max(x1 - x0, FLT_MIN);                                      \
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

#define COMPUTE_SPLINE_WITH_DERIVATIVE(                                        \
  spline, numActiveNodes, Vec, in, out, delta)                                 \
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
    Vec x_low = spline##_xs[0];                                                \
    Vec y_low = spline##_ys[0];                                                \
    Vec t_low = spline##_ts[0];                                                \
                                                                               \
    Vec x_high = spline##_xs[0];                                               \
    Vec y_high = spline##_ys[0];                                               \
    Vec t_high = spline##_ts[0];                                               \
                                                                               \
    for (int n = 1; n < numActiveNodes; ++n) {                                 \
      auto const is_left = (in > spline##_xs[n]) && (spline##_xs[n] > x0);     \
      x0 = select(is_left, spline##_xs[n], x0);                                \
      y0 = select(is_left, spline##_ys[n], y0);                                \
      t0 = select(is_left, spline##_ts[n], t0);                                \
      s0 = select(is_left, spline##_ss[n], s0);                                \
                                                                               \
      auto const is_right = (in <= spline##_xs[n]) && (spline##_xs[n] < x1);   \
      x1 = select(is_right, spline##_xs[n], x1);                               \
      y1 = select(is_right, spline##_ys[n], y1);                               \
      t1 = select(is_right, spline##_ts[n], t1);                               \
      s1 = select(is_right, spline##_ss[n], s1);                               \
                                                                               \
      auto const is_lowest = spline##_xs[n] < x_low;                           \
      x_low = select(is_lowest, spline##_xs[n], x_low);                        \
      y_low = select(is_lowest, spline##_ys[n], y_low);                        \
      t_low = select(is_lowest, spline##_ts[n], t_low);                        \
                                                                               \
      auto const is_highest = spline##_xs[n] > x_high;                         \
      x_high = select(is_highest, spline##_xs[n], x_high);                     \
      y_high = select(is_highest, spline##_ys[n], y_high);                     \
      t_high = select(is_highest, spline##_ts[n], t_high);                     \
    }                                                                          \
                                                                               \
    auto const is_high = x1 == std::numeric_limits<float>::max();              \
    auto const is_low = x0 == std::numeric_limits<float>::lowest();            \
                                                                               \
    Vec const dx = max(x1 - x0, FLT_MIN);                                      \
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

#define LOAD_WAVESHAPER_STATE(shaper, numActiveNodes, Vec, maxNumNodes)        \
  LOAD_SPLINE_STATE(shaper, numActiveNodes, Vec, maxNumNodes);                 \
  Vec shaper##_ds = Vec().load_a(shaper->GetDcState());                        \
  Vec shaper##_dt = Vec().load_a(shaper->GetDcTarget());                       \
  Vec shaper##_ws = Vec().load_a(shaper->GetWetState());                       \
  Vec shaper##_wt = Vec().load_a(shaper->GetWetTarget());                      \
  Vec shaper##_hi = Vec().load_a(shaper->GetHighPassIn());                     \
  Vec shaper##_ho = Vec().load_a(shaper->GetHighPassOut());                    \
  Vec shaper##_ha = Vec().load_a(shaper->GetHighPassAlpha());                  \
  auto const shaper##_symm = Vec().load_a(shaper->isSymmetric) != 0.0;

#define STORE_WAVESHAPER_STATE(shaper, numActiveNodes)                         \
  STORE_SPLINE_STATE(shaper, numActiveNodes)                                   \
  shaper##_ds.store_a(shaper->GetDcState());                                   \
  shaper##_ws.store_a(shaper->GetWetState());                                  \
  shaper##_hi.store_a(shaper->GetHighPassIn());                                \
  shaper##_ho.store_a(shaper->GetHighPassOut());

#define WAVESHAPER_AUTOMATION(shaper, numActiveNodes, Vec)                     \
  SPILINE_AUTOMATION(shaper, numActiveNodes, Vec);                             \
  shaper##_ws = shaper##_alpha * (shaper##_ws - shaper##_wt) + shaper##_wt;    \
  shaper##_ds = shaper##_alpha * (shaper##_ds - shaper##_dt) + shaper##_dt;

#define COMPUTE_WAVESHAPER(shaper, numActiveNodes, Vec, in_, out)              \
  {                                                                            \
    Vec const with_dc = in_ + shaper##_ds;                                     \
    Vec const in = select(shaper##_symm, abs(with_dc), with_dc);               \
    COMPUTE_SPLINE(shaper, numActiveNodes, Vec, in, out);                      \
    out = select(shaper##_symm, sign_combine(out, with_dc), out);              \
    shaper##_ho = shaper##_ha * (shaper##_ho + out - shaper##_hi);             \
    shaper##_hi = out;                                                         \
    out = shaper##_ho;                                                         \
    out = in_ + shaper##_ws * (out - in_);                                     \
  }

#define COMPUTE_WAVESHAPER_WITH_DERIVATIVE(                                    \
  shaper, numActiveNodes, Vec, in, out, delta)                                 \
  {                                                                            \
    Vec const with_dc = in_ + shaper##_ds;                                     \
    Vec const in = select(shaper##_symm, abs(with_dc), with_dc);               \
    COMPUTE_SPLINE_WITH_DERIVATIVE(                                            \
      shaper, numActiveNodes, Vec, in, out, delta);                            \
    /*delta *= select(shaper##_symm, sign_combine(1.0, with_dc), 1.0);*/       \
    out = select(shaper##_symm, sign_combine(out, with_dc), out);              \
    /*delta *= select(shaper##_symm, sign_combine(1.0, with_dc), 1.0);*/       \
    shaper##_ho = shaper##_ha * (shaper##_ho + out - shaper##_hi);             \
    shaper##_hi = out;                                                         \
    out = shaper##_ho;                                                         \
    delta *= shaper##_ha;                                                      \
    out = in_ + shaper##_ws * (out - in_);                                     \
    delta = 1.0 + shaper##_ws * (delta - 1.0);                                 \
  }
