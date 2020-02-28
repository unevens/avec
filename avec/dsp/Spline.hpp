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
#include "avec/Avec.hpp"

namespace avec {

template<class Vec>
struct SplineInterface
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;

  struct Knot final
  {
    Scalar x[Vec::size()];
    Scalar y[Vec::size()];
    Scalar t[Vec::size()];
    Scalar s[Vec::size()];
  };

  struct AutomatableKnot final
  {
    Knot state;
    Knot target;
  };

  virtual AutomatableKnot* getKnots() = 0;

  virtual int getNumKnots() = 0;

  virtual Scalar* getSmoothingAlpha() = 0;

  virtual void setSmoothingFrequency(Scalar frequency) = 0;

  virtual void processBlock(VecBuffer<Vec> const& input,
                            VecBuffer<Vec>& output) = 0;

  virtual void reset() = 0;

  virtual ~SplineInterface() {}
};

template<class Vec>
struct WaveShaperInterface : public SplineInterface<Vec>
{
  using Scalar = typename SplineInterface<Vec>::Scalar;

  virtual void setHighPassFrequency(Scalar frequency) = 0;

  virtual void setHighPassFrequency(Scalar frequency, int channel) = 0;

  virtual void setDc(Scalar frequency) = 0;

  virtual void setDc(Scalar frequency, int channel) = 0;

  virtual void setWet(Scalar frequency) = 0;

  virtual void setWet(Scalar frequency, int channel) = 0;

  virtual void setIsSymmetric(bool isSymmetric) = 0;

  virtual void setIsSymmetric(Scalar isSymmetric, int channel) = 0;

  virtual Scalar* getHighPassAlpha() = 0;

  virtual Scalar* getDcState() = 0;
  virtual Scalar* getDcTarget() = 0;

  virtual Scalar* getWetState() = 0;
  virtual Scalar* getWetTarget() = 0;

  virtual Scalar* getHighPassIn() = 0;
  virtual Scalar* getHighPassOut() = 0;

  virtual Scalar* getIsSymmetric() = 0;
};

template<class Vec, int numKnots_>
struct Spline final : public SplineInterface<Vec>
{
  static constexpr int numKnots = numKnots_;

  using Interface = SplineInterface<Vec>;
  using Scalar = typename Interface::Scalar;
  using AutomatableKnot = typename Interface::AutomatableKnot;

  struct Data final
  {
    Scalar smoothingAlpha[Vec::size()];
    AutomatableKnot knots[numKnots];
  };

  aligned_ptr<Data> data;

  AutomatableKnot* getKnots() override { return data->knots; }

  int getNumKnots() override { return numKnots; }

  void setSmoothingFrequency(Scalar frequency) override
  {
    std::fill_n(data->smoothingAlpha, Vec::size(), exp(-frequency));
  }

  Scalar* getSmoothingAlpha() override { return data->smoothingAlpha; };

  virtual void reset() override
  {
    for (auto& knot : data->knots) {
      std::copy(&knot.target, &knot.target + 1, &knot.state);
    }
  }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output) override;

  Spline()
    : data(avec::Aligned<Data>::make())
  {
    std::fill_n(data->smoothingAlpha, sizeof(Data) / sizeof(Scalar), 0.0);
  }
};

template<class Vec, int numKnots_>
struct WaveShaper final : public WaveShaperInterface<Vec>
{
  static constexpr int numKnots = numKnots_;
  using Interface = WaveShaperInterface<Vec>;
  using Scalar = typename Interface::Scalar;
  using AutomatableKnot = typename Interface::AutomatableKnot;

  struct Settings final
  {
    Scalar dc[Vec::size()];
    Scalar wet[Vec::size()];
  };

  struct HighPass final
  {
    Scalar in[Vec::size()];
    Scalar out[Vec::size()];
    Scalar alpha[Vec::size()];
  };

  struct Data final
  {
    Scalar smoothingAlpha[Vec::size()];
    AutomatableKnot knots[numKnots];
    Scalar isSymmetric[Vec::size()];
    Settings settingsTarget;
    Settings settingsState;
    HighPass highPass;
  };

  aligned_ptr<Data> data;

  AutomatableKnot* getKnots() override { return data->knots; }

  int getNumKnots() override { return numKnots; }

  void setSmoothingFrequency(Scalar frequency) override
  {
    std::fill_n(data->smoothingAlpha, Vec::size(), exp(-frequency));
  }

  void setHighPassFrequency(Scalar frequency) override
  {
    std::fill_n(data->highPass.alpha, Vec::size(), exp(-frequency));
  }

  void setHighPassFrequency(Scalar frequency, int channel) override
  {
    data->highPass.alpha[channel] = exp(-frequency);
  }

  void setDc(Scalar dc) override
  {
    std::fill_n(data->settingsTarget.dc, Vec::size(), dc);
  }

  void setDc(Scalar dc, int channel) override
  {
    data->settingsTarget.dc[channel] = dc;
  }

  void setWet(Scalar wet) override
  {
    std::fill_n(data->settingsTarget.wet, Vec::size(), wet);
  }

  void setWet(Scalar wet, int channel) override
  {
    data->settingsTarget.wet[channel] = wet;
  }

  void setIsSymmetric(bool isSymmetric) override
  {
    std::fill_n(data->isSymmetric, Vec::size(), isSymmetric ? 1.0 : 0.0);
  }

  void setIsSymmetric(Scalar isSymmetric, int channel) override
  {
    data->isSymmetric[channel] = isSymmetric;
  }

  Scalar* getSmoothingAlpha() override { return data->smoothingAlpha; };

  Scalar* getHighPassAlpha() override { return data->highPass.alpha; }

  Scalar* getDcState() override { return data->settingsState.dc; }
  Scalar* getDcTarget() override { return data->settingsTarget.dc; }

  Scalar* getWetState() override { return data->settingsState.wet; }
  Scalar* getWetTarget() override { return data->settingsTarget.wet; }

  Scalar* getHighPassIn() override { return data->highPass.in; }
  Scalar* getHighPassOut() override { return data->highPass.out; }

  Scalar* getIsSymmetric() override { return data->isSymmetric; }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output) override;

  void reset() override
  {
    for (auto& knot : data->knots) {
      std::copy(&knot.target, &knot.target + 1, &knot.state);
    }
    std::copy(&data->settingsTarget.dc[0],
              &data->settingsTarget.dc[0] + 2 * Vec::size(),
              &data->settingsState.dc[0]);
    std::fill_n(&data->highPass.in[0], 2 * Vec::size(), 0.0);
  }

  WaveShaper()
    : data(avec::Aligned<Data>::make())
  {
    std::fill_n(data->smoothingAlpha, sizeof(Data) / sizeof(Scalar), 0.0);
  }
};

template<template<class, int> class SplineClass, class Vec>
struct SplineHolder final
{
  using SplineInterface = typename SplineClass<Vec, 1>::Interface;

  std::vector<std::unique_ptr<SplineInterface>> splines;

  SplineInterface* getSpline(int numKnots)
  {
    int index = numKnots - 1;
    if (index < splines.size()) {
      return splines[index].get();
    }
    return nullptr;
  }

  template<int numKnots>
  void initialize();

  void reset()
  {
    for (auto& s : splines) {
      s->reset();
    }
  }

  template<int numKnots>
  static SplineHolder make();
};

template<template<class, int> class SplineClass, class Vec, int maxNumKnots>
struct SplineFactory
{
  static void initialize(SplineHolder<SplineClass, Vec>& holder)
  {
    using Interface = typename SplineClass<Vec, maxNumKnots>::Interface;

    holder.splines.resize(
      std::max(holder.splines.size(), (std::size_t)maxNumKnots));

    holder.splines[maxNumKnots - 1] =
      std::unique_ptr<Interface>(new SplineClass<Vec, maxNumKnots>);

    SplineFactory<SplineClass, Vec, maxNumKnots - 1>::initialize(holder);
  }

  static SplineHolder<SplineClass, Vec> make()
  {
    auto holder = SplineHolder<SplineClass, Vec>{};
    initialize(holder);
    return holder;
  }
};

template<template<class, int> class SplineClass, class Vec>
struct SplineFactory<SplineClass, Vec, 0>
{
  static void initialize(SplineHolder<SplineClass, Vec>& holder)
  {
    // stops the recursion
  }

  static SplineHolder<SplineClass, Vec> make()
  {
    return SplineHolder<SplineClass, Vec>{};
  }
};

template<template<class, int> class SplineClass, class Vec>
template<int numKnots>
void
SplineHolder<SplineClass, Vec>::initialize()
{
  SplineFactory<SplineClass, Vec, numKnots>::initialize(*this);
}

template<template<class, int> class SplineClass, class Vec>
template<int numKnots>
SplineHolder<SplineClass, Vec>
SplineHolder<SplineClass, Vec>::make()
{
  return SplineFactory<SplineClass, Vec, numKnots>::make();
}

// implementation

template<class Vec, int numKnots_>
inline void
Spline<Vec, numKnots_>::processBlock(VecBuffer<Vec> const& input,
                                     VecBuffer<Vec>& output)
{
  int const numSamples = input.getNumSamples();
  output.setNumSamples(numSamples);

  Vec const alpha = this->data->smoothingAlpha[0];

  Vec xs[numKnots];
  Vec ys[numKnots];
  Vec ts[numKnots];
  Vec ss[numKnots];

  Vec xt[numKnots];
  Vec yt[numKnots];
  Vec tt[numKnots];
  Vec st[numKnots];

  for (int n = 0; n < numKnots; ++n) {
    xs[n] = Vec().load_a(this->data->knots[n].state.x);
    ys[n] = Vec().load_a(this->data->knots[n].state.y);
    ts[n] = Vec().load_a(this->data->knots[n].state.t);
    ss[n] = Vec().load_a(this->data->knots[n].state.s);
    xt[n] = Vec().load_a(this->data->knots[n].target.x);
    yt[n] = Vec().load_a(this->data->knots[n].target.y);
    tt[n] = Vec().load_a(this->data->knots[n].target.t);
    st[n] = Vec().load_a(this->data->knots[n].target.s);
  }

  for (int i = 0; i < numSamples; ++i) {
    Vec const in = input[i];

    // advance automation
    for (int n = 0; n < numKnots; ++n) {
      xs[n] = alpha * (xs[n] - xt[n]) + xt[n];
      ys[n] = alpha * (ys[n] - yt[n]) + yt[n];
      ts[n] = alpha * (ts[n] - tt[n]) + tt[n];
      ss[n] = alpha * (ss[n] - st[n]) + st[n];
    }

    // left knot paramters

    Vec x0 = std::numeric_limits<float>::lowest();
    Vec y0 = 0.f;
    Vec t0 = 0.f;
    Vec s0 = 0.f;

    // right knot paramters

    Vec x1 = std::numeric_limits<float>::max();
    Vec y1 = 0.f;
    Vec t1 = 0.f;
    Vec s1 = 0.f;

    // parameters for segment below the range of the spline

    Vec x_low = xs[0];
    Vec y_low = ys[0];
    Vec t_low = ts[0];

    // parameters for segment above the range of the spline

    Vec x_high = xs[0];
    Vec y_high = ys[0];
    Vec t_high = ts[0];

    // find interval and set left and right knot parameters

    for (int n = 0; n < numKnots; ++n) {
      auto const is_left = (in > xs[n]) && (xs[n] > x0);
      x0 = select(is_left, xs[n], x0);
      y0 = select(is_left, ys[n], y0);
      t0 = select(is_left, ts[n], t0);
      s0 = select(is_left, ss[n], s0);

      auto const is_right = (in <= xs[n]) && (xs[n] < x1);
      x1 = select(is_right, xs[n], x1);
      y1 = select(is_right, ys[n], y1);
      t1 = select(is_right, ts[n], t1);
      s1 = select(is_right, ss[n], s1);

      auto const is_lowest = xs[n] < x_low;
      x_low = select(is_lowest, xs[n], x_low);
      y_low = select(is_lowest, ys[n], y_low);
      t_low = select(is_lowest, ts[n], t_low);

      auto const is_highest = xs[n] > x_high;
      x_high = select(is_highest, xs[n], x_high);
      y_high = select(is_highest, ys[n], y_high);
      t_high = select(is_highest, ts[n], t_high);
    }

    auto const is_high = x1 == std::numeric_limits<float>::max();
    auto const is_low = x0 == std::numeric_limits<float>::lowest();

    // compute spline and segment coeffcients

    Vec const dx = max(x1 - x0, std::numeric_limits<float>::min());
    Vec const dy = y1 - y0;
    Vec const a = t0 * dx - dy;
    Vec const b = -t1 * dx + dy;
    Vec const ix = 1.0 / dx;
    Vec const m = dy * ix;
    Vec const o = y0 - m * x0;

    // compute spline

    Vec const j = (in - x0) * ix;
    Vec const k = 1.0 - j;
    Vec const hermite = k * y0 + j * y1 + j * k * (a * k + b * j);

    // compute segment and interpolate using smoothness

    Vec const segment = m * in + o;
    Vec const smoothness = s1 + k * (s0 - s1);
    Vec const curve = segment + smoothness * (hermite - segment);

    // override the result if the input is outside the spline range

    Vec const low = y_low + (in - x_low) * t_low;
    Vec const high = y_high + (in - x_high) * t_high;

    Vec const out = select(is_high, high, select(is_low, low, curve));

    // store the out

    output[i] = out;
  }

  // update spline state

  for (int n = 0; n < numKnots; ++n) {
    xs[n].store_a(this->data->knots[n].state.x);
    ys[n].store_a(this->data->knots[n].state.y);
    ts[n].store_a(this->data->knots[n].state.t);
    ss[n].store_a(this->data->knots[n].state.s);
    xt[n].store_a(this->data->knots[n].target.x);
    yt[n].store_a(this->data->knots[n].target.y);
    tt[n].store_a(this->data->knots[n].target.t);
    st[n].store_a(this->data->knots[n].target.s);
  }
}

template<class Vec, int numKnots_>
inline void
WaveShaper<Vec, numKnots_>::processBlock(VecBuffer<Vec> const& input,
                                         VecBuffer<Vec>& output)
{
  int const numSamples = input.getNumSamples();
  output.setNumSamples(numSamples);

  Vec const alpha = this->data->smoothingAlpha[0];

  Vec xs[numKnots];
  Vec ys[numKnots];
  Vec ts[numKnots];
  Vec ss[numKnots];

  Vec xt[numKnots];
  Vec yt[numKnots];
  Vec tt[numKnots];
  Vec st[numKnots];

  Vec ds = Vec().load_a(this->data->settingsState.dc);
  Vec dt = Vec().load_a(this->data->settingsTarget.dc);

  Vec ws = Vec().load_a(this->data->settingsState.wet);
  Vec wt = Vec().load_a(this->data->settingsTarget.wet);

  Vec hi = Vec().load_a(this->data->highPass.in);
  Vec ho = Vec().load_a(this->data->highPass.out);
  Vec ha = Vec().load_a(this->data->highPass.alpha);

  auto symm = Vec().load_a(this->data->isSymmetric) != 0.0;

  for (int n = 0; n < numKnots; ++n) {
    xs[n] = Vec().load_a(this->data->knots[n].state.x);
    ys[n] = Vec().load_a(this->data->knots[n].state.y);
    ts[n] = Vec().load_a(this->data->knots[n].state.t);
    ss[n] = Vec().load_a(this->data->knots[n].state.s);
    xt[n] = Vec().load_a(this->data->knots[n].target.x);
    yt[n] = Vec().load_a(this->data->knots[n].target.y);
    tt[n] = Vec().load_a(this->data->knots[n].target.t);
    st[n] = Vec().load_a(this->data->knots[n].target.s);
  }

  for (int i = 0; i < numSamples; ++i) {

    // advance automation

    for (int n = 0; n < numKnots; ++n) {
      xs[n] = alpha * (xs[n] - xt[n]) + xt[n];
      ys[n] = alpha * (ys[n] - yt[n]) + yt[n];
      ts[n] = alpha * (ts[n] - tt[n]) + tt[n];
      ss[n] = alpha * (ss[n] - st[n]) + st[n];
    }
    ws = alpha * (ws - wt) + wt;
    ds = alpha * (ds - dt) + dt;

    Vec const in_back = input[i];

    Vec const with_dc = in_back + ds;

    Vec const in = select(symm, abs(with_dc), with_dc);

    // left knot paramters

    Vec x0 = std::numeric_limits<float>::lowest();
    Vec y0 = 0.f;
    Vec t0 = 0.f;
    Vec s0 = 0.f;

    // right knot paramters

    Vec x1 = std::numeric_limits<float>::max();
    Vec y1 = 0.f;
    Vec t1 = 0.f;
    Vec s1 = 0.f;

    // parameters for segment below the range of the spline

    Vec x_low = xs[0];
    Vec y_low = ys[0];
    Vec t_low = ts[0];

    // parameters for segment above the range of the spline

    Vec x_high = xs[0];
    Vec y_high = ys[0];
    Vec t_high = ts[0];

    // find interval and set left and right knot parameters

    for (int n = 0; n < numKnots; ++n) {
      auto const is_left = (in > xs[n]) && (xs[n] > x0);
      x0 = select(is_left, xs[n], x0);
      y0 = select(is_left, ys[n], y0);
      t0 = select(is_left, ts[n], t0);
      s0 = select(is_left, ss[n], s0);

      auto const is_right = (in <= xs[n]) && (xs[n] < x1);
      x1 = select(is_right, xs[n], x1);
      y1 = select(is_right, ys[n], y1);
      t1 = select(is_right, ts[n], t1);
      s1 = select(is_right, ss[n], s1);

      auto const is_lowest = xs[n] < x_low;
      x_low = select(is_lowest, xs[n], x_low);
      y_low = select(is_lowest, ys[n], y_low);
      t_low = select(is_lowest, ts[n], t_low);

      auto const is_highest = xs[n] > x_high;
      x_high = select(is_highest, xs[n], x_high);
      y_high = select(is_highest, ys[n], y_high);
      t_high = select(is_highest, ts[n], t_high);
    }

    auto const is_high = x1 == std::numeric_limits<float>::max();
    auto const is_low = x0 == std::numeric_limits<float>::lowest();

    // compute spline and segment coefficients

    Vec const dx = max(x1 - x0, std::numeric_limits<float>::min());
    Vec const dy = y1 - y0;
    Vec const a = t0 * dx - dy;
    Vec const b = -t1 * dx + dy;
    Vec const ix = 1.0 / dx;
    Vec const m = dy * ix;
    Vec const o = y0 - m * x0;

    // compute spline

    Vec const j = (in - x0) * ix;
    Vec const k = 1.0 - j;
    Vec const hermite = k * y0 + j * y1 + j * k * (a * k + b * j);

    // compute segment and interpolate using smoothness

    Vec const segment = m * in + o;
    Vec const smoothness = s1 + k * (s0 - s1);
    Vec const curve = segment + smoothness * (hermite - segment);

    // override the result if the input is outside the spline range

    Vec const low = y_low + (in - x_low) * t_low;
    Vec const high = y_high + (in - x_high) * t_high;

    Vec out = select(is_high, high, select(is_low, low, curve));

    // symmetry

    out = select(symm, sign_combine(out, with_dc), out);

    // high pass, simple 1 pole

    ho = ha * (ho + out - hi);
    hi = out;
    out = ho;

    // dry-wet mix

    out = in_back + ws * (out - in_back);

    // store the out

    output[i] = out;
  }

  // update spline state

  for (int n = 0; n < numKnots; ++n) {
    xs[n].store_a(this->data->knots[n].state.x);
    ys[n].store_a(this->data->knots[n].state.y);
    ts[n].store_a(this->data->knots[n].state.t);
    ss[n].store_a(this->data->knots[n].state.s);
  }
  ds.store_a(this->data->settingsState.dc);
  ws.store_a(this->data->settingsState.wet);
  hi.store_a(this->data->highPass.in);
  ho.store_a(this->data->highPass.out);
}

} // namespace avec
