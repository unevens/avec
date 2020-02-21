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

  struct Node final
  {
    Scalar x[Vec::size()];
    Scalar y[Vec::size()];
    Scalar t[Vec::size()];
    Scalar s[Vec::size()];
  };

  struct AutomatableNode final
  {
    Node state;
    Node target;
  };

  virtual AutomatableNode* GetNodes() = 0;

  virtual int GetNumNodes() = 0;

  virtual Scalar* GetSmoothingAlpha() = 0;

  virtual void SetSmoothingFrequency(Scalar frequency) = 0;

  virtual void ProcessBlock(VecBuffer<Vec> const& input,
                            VecBuffer<Vec>& output) = 0;

  virtual void Reset() = 0;
};

template<class Vec>
struct WaveShaperInterface : public SplineInterface<Vec>
{
  using Scalar = typename SplineInterface<Vec>::Scalar;

  virtual void SetHighPassFrequency(Scalar frequency) = 0;

  virtual void SetHighPassFrequency(Scalar frequency, int channel) = 0;

  virtual void SetDc(Scalar frequency) = 0;

  virtual void SetDc(Scalar frequency, int channel) = 0;

  virtual void SetWet(Scalar frequency) = 0;

  virtual void SetWet(Scalar frequency, int channel) = 0;

  virtual void SetIsSymmetric(bool isSymmetric) = 0;

  virtual void SetIsSymmetric(Scalar isSymmetric, int channel) = 0;

  virtual Scalar* GetHighPassAlpha() = 0;

  virtual Scalar* GetDcState() = 0;
  virtual Scalar* GetDcTarget() = 0;

  virtual Scalar* GetWetState() = 0;
  virtual Scalar* GetWetTarget() = 0;

  virtual Scalar* GetHighPassIn() = 0;
  virtual Scalar* GetHighPassOut() = 0;

  virtual Scalar* GetIsSymmetric() = 0;
};

template<class Vec, int numNodes_>
struct Spline final : public SplineInterface<Vec>
{
  static constexpr int numNodes = numNodes_;

  using Interface = SplineInterface<Vec>;
  using Scalar = typename Interface::Scalar;
  using AutomatableNode = typename Interface::AutomatableNode;

  struct Data final
  {
    Scalar smoothingAlpha[Vec::size()];
    AutomatableNode nodes[numNodes];
  };

  aligned_ptr<Data> data;

  AutomatableNode* GetNodes() override { return data->nodes; }

  int GetNumNodes() override { return numNodes; }

  void SetSmoothingFrequency(Scalar frequency) override
  {
    std::fill_n(data->smoothingAlpha, Vec::size(), exp(-frequency));
  }

  Scalar* GetSmoothingAlpha() override { return data->smoothingAlpha; };

  virtual void Reset() override
  {
    for (auto& node : data->nodes) {
      std::copy(&node.target.x[0],
                &node.target.x[0] + 4 * Vec::size(),
                &node.state.x[0]);
    }
  }

  void ProcessBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output) override;

  Spline()
    : data(avec::Aligned<Data>::New())
  {
    std::fill_n(data->smoothingAlpha, sizeof(Data) / sizeof(Scalar), 0.0);
  }
};

template<class Vec, int numNodes_>
struct WaveShaper final : public WaveShaperInterface<Vec>
{
  static constexpr int numNodes = numNodes_;
  using Interface = WaveShaperInterface<Vec>;
  using Scalar = typename Interface::Scalar;
  using AutomatableNode = typename Interface::AutomatableNode;

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
    AutomatableNode nodes[numNodes];
    Scalar isSymmetric[Vec::size()];
    Settings settingsTarget;
    Settings settingsState;
    HighPass highPass;
  };

  aligned_ptr<Data> data;

  AutomatableNode* GetNodes() override { return data->nodes; }

  int GetNumNodes() override { return numNodes; }

  void SetSmoothingFrequency(Scalar frequency) override
  {
    std::fill_n(data->smoothingAlpha, Vec::size(), exp(-frequency));
  }

  void SetHighPassFrequency(Scalar frequency) override
  {
    std::fill_n(data->highPass.alpha, Vec::size(), exp(-frequency));
  }

  void SetHighPassFrequency(Scalar frequency, int channel) override
  {
    data->highPass.alpha[channel] = exp(-frequency);
  }

  void SetDc(Scalar dc) override
  {
    std::fill_n(data->settingsTarget.dc, Vec::size(), dc);
  }

  void SetDc(Scalar dc, int channel) override
  {
    data->settingsTarget.dc[channel] = dc;
  }

  void SetWet(Scalar wet) override
  {
    std::fill_n(data->settingsTarget.wet, Vec::size(), wet);
  }

  void SetWet(Scalar wet, int channel) override
  {
    data->settingsTarget.wet[channel] = wet;
  }

  void SetIsSymmetric(bool isSymmetric) override
  {
    std::fill_n(data->isSymmetric, Vec::size(), isSymmetric ? 1.0 : 0.0);
  }

  void SetIsSymmetric(Scalar isSymmetric, int channel) override
  {
    data->isSymmetric[channel] = isSymmetric;
  }

  Scalar* GetSmoothingAlpha() override { return data->smoothingAlpha; };

  Scalar* GetHighPassAlpha() override { return data->highPass.alpha; }

  Scalar* GetDcState() override { return data->settingsState.dc; }
  Scalar* GetDcTarget() override { return data->settingsTarget.dc; }

  Scalar* GetWetState() override { return data->settingsState.wet; }
  Scalar* GetWetTarget() override { return data->settingsTarget.wet; }

  Scalar* GetHighPassIn() override { return data->highPass.in; }
  Scalar* GetHighPassOut() override { return data->highPass.out; }

  Scalar* GetIsSymmetric() override { return data->isSymmetric; }

  void ProcessBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output) override;

  void Reset() override
  {
    for (auto& node : data->nodes) {
      std::copy(&node.target.x[0],
                &node.target.x[0] + 4 * Vec::size(),
                &node.state.x[0]);
    }
    std::copy(&data->settingsTarget.dc[0],
              &data->settingsTarget.dc[0] + 2 * Vec::size(),
              &data->settingsState.dc[0]);
    std::fill_n(&data->highPass.in[0], 2 * Vec::size(), 0.0);
  }

  WaveShaper()
    : data(avec::Aligned<Data>::New())
  {
    std::fill_n(data->smoothingAlpha, sizeof(Data) / sizeof(Scalar), 0.0);
  }
};

template<template<class, int> class SplineClass, class Vec>
struct SplineHolder final
{
  using SplineInterface = typename SplineClass<Vec, 1>::Interface;

  std::vector<std::unique_ptr<SplineInterface>> splines;

  SplineInterface* GetSpline(int numNodes)
  {
    int index = numNodes - 1;
    if (index < splines.size()) {
      return splines[index].get();
    }
    return nullptr;
  }

  template<int numNodes>
  void Initialize();

  void Reset()
  {
    for (auto& s : splines) {
      s->Reset();
    }
  }

  template<int numNodes>
  static SplineHolder New();
};

template<template<class, int> class SplineClass, class Vec, int maxNumNodes>
struct SplineFactory
{
  static void Initialize(SplineHolder<SplineClass, Vec>& holder)
  {
    using Interface = typename SplineClass<Vec, maxNumNodes>::Interface;

    holder.splines.resize(
      std::max(holder.splines.size(), (std::size_t)maxNumNodes));

    holder.splines[maxNumNodes - 1] =
      std::unique_ptr<Interface>(new SplineClass<Vec, maxNumNodes>);

    SplineFactory<SplineClass, Vec, maxNumNodes - 1>::Initialize(holder);
  }

  static SplineHolder<SplineClass, Vec> New()
  {
    auto holder = SplineHolder<SplineClass, Vec>{};
    Initialize(holder);
    return holder;
  }
};

template<template<class, int> class SplineClass, class Vec>
struct SplineFactory<SplineClass, Vec, 0>
{
  static void Initialize(SplineHolder<SplineClass, Vec>& holder)
  {
    // stops the recursion
  }

  static SplineHolder<SplineClass, Vec> New()
  {
    return SplineHolder<SplineClass, Vec>{};
  }
};

template<template<class, int> class SplineClass, class Vec>
template<int numNodes>
void
SplineHolder<SplineClass, Vec>::Initialize()
{
  SplineFactory<SplineClass, Vec, numNodes>::Initialize(*this);
}

template<template<class, int> class SplineClass, class Vec>
template<int numNodes>
SplineHolder<SplineClass, Vec>
SplineHolder<SplineClass, Vec>::New()
{
  return SplineFactory<SplineClass, Vec, numNodes>::New();
}

// implementation

template<class Vec, int numNodes_>
inline void
Spline<Vec, numNodes_>::ProcessBlock(VecBuffer<Vec> const& input,
                                     VecBuffer<Vec>& output)
{
  int const numSamples = input.GetNumSamples();
  output.SetNumSamples(numSamples);

  Vec const alpha = this->data->smoothingAlpha[0];

  Vec xs[numNodes];
  Vec ys[numNodes];
  Vec ts[numNodes];
  Vec ss[numNodes];

  Vec xt[numNodes];
  Vec yt[numNodes];
  Vec tt[numNodes];
  Vec st[numNodes];

  for (int n = 0; n < numNodes; ++n) {
    xs[n] = Vec().load_a(this->data->nodes[n].state.x);
    ys[n] = Vec().load_a(this->data->nodes[n].state.y);
    ts[n] = Vec().load_a(this->data->nodes[n].state.t);
    ss[n] = Vec().load_a(this->data->nodes[n].state.s);
    xt[n] = Vec().load_a(this->data->nodes[n].target.x);
    yt[n] = Vec().load_a(this->data->nodes[n].target.y);
    tt[n] = Vec().load_a(this->data->nodes[n].target.t);
    st[n] = Vec().load_a(this->data->nodes[n].target.s);
  }

  for (int i = 0; i < numSamples; ++i) {
    Vec const in = input[i];

    // advance automation
    for (int n = 0; n < numNodes; ++n) {
      xs[n] = alpha * (xs[n] - xt[n]) + xt[n];
      ys[n] = alpha * (ys[n] - yt[n]) + yt[n];
      ts[n] = alpha * (ts[n] - tt[n]) + tt[n];
      ss[n] = alpha * (ss[n] - st[n]) + st[n];
    }

    // left node paramters

    Vec x0 = std::numeric_limits<float>::lowest();
    Vec y0 = 0.f;
    Vec t0 = 0.f;
    Vec s0 = 0.f;

    // right node paramters

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

    // find interval and set left and right node parameters

    for (int n = 0; n < numNodes; ++n) {
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

  for (int n = 0; n < numNodes; ++n) {
    xs[n].store_a(this->data->nodes[n].state.x);
    ys[n].store_a(this->data->nodes[n].state.y);
    ts[n].store_a(this->data->nodes[n].state.t);
    ss[n].store_a(this->data->nodes[n].state.s);
    xt[n].store_a(this->data->nodes[n].target.x);
    yt[n].store_a(this->data->nodes[n].target.y);
    tt[n].store_a(this->data->nodes[n].target.t);
    st[n].store_a(this->data->nodes[n].target.s);
  }
}

template<class Vec, int numNodes_>
inline void
WaveShaper<Vec, numNodes_>::ProcessBlock(VecBuffer<Vec> const& input,
                                         VecBuffer<Vec>& output)
{
  int const numSamples = input.GetNumSamples();
  output.SetNumSamples(numSamples);

  Vec const alpha = this->data->smoothingAlpha[0];

  Vec xs[numNodes];
  Vec ys[numNodes];
  Vec ts[numNodes];
  Vec ss[numNodes];

  Vec xt[numNodes];
  Vec yt[numNodes];
  Vec tt[numNodes];
  Vec st[numNodes];

  Vec ds = Vec().load_a(this->data->settingsState.dc);
  Vec dt = Vec().load_a(this->data->settingsTarget.dc);

  Vec ws = Vec().load_a(this->data->settingsState.wet);
  Vec wt = Vec().load_a(this->data->settingsTarget.wet);

  Vec hi = Vec().load_a(this->data->highPass.in);
  Vec ho = Vec().load_a(this->data->highPass.out);
  Vec ha = Vec().load_a(this->data->highPass.alpha);

  auto symm = Vec().load_a(this->data->isSymmetric) != 0.0;

  for (int n = 0; n < numNodes; ++n) {
    xs[n] = Vec().load_a(this->data->nodes[n].state.x);
    ys[n] = Vec().load_a(this->data->nodes[n].state.y);
    ts[n] = Vec().load_a(this->data->nodes[n].state.t);
    ss[n] = Vec().load_a(this->data->nodes[n].state.s);
    xt[n] = Vec().load_a(this->data->nodes[n].target.x);
    yt[n] = Vec().load_a(this->data->nodes[n].target.y);
    tt[n] = Vec().load_a(this->data->nodes[n].target.t);
    st[n] = Vec().load_a(this->data->nodes[n].target.s);
  }

  for (int i = 0; i < numSamples; ++i) {

    // advance automation

    for (int n = 0; n < numNodes; ++n) {
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

    // left node paramters

    Vec x0 = std::numeric_limits<float>::lowest();
    Vec y0 = 0.f;
    Vec t0 = 0.f;
    Vec s0 = 0.f;

    // right node paramters

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

    // find interval and set left and right node parameters

    for (int n = 0; n < numNodes; ++n) {
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

  for (int n = 0; n < numNodes; ++n) {
    xs[n].store_a(this->data->nodes[n].state.x);
    ys[n].store_a(this->data->nodes[n].state.y);
    ts[n].store_a(this->data->nodes[n].state.t);
    ss[n].store_a(this->data->nodes[n].state.s);
  }
  ds.store_a(this->data->settingsState.dc);
  ws.store_a(this->data->settingsState.wet);
  hi.store_a(this->data->highPass.in);
  ho.store_a(this->data->highPass.out);
}

} // namespace avec
