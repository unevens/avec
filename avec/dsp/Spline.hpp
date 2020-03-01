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
struct SplineInterface;

template<class Vec>
struct SplineAutomatorInterface;

template<class Vec, int numKnots_>
struct Spline;

template<class Vec, int numKnots_>
struct SplineAutomator;

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

  virtual Knot* getKnots() = 0;

  virtual int getNumKnots() = 0;

  virtual void setIsSymmetric(bool isSymmetric) = 0;

  virtual void setIsSymmetric(bool isSymmetric, int channel) = 0;

  virtual Scalar* getIsSymmetric() = 0;

  virtual void processBlock(VecBuffer<Vec> const& input,
                            VecBuffer<Vec>& output) = 0;

  virtual void processBlock(VecBuffer<Vec> const& input,
                            VecBuffer<Vec>& output,
                            SplineAutomatorInterface<Vec>* automator) = 0;

  virtual ~SplineInterface() {}
};

template<class Vec>
struct SplineAutomatorInterface
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  using Knot = typename SplineInterface<Vec>::Knot;

  virtual Knot* getKnots() = 0;

  virtual Scalar* getSmoothingAlpha() = 0;

  virtual void setSmoothingAlpha(Scalar alpha) = 0;
  virtual void reset(SplineInterface<Vec>* spline) = 0;

  virtual ~SplineAutomatorInterface() {}
};

template<class Vec, int numKnots_>
struct Spline final : public SplineInterface<Vec>
{
  static constexpr int numKnots = numKnots_;

  using Interface = SplineInterface<Vec>;
  using Scalar = typename Interface::Scalar;
  using Knot = typename Interface::Knot;

  struct Data final
  {
    Scalar isSymmetric[Vec::size()];
    Knot knots[numKnots];
  };

  aligned_ptr<Data> data;

  Knot* getKnots() override { return data->knots; }

  int getNumKnots() override { return numKnots; }

  void setIsSymmetric(bool isSymmetric) override
  {
    std::fill_n(data->isSymmetric, Vec::size(), isSymmetric ? 1.0 : 0.0);
  }

  void setIsSymmetric(bool isSymmetric, int channel) override
  {
    data->isSymmetric[channel] = isSymmetric ? 1.0 : 0.0;
  }

  Scalar* getIsSymmetric() override { return data->isSymmetric; }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output) override;

  void processBlock(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    SplineAutomatorInterface<Vec>* automator = nullptr) override;

  Spline()
    : data(avec::Aligned<Data>::make())
  {
    std::fill_n(data->isSymmetric, sizeof(Data) / sizeof(Scalar), 0.0);
  }
};

template<class Vec, int numKnots_>
struct SplineAutomator final : public SplineAutomatorInterface<Vec>
{
  static constexpr int numKnots = numKnots_;

  using Interface = SplineAutomatorInterface<Vec>;
  using Knot = typename SplineInterface<Vec>::Knot;

  struct Data final
  {
    Scalar smoothingAlpha[Vec::size()];
    Knot knots[numKnots];
  };

  aligned_ptr<Data> data;

  Knot* getKnots() override { return data->knots; }

  void setSmoothingAlpha(Scalar alpha) override
  {
    std::fill_n(data->smoothingAlpha, Vec::size(), alpha);
  }

  Scalar* getSmoothingAlpha() override { return data->smoothingAlpha; };

  void reset(SplineInterface<Vec>* spline) override;

  SplineAutomator()
    : data(avec::Aligned<Data>::make())
  {
    std::fill_n(data->smoothingAlpha, sizeof(Data) / sizeof(Scalar), 0.0);
  }
};

template<class Vec>
struct SplineHolder final
{
  std::vector<std::unique_ptr<SplineInterface<Vec>>> splines;
  std::vector<std::unique_ptr<SplineAutomatorInterface<Vec>>> automators;

  std::pair<SplineInterface<Vec>*, SplineAutomatorInterface<Vec>*> getSpline(
    int numKnots)
  {
    int index = numKnots - 1;
    if (index < splines.size()) {
      return { splines[index].get(),
               index < automators.size() ? automators[index].get() : nullptr };
    }
    return { nullptr, nullptr };
  }

  template<int numKnots>
  void initialize(bool makeAutomators);

  void reset()
  {
    int i = 0;
    for (auto& a : automators) {
      a->reset(splines[i++]);
    }
  }

  template<int numKnots>
  static SplineHolder make(bool makeAutomators);
};

template<class Vec, int maxNumKnots>
struct SplineFactory
{
  static void initialize(SplineHolder<Vec>& holder, bool makeAutomators)
  {
    holder.splines.resize(
      std::max(holder.splines.size(), (std::size_t)maxNumKnots));

    holder.splines[maxNumKnots - 1] =
      std::unique_ptr<SplineInterface<Vec>>(new Spline<Vec, maxNumKnots>);

    if (makeAutomators) {

      holder.automators.resize(
        std::max(holder.automators.size(), (std::size_t)maxNumKnots));

      holder.automators[maxNumKnots - 1] =
        std::unique_ptr<SplineAutomatorInterface<Vec>>(
          new SplineAutomator<Vec, maxNumKnots>);
    }

    SplineFactory<Vec, maxNumKnots - 1>::initialize(holder, makeAutomators);
  }

  static SplineHolder<Vec> make(bool makeAutomators)
  {
    auto holder = SplineHolder<Vec>{};
    initialize(holder, makeAutomators);
    return holder;
  }
};

template<class Vec>
struct SplineFactory<Vec, 0>
{
  static void initialize(SplineHolder<Vec>& holder, bool makeAutomators)
  {
    // stops the recursion
  }

  static SplineHolder<Vec> make(bool makeAutomators)
  {
    return SplineHolder<Vec>{};
  }
};

template<class Vec>
template<int numKnots>
void
SplineHolder<Vec>::initialize(bool makeAutomators)
{
  SplineFactory<Vec, numKnots>::initialize(*this, makeAutomators);
}

template<class Vec>
template<int numKnots>
SplineHolder<Vec>
SplineHolder<Vec>::make(bool makeAutomators)
{
  return SplineFactory<Vec, numKnots>::make(makeAutomators);
}

// implementation

template<class Vec, int numKnots_>
inline void
Spline<Vec, numKnots_>::processBlock(VecBuffer<Vec> const& input,
                                     VecBuffer<Vec>& output,
                                     SplineAutomatorInterface<Vec>* automator)
{
  int const numSamples = input.getNumSamples();
  output.setNumSamples(numSamples);

  Vec const alpha = Vec().load_a(automator->getSmoothingAlpha());

  Vec x[numKnots];
  Vec y[numKnots];
  Vec t[numKnots];
  Vec s[numKnots];

  Vec x_a[numKnots];
  Vec y_a[numKnots];
  Vec t_a[numKnots];
  Vec s_a[numKnots];

  auto symm = Vec().load_a(this->data->isSymmetric) != 0.0;

  for (int n = 0; n < numKnots; ++n) {
    x[n] = Vec().load_a(this->data->knots[n].x);
    y[n] = Vec().load_a(this->data->knots[n].y);
    t[n] = Vec().load_a(this->data->knots[n].t);
    s[n] = Vec().load_a(this->data->knots[n].s);
  }

  auto* automationKnots = automator->getKnots();

  for (int n = 0; n < numKnots; ++n) {
    x_a[n] = Vec().load_a(automationKnots[n].x);
    y_a[n] = Vec().load_a(automationKnots[n].y);
    t_a[n] = Vec().load_a(automationKnots[n].t);
    s_a[n] = Vec().load_a(automationKnots[n].s);
  }

  for (int i = 0; i < numSamples; ++i) {

    // advance automation

    for (int n = 0; n < numKnots; ++n) {
      x[n] = alpha * (x[n] - x_a[n]) + x_a[n];
      y[n] = alpha * (y[n] - y_a[n]) + y_a[n];
      t[n] = alpha * (t[n] - t_a[n]) + t_a[n];
      s[n] = alpha * (s[n] - s_a[n]) + s_a[n];
    }

    Vec const in_signed = input[i];

    Vec const in = select(symm, abs(in_signed), in_signed);

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

    Vec x_low = x[0];
    Vec y_low = y[0];
    Vec t_low = t[0];

    // parameters for segment above the range of the spline

    Vec x_high = x[0];
    Vec y_high = y[0];
    Vec t_high = t[0];

    // find interval and set left and right knot parameters

    for (int n = 0; n < numKnots; ++n) {
      auto const is_left = (in > x[n]) && (x[n] > x0);
      x0 = select(is_left, x[n], x0);
      y0 = select(is_left, y[n], y0);
      t0 = select(is_left, t[n], t0);
      s0 = select(is_left, s[n], s0);

      auto const is_right = (in <= x[n]) && (x[n] < x1);
      x1 = select(is_right, x[n], x1);
      y1 = select(is_right, y[n], y1);
      t1 = select(is_right, t[n], t1);
      s1 = select(is_right, s[n], s1);

      auto const is_lowest = x[n] < x_low;
      x_low = select(is_lowest, x[n], x_low);
      y_low = select(is_lowest, y[n], y_low);
      t_low = select(is_lowest, t[n], t_low);

      auto const is_highest = x[n] > x_high;
      x_high = select(is_highest, x[n], x_high);
      y_high = select(is_highest, y[n], y_high);
      t_high = select(is_highest, t[n], t_high);
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

    // symmetry

    output[i] = select(symm, sign_combine(out, in_signed), out);
  }

  // update spline state

  for (int n = 0; n < numKnots; ++n) {
    x[n].store_a(this->data->knots[n].x);
    y[n].store_a(this->data->knots[n].y);
    t[n].store_a(this->data->knots[n].t);
    s[n].store_a(this->data->knots[n].s);
  }
}

template<class Vec, int numKnots_>
inline void
Spline<Vec, numKnots_>::processBlock(VecBuffer<Vec> const& input,
                                     VecBuffer<Vec>& output)
{
  int const numSamples = input.getNumSamples();
  output.setNumSamples(numSamples);

  Vec x[numKnots];
  Vec y[numKnots];
  Vec t[numKnots];
  Vec s[numKnots];

  auto symm = Vec().load_a(this->data->isSymmetric) != 0.0;

  for (int n = 0; n < numKnots; ++n) {
    x[n] = Vec().load_a(this->data->knots[n].x);
    y[n] = Vec().load_a(this->data->knots[n].y);
    t[n] = Vec().load_a(this->data->knots[n].t);
    s[n] = Vec().load_a(this->data->knots[n].s);
  }

  for (int i = 0; i < numSamples; ++i) {

    Vec const in_signed = input[i];

    Vec const in = select(symm, abs(in_signed), in_signed);

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

    Vec x_low = x[0];
    Vec y_low = y[0];
    Vec t_low = t[0];

    // parameters for segment above the range of the spline

    Vec x_high = x[0];
    Vec y_high = y[0];
    Vec t_high = t[0];

    // find interval and set left and right knot parameters

    for (int n = 0; n < numKnots; ++n) {
      auto const is_left = (in > x[n]) && (x[n] > x0);
      x0 = select(is_left, x[n], x0);
      y0 = select(is_left, y[n], y0);
      t0 = select(is_left, t[n], t0);
      s0 = select(is_left, s[n], s0);

      auto const is_right = (in <= x[n]) && (x[n] < x1);
      x1 = select(is_right, x[n], x1);
      y1 = select(is_right, y[n], y1);
      t1 = select(is_right, t[n], t1);
      s1 = select(is_right, s[n], s1);

      auto const is_lowest = x[n] < x_low;
      x_low = select(is_lowest, x[n], x_low);
      y_low = select(is_lowest, y[n], y_low);
      t_low = select(is_lowest, t[n], t_low);

      auto const is_highest = x[n] > x_high;
      x_high = select(is_highest, x[n], x_high);
      y_high = select(is_highest, y[n], y_high);
      t_high = select(is_highest, t[n], t_high);
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

    // symmetry

    output[i] = select(symm, sign_combine(out, in_signed), out);
  }
}

template<class Vec, int numKnots_>
inline void
SplineAutomator<Vec, numKnots_>::reset(SplineInterface<Vec>* spline)
{
  int i = 0;
  auto splineKnots = spline->getKnots();
  for (auto& knot : data->knots) {
    std::copy(&knot, &knot + 1, &splineKnots[i++]);
  }
}

} // namespace avec
