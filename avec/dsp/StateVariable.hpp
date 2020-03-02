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
#include "avec/dsp/SplineMacro.hpp"

namespace avec {

template<class Vec>
struct StateVariable
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  static constexpr Scalar pi = 3.141592653589793238;

  Scalar smoothingAlpha[Vec::size()];
  Scalar state[2 * Vec::size()];
  Scalar frequency[Vec::size()];
  Scalar resonance[Vec::size()];
  Scalar frequencyTarget[Vec::size()];
  Scalar resonanceTarget[Vec::size()];

  StateVariable()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    reset();
  }

  void reset()
  {
    std::copy(frequencyTarget, frequencyTarget + Vec::size(), frequency);
    std::copy(resonanceTarget, resonanceTarget + Vec::size(), resonance);
    std::fill_n(state, 2 * Vec::size(), 0.0);
  }

  void setFrequency(Scalar normalized, int channel)
  {
    frequencyTarget[channel] = tan(pi * normalized);
  }

  void setFrequency(Scalar normalized)
  {
    std::fill_n(frequencyTarget, Vec::size(), tan(pi * normalized));
  }

  void setResonance(Scalar value)
  {
    std::fill_n(resonance, Vec::size(), 2.0 * value);
  }

  void setResonance(Scalar value, int channel)
  {
    resonance[channel] = 2.0 * value;
  }

  void setBandPass(Scalar bandwidth,
                   Scalar normalizedCenterFrequency,
                   int channel)
  {
    Scalar b = pow(2.0, bandwidth * 0.5);
    Scalar w0 = tan(pi * normalizedCenterFrequency * b);
    Scalar w1 = tan(pi * normalizedCenterFrequency * b);
    frequencyTarget[channel] = sqrt(w0 * w1);
    Scalar d = w0 / w1;
    resonance[channel] = 0.5 * (d - 1.0 / d);
  }

  void setBandPass(Scalar bandwidth, Scalar normalizedCenterFrequency)
  {
    Scalar b = pow(2.0, bandwidth * 0.5);
    Scalar w0 = tan(pi * normalizedCenterFrequency * b);
    Scalar w1 = tan(pi * normalizedCenterFrequency * b);
    std::fill_n(frequencyTarget, Vec::size(), sqrt(w0 * w1));
    Scalar d = w0 / w1;
    std::fill_n(resonance, Vec::size(), 0.5 * (d - 1.0 / d));
  }

  void setSmoothingAlpha(Scalar alpha)
  {
    std::fill_n(smoothingAlpha, Vec::size(), alpha);
  }

  void highPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + Vec::size());

    Vec g = Vec().load_a(frequency);
    Vec g_a = Vec().load_a(frequencyTarget);

    Vec r = Vec().load_a(resonance);
    Vec r_a = Vec().load_a(resonanceTarget);

    Vec alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec in = input[i];

      Vec r_g = r + g;

      Vec high = (in - r_g * s1 - s2) / (1.0 + r_g * g);

      Vec v1 = g * high;
      Vec band = v1 + s1;
      s1 = band + v1;

      Vec v2 = g * band;
      Vec low = v2 + s2;
      s2 = low + v2;

      output[i] = high;
    }

    s1.store_a(state);
    s2.store_a(state + Vec::size());
    g.store_a(frequency);
    r.store_a(resonance);
  }

  void bandPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    bandPassAlgorithm<1>(input, output);
  }

  void lowPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    bandPassAlgorithm<0>(input, output);
  }

private:
  template<int isBandPass>
  void bandPassAlgorithm(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + Vec::size());

    Vec g = Vec().load_a(frequency);
    Vec g_a = Vec().load_a(frequencyTarget);

    Vec r = Vec().load_a(resonance);
    Vec r_a = Vec().load_a(resonanceTarget);

    Vec alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec in = input[i];

      Vec band = (g * (in - s2) + s1) / (1.0 + g * (r + g));

      s1 = 2.0 * band - s1;

      Vec v2 = g * band;
      Vec low = v2 + s2;
      s2 = low + v2;

      if constexpr (isBandPass) {
        output[i] = band * r;
      }
      else {
        output[i] = low;
      }
    }

    s1.store_a(state);
    s2.store_a(state + Vec::size());
    g.store_a(frequency);
    r.store_a(resonance);
  }
};

} // namespace avec