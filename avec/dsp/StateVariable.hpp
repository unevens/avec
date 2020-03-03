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
struct StateVariable
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  static constexpr Scalar pi = 3.141592653589793238;

  Scalar smoothingAlpha[Vec::size()];
  Scalar state[2 * Vec::size()];
  Scalar memory[2 * Vec::size()]; // for antisaturator
  Scalar frequency[Vec::size()];
  Scalar resonance[Vec::size()];
  Scalar frequencyTarget[Vec::size()];
  Scalar resonanceTarget[Vec::size()];

  StateVariable()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    setFrequency(0.25);
    setResonance(0.0);
    reset();
  }

  void reset()
  {
    std::copy(frequencyTarget, frequencyTarget + Vec::size(), frequency);
    std::copy(resonanceTarget, resonanceTarget + Vec::size(), resonance);
    std::fill_n(state, 3 * Vec::size(), 0.0);
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
    std::fill_n(resonanceTarget, Vec::size(), 2.0 * (1.0 - value));
  }

  void setResonance(Scalar value, int channel)
  {
    resonanceTarget[channel] = 2.0 * (1.0 - value);
  }

  void setupNormalizedBandPass(Scalar bandwidth,
                               Scalar normalizedFrequency,
                               int channel)
  {
    auto [w, r] = normalizedBandPassPrewarp(bandwidth, normalizedFrequency);
    frequencyTarget[channel] = w;
    resonanceTarget[channel] = r;
  }

  void setupNormalizedBandPass(Scalar bandwidth, Scalar normalizedFrequency)
  {
    auto [w, r] = normalizedBandPassPrewarp(bandwidth, normalizedFrequency);
    std::fill_n(frequencyTarget, Vec::size(), w);
    std::fill_n(resonanceTarget, Vec::size(), r);
  }

  void setSmoothingAlpha(Scalar alpha)
  {
    std::fill_n(smoothingAlpha, Vec::size(), alpha);
  }

  // linear

  void highPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + Vec::size());

    Vec g = Vec().load_a(frequency);
    Vec const g_a = Vec().load_a(frequencyTarget);

    Vec r = Vec().load_a(resonance);
    Vec const r_a = Vec().load_a(resonanceTarget);

    Vec const alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec const in = input[i];

      Vec const g_r = r + g;

      Vec const high = (in - g_r * s1 - s2) / (1.0 + g_r * g);

      Vec const v1 = g * high;
      Vec const band = v1 + s1;
      s1 = band + v1;

      Vec const v2 = g * band;
      Vec const low = v2 + s2;
      s2 = low + v2;

      output[i] = high;
    }

    s1.store_a(state);
    s2.store_a(state + Vec::size());
    g.store_a(frequency);
    r.store_a(resonance);
  }

  template<class Saturator, class SaturatorWithDerivative>
  void highPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + Vec::size());

    Vec g = Vec().load_a(frequency);
    Vec const g_a = Vec().load_a(frequencyTarget);

    Vec r = Vec().load_a(resonance);
    Vec const r_a = Vec().load_a(resonanceTarget);

    Vec const alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec const in = input[i];

      Vec const ng = saturationGain(s1);

      Vec const coef = r + g + 2.0 * ng;

      Vec high = (in - coef * s1 - s2) / (1.0 + coef * g);

      Vec const g_r = r + g;
      for (int it = 0; it < numIterations; ++it) {
        Vec nonlinear, derivative;
        Vec const band = g * high + s1;
        computeSaturationAndDerivative(band, nonlinear, derivative);
        Vec const imp = -high + in - g_r * band - nonlinear - s2;
        Vec const delta = -1.0 - g * (g_r + derivative);
        high -= imp / delta;
      }

      Vec const v1 = g * high;
      Vec const band = v1 + s1;
      s1 = band + v1;

      Vec const v2 = g * band;
      Vec const low = v2 + s2;
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
    bandPassAlgorithm<bandPassOutput>(input, output);
  }

  void lowPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    bandPassAlgorithm<lowPassOutput>(input, output);
  }

  void normalizedBandPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    bandPassAlgorithm<normalizedBandPassOutput>(input, output);
  }

  // nonlinear

  template<class Saturator, class SaturationGain, class SaturatorWithDerivative>
  void lowPass(VecBuffer<Vec> const& input,
               VecBuffer<Vec>& output,
               int numIterations,
               Saturator saturate,
               SaturationGain saturationGain,
               SaturatorWithDerivative computeSaturationAndDerivative)
  {
    withAntisaturation<lowPassOutput>(input,
                                      output,
                                      numIterations,
                                      saturate,
                                      saturationGain,
                                      computeSaturationAndDerivative);
  }

  template<class Saturator, class SaturationGain, class SaturatorWithDerivative>
  void bandPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturate,
                SaturationGain saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative)
  {
    withAntisaturation<bandPassOutput>(input,
                                       output,
                                       numIterations,
                                       saturate,
                                       saturationGain,
                                       computeSaturationAndDerivative);
  }

  template<class Saturator, class SaturationGain, class SaturatorWithDerivative>
  void normalizedBandPass(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    int numIterations,
    Saturator saturate,
    SaturationGain saturationGain,
    SaturatorWithDerivative computeSaturationAndDerivative)
  {
    withAntisaturation<normalizedBandPassOutput>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative);
  }

  template<class Saturator, class SaturationGain, class SaturatorWithDerivative>
  void highPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturate,
                SaturationGain saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative)
  {
    withAntisaturation<highPassOutput>(input,
                                       output,
                                       numIterations,
                                       saturate,
                                       saturationGain,
                                       computeSaturationAndDerivative);
  }

private:
  enum MultimodeOutputs
  {
    lowPassOutput = 0,
    bandPassOutput,
    normalizedBandPassOutput,
    highPassOutput
  };

  static std::pair<Scalar, Scalar> normalizedBandPassPrewarp(
    Scalar bandwidth,
    Scalar normalizedFrequency)
  {
    Scalar const b = pow(2.0, bandwidth * 0.5);
    Scalar const n0 = normalizedFrequency / b;
    Scalar const n1 = std::min(1.0, normalizedFrequency * b);
    Scalar const w0 = tan(pi * n0);
    Scalar const w1 = tan(pi * n1);
    Scalar const w = sqrt(w0 * w1);
    Scalar const r = 0.5 * w1 / w0;
    return { w, r };
  }

  template<int multimodeOutput>
  void bandPassAlgorithm(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + Vec::size());

    Vec g = Vec().load_a(frequency);
    Vec const g_a = Vec().load_a(frequencyTarget);

    Vec r = Vec().load_a(resonance);
    Vec const r_a = Vec().load_a(resonanceTarget);

    Vec const alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec const in = input[i];

      Vec const band = (g * (in - s2) + s1) / (1.0 + g * (r + g));

      s1 = 2.0 * band - s1;

      Vec const v2 = g * band;
      Vec const low = v2 + s2;
      s2 = low + v2;

      if constexpr (multimodeOutput == lowPassOutput) {
        output[i] = low;
      }
      else if constexpr (multimodeOutput == bandPassOutput) {
        output[i] = band;
      }
      else if constexpr (multimodeOutput == normalizedBandPassOutput) {
        output[i] = band * r;
      }
      else {
        static_assert(false,
                      "multimodeOutput must be a member of the enum "
                      "MultimodeOutputs, except highPassOutput.");
      }
    }

    s1.store_a(state);
    s2.store_a(state + Vec::size());
    g.store_a(frequency);
    r.store_a(resonance);
  }

  template<int multimodeOutput,
           class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative>
  void withAntisaturation(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    int numIterations,
    Saturator saturate,
    SaturationGain saturationGain,
    SaturatorWithDerivative computeSaturationAndDerivative)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + Vec::size());
    Vec u = Vec().load_a(memory);

    Vec g = Vec().load_a(frequency);
    Vec const g_a = Vec().load_a(frequencyTarget);

    Vec r = Vec().load_a(resonance) - 2.0;
    Vec const r_a = Vec().load_a(resonanceTarget) - 2.0;

    Vec const alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec const g_r = r + g;

      Vec const in = input[i];

      // Mistran's cheap method, solving for antisaturated bandpass "u"

      Vec sigma = saturationGain(u); // saturate(u)/u

      Vec d = 1.0 + g * (g_r);

      u = (s1 + g(x - s2)) / (2.0 * g + sigma * d);

      // Newton - Raphson

      Vec g2 = 2.0 * g;
      for (int it = 0; it < numIterations; ++it) {
        Vec band, delta_band_delta_u;
        computeSaturationAndDerivative(u, band, delta_band_delta_u);
        Vec const imp = band * d - g * (in - 2.0 * u - s2) - s1;
        Vec const delta = delta_band_delta_u * d - g2;
        u -= imp / delta;
      }

      Vec band = saturate(u);

      Vec const v2 = g * band;
      Vec const low = v2 + s2;
      s2 = low + v2;
      if constexpr (multimodeOutput == lowPassOutput) {
        output[i] = low;
      }
      else if constexpr (multimodeOutput == bandPassOutput) {
        output[i] = u;
      }
      else if constexpr (multimodeOutput == normalizedBandPassOutput) {
        output[i] = band * r + 2.0 * u;
      }
      else if constexpr (multimodeOutput == highPassOutput) {
        output[i] = in - (g_r * band + s2 + u);
      }
      else {
        static_assert(
          false,
          "multimodeOutput must be a member of the enum MultimodeOutputs.");
      }
    }

    s1.store_a(state);
    s2.store_a(state + Vec::size());
    u.store_a(memory);
    g.store_a(frequency);
    r.store_a(resonance);
  }
};

} // namespace avec