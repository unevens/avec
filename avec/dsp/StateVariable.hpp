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

//#include <JuceHeader.h>
#define DBG(x)

namespace avec {

template<class Vec>
struct StateVariable
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  static constexpr Scalar pi = 3.141592653589793238;

  enum class Output
  {
    lowPass = 0,
    highPass,
    bandPass,
    normalizedBandPass
  };

  Scalar smoothingAlpha[Vec::size()];
  Scalar state[2 * Vec::size()];
  Scalar memory[Vec::size()]; // for antisaturator
  Scalar frequency[Vec::size()];
  Scalar resonance[Vec::size()];
  Scalar frequencyTarget[Vec::size()];
  Scalar resonanceTarget[Vec::size()];
  Scalar outputMode[Vec::size()];

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

  void setOutput(Output output, int channel)
  {
    outputMode[channel] = static_cast<int>(output);
  }

  void setOutput(Output output)
  {
    std::fill_n(outputMode, Vec::size(), static_cast<int>(output));
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

  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<-1>(input, output);
  }

  void bandPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<static_cast<int>(Output::bandPass)>(input, output);
  }

  void lowPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<static_cast<int>(Output::lowPass)>(input, output);
  }

  void normalizedBandPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<static_cast<int>(Output::normalizedBandPass)>(input, output);
  }

  // nonlinear

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int numIterations,
                    Saturator saturate,
                    SaturationGain saturationGain,
                    SaturatorWithDerivative computeSaturationAndDerivative,
                    SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<-1>(input,
                           output,
                           numIterations,
                           saturate,
                           saturationGain,
                           computeSaturationAndDerivative,
                           saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void lowPass(VecBuffer<Vec> const& input,
               VecBuffer<Vec>& output,
               int numIterations,
               Saturator saturate,
               SaturationGain saturationGain,
               SaturatorWithDerivative computeSaturationAndDerivative,
               SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::lowPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void bandPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturate,
                SaturationGain saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative,
                SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::bandPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void normalizedBandPass(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    int numIterations,
    Saturator saturate,
    SaturationGain saturationGain,
    SaturatorWithDerivative computeSaturationAndDerivative,
    SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::normalizedBandPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void highPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturate,
                SaturationGain saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative,
                SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::highPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

private:
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
  void linear(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
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

    Vec const output_mode = Vec().load_a(outputMode);
    auto const is_high_pass = output_mode == static_cast<int>(Output::highPass);
    auto const is_band_pass = output_mode == static_cast<int>(Output::bandPass);
    auto const is_nrm_band_pass =
      output_mode == static_cast<int>(Output::normalizedBandPass);

    if constexpr (multimodeOutput == static_cast<int>(Output::highPass)) {

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
    }

    else {

      for (int i = 0; i < numSamples; ++i) {

        g = alpha * (g - g_a) + g_a;
        r = alpha * (r - r_a) + r_a;

        Vec const in = input[i];

        Vec const band = (g * (in - s2) + s1) / (1.0 + g * (r + g));

        s1 = band + band - s1;

        Vec const v2 = g * band;
        Vec const low = v2 + s2;
        s2 = low + v2;

        if constexpr (multimodeOutput == -1) {
          Vec normalized_band = band * r;
          Vec high = in - (g * r * band + s2);

          output[i] = select(is_band_pass,
                             band,
                             select(is_nrm_band_pass,
                                    normalized_band,
                                    select(is_high_pass, high, low)));
        }
        else if constexpr (multimodeOutput ==
                           static_cast<int>(Output::lowPass)) {
          output[i] = low;
        }
        else if constexpr (multimodeOutput ==
                           static_cast<int>(Output::bandPass)) {
          output[i] = band;
        }
        else if constexpr (multimodeOutput ==
                           static_cast<int>(Output::normalizedBandPass)) {
          output[i] = band * r;
        }
        else {
          static_assert(false, "Wrong multimodeOutput.");
        }
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
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void withAntisaturation(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    int numIterations,
    Saturator saturate,
    SaturationGain saturationGain,
    SaturatorWithDerivative computeSaturationAndDerivative,
    SaturatorAutomation saturatorAutomation)
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

    Vec const output_mode = Vec().load_a(outputMode);
    auto const is_high_pass = output_mode == static_cast<int>(Output::highPass);
    auto const is_band_pass = output_mode == static_cast<int>(Output::bandPass);
    auto const is_nrm_band_pass =
      output_mode == static_cast<int>(Output::normalizedBandPass);

    for (int i = 0; i < numSamples; ++i) {

      saturatorAutomation();

      g = alpha * (g - g_a) + g_a;
      r = alpha * (r - r_a) + r_a;

      Vec const g_r = r + g;
      Vec g_2 = g + g;
      Vec const d = 1.0 + g * (g_r);

      Vec const in = input[i];

      // Mistran's cheap method, solving for antisaturated bandpass "u"

      DBG(String("prev u: ") + String(u[0]));
      DBG(String("prev s1: ") + String(s1[0]));
      DBG(String("prev s2: ") + String(s2[0]));

      Vec sigma = saturationGain(u); // saturate(u)/u

      DBG(String("sigma: ") + String(sigma[0]));

      u = (s1 + g * (in - s2)) / (sigma * d + g_2);

      // Newton - Raphson

      DBG(String("m u: ") + String(u[0]));

      for (int it = 0; it < numIterations; ++it) {
        Vec band, delta_band_delta_u;
        computeSaturationAndDerivative(u, band, delta_band_delta_u);
        Vec const imp = band * d - g * (in - (u + u) - s2) - s1;
        Vec const delta = delta_band_delta_u * d + g_2;
        u -= imp / delta;
        DBG(String("n u ") + String(it) + String(": ") + String(u[0]) +
            String(", ") + String(imp[0]) + String(", ") + String(delta[0]));
      }

      Vec band = saturate(u);
      DBG(String("band ") + String(band[0]));

      s1 = band + band - s1;
      DBG(String("s1") + String(s1[0]));

      Vec const v2 = g * band;
      Vec const low = v2 + s2;
      s2 = low + v2;
      DBG(String("low ") + String(low[0]));
      DBG(String("s2 ") + String(s2[0]));

      if constexpr (multimodeOutput == -1) {
        Vec normalized_band = band * r + 2.0 * u;
        Vec high = in - (g_r * band + s2 + u);

        output[i] = select(is_band_pass,
                           band,
                           select(is_nrm_band_pass,
                                  normalized_band,
                                  select(is_high_pass, high, low)));
      }
      else if constexpr (multimodeOutput == static_cast<int>(Output::lowPass)) {
        output[i] = low;
      }
      else if constexpr (multimodeOutput ==
                         static_cast<int>(Output::bandPass)) {
        output[i] = band;
      }
      else if constexpr (multimodeOutput ==
                         static_cast<int>(Output::normalizedBandPass)) {
        output[i] = band * r + 2.0 * u;
      }
      else if constexpr (multimodeOutput ==
                         static_cast<int>(Output::highPass)) {
        output[i] = in - (g_r * band + s2 + u);
      }
      else {
        static_assert(false, "Wrong multimodeOutput.");
      }
    }

    s1.store_a(state);
    s2.store_a(state + Vec::size());
    u.store_a(memory);
    g.store_a(frequency);
    r += 2.0;
    r.store_a(resonance);
  }
};

} // namespace avec