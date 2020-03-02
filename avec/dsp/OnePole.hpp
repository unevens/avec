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
struct OnePole
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  static constexpr Scalar pi = 3.141592653589793238;

  Scalar smoothingAlpha[Vec::size()];
  Scalar state[Vec::size()];
  Scalar frequency[Vec::size()];
  Scalar frequencyTarget[Vec::size()];

  OnePole()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    setFrequency(0.25);
    reset();
  }

  void reset()
  {
    std::copy(frequencyTarget, frequencyTarget + Vec::size(), frequency);
    std::fill_n(state, Vec::size(), 0.0);
  }

  void setFrequency(Scalar normalized, int channel)
  {
    frequencyTarget[channel] = tan(pi * normalized);
  }

  void setFrequency(Scalar normalized)
  {
    std::fill_n(frequencyTarget, Vec::size(), tan(pi * normalized));
  }

  void setSmoothingAlpha(Scalar alpha)
  {
    std::fill_n(smoothingAlpha, Vec::size(), alpha);
  }

  void lowPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    processBlock<1>(input, output);
  }

  void highPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    processBlock<0>(input, output);
  }

private:
  template<int isLowPass>
  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s = Vec().load_a(state);
    Vec g = Vec().load_a(frequency);
    Vec g_a = Vec().load_a(frequencyTarget);
    Vec alpha = Vec().load_a(smoothingAlpha);

    for (int i = 0; i < numSamples; ++i) {

      g = alpha * (g - g_a) + g_a;

      Vec in = input[i];

      Vec v = g * (in - s) / (1.0 + g);
      Vec low = v + s;

      s = low + v;

      if constexpr (isLowPass) {
        output[i] = low;
      }
      else {
        output[i] = in - low;
      }
    }

    s.store_a(state);
    g.store_a(frequency);
  }
};

} // namespace avec