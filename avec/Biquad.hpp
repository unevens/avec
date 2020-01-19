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
#include "avec/InterleavedBuffer.hpp"
#include <array>

namespace avec {

enum class BiquadFilterType
{
  LowPass = 0,
  HighPass,
  LowShelf,
  HighShelf,
  BandPass,
  Peak,
  Notch,
  AllPass
};

template<typename Vec>
class VecBiquadFilter
{
public:
  VecBiquadFilter(BiquadFilterType filterType_,
                  double frequency_ = 0.1,
                  double quality_ = 0.79,
                  double gain_ = 0.0)
    : buffer(7 * Vec::size())
  {
    std::fill(filterType.begin(), filterType.end(), filterType_);
    std::fill(frequency.begin(), frequency.end(), frequency_);
    std::fill(quality.begin(), quality.end(), quality_);
    std::fill(gain.begin(), gain.end(), gain_);
    Setup();
  }

  void ProcessBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int numSamples)
  {
    Vec a1 = buffer[0];
    Vec a2 = buffer[1];
    Vec b0 = buffer[2];
    Vec b1 = buffer[3];
    Vec b2 = buffer[4];
    Vec prev0 = buffer[5];
    Vec prev1 = buffer[6];
    for (int i = 0; i < numSamples; ++i) {
      Vec in = input[i];
      Vec next_buffer_0 = in - a1 * prev0 - a2 * prev1;
      Vec out = b0 * next_buffer_0 + b1 * prev0 + b2 * prev1;
      prev1 = prev0;
      prev0 = next_buffer_0;
      output[i] = out;
    }
    buffer[5] = prev0;
    buffer[6] = prev1;
  }

  void Reset()
  {
    buffer[5] = 0.0;
    buffer[6] = 0.0;
  }

  void SetFrequency(int channel, double value)
  {
    frequency[channel] = value;
    Setup();
  }

  void SetGain(int channel, double value)
  {
    gain[channel] = value;
    Setup();
  }

  void SetQuality(int channel, double value)
  {
    quality[channel] = value;
    Setup();
  }

  void SetBiquadFilterType(int channel, BiquadFilterType value)
  {
    filterType[channel] = value;
    Setup();
  }

  void SetFrequency(double value)
  {
    std::fill(frequency.begin(), frequency.end(), value);
    Setup();
  }

  void SetGain(double value)
  {
    std::fill(gain.begin(), gain.end(), value);
    Setup();
  }

  void SetQuality(double value)
  {
    std::fill(quality.begin(), quality.end(), value);
    Setup();
  }

  void SetBiquadFilterType(BiquadFilterType value)
  {
    std::fill(filterType.begin(), filterType.end(), value);
    Setup();
  }

  double GetFrequency(int channel) const { return frequency[channel]; }

  double GetGain(int channel) const { return gain[channel]; }

  double GetQuality(int channel) const { return quality[channel]; }

  BiquadFilterType GetBiquadFilterType(int channel) const
  {
    return filterType[channel];
  }

private:
  void Setup()
  {
    for (int i = 0; i < Vec::size(); ++i) {
      double a1 = 0.0;
      double a2 = 0.0;
      double b0 = 0.0;
      double b1 = 0.0;
      double b2 = 0.0;
      double g = pow(10.0, gain[i] / 40.0);
      double cs = cos(frequency[i]);
      double sn = sin(frequency[i]);
      double alpha = sn / (2.0 * quality[i]);
      double sq = 2.0 * sqrt(g) * alpha;
      switch (filterType[i]) {
        case BiquadFilterType::HighShelf: {
          double a0inv = 1.0 / ((g + 1.0) - (g - 1.0) * cs + sq);
          b0 = a0inv * g * ((g + 1.0) + (g - 1.0) * cs + sq);
          b1 = -2.0 * a0inv * g * ((g - 1.0) + (g + 1.0) * cs);
          b2 = a0inv * g * ((g + 1.0) + (g - 1.0) * cs - sq);
          a1 = 2.0 * a0inv * ((g - 1.0) - (g + 1.0) * cs);
          a2 = a0inv * ((g + 1.0) - (g - 1.0) * cs - sq);
        } break;
        case BiquadFilterType::HighPass: {
          double a0inv = 1.0 / (1.0 + alpha);
          b0 = a0inv * 0.5 * (1.0 + cs);
          b1 = a0inv * (-1.0 - cs);
          b2 = a0inv * 0.5 * (1.0 + cs);
          a1 = -2.0 * a0inv * cs;
          a2 = (1.0 - alpha) * a0inv;
        } break;
        case BiquadFilterType::LowShelf: {
          double a0inv = 1.0 / ((g + 1.0) + (g - 1.0) * cs + sq);
          b0 = a0inv * g * ((g + 1.0) - (g - 1.0) * cs + sq);
          b1 = 2.0 * a0inv * g * ((g - 1.0) - (g + 1.0) * cs);
          b2 = a0inv * g * ((g + 1.0) - (g - 1.0) * cs - sq);
          a1 = -2.0 * a0inv * ((g - 1.0) + (g + 1.0) * cs);
          a2 = a0inv * ((g + 1.0) + (g - 1.0) * cs - sq);
        } break;
        case BiquadFilterType::LowPass: {
          double a0inv = 1.0 / (1.0 + alpha);
          b0 = a0inv * (1.0 - cs) * 0.5;
          b1 = a0inv * (1.0 - cs);
          b2 = a0inv * (1.0 - cs) * 0.5;
          a1 = a0inv * (-2.0 * cs);
          a2 = a0inv * (1.0 - alpha);
        } break;
        case BiquadFilterType::Peak: {
          double a0inv = 1.0 / (1.0 + alpha / g);
          b0 = a0inv * (1.0 + alpha * g);
          b1 = a0inv * (-2.0 * cs);
          b2 = a0inv * (1.0 - alpha * g);
          a1 = a0inv * (-2.0 * cs);
          a2 = a0inv * (1.0 - alpha / g);
        } break;
        case BiquadFilterType::BandPass: {
          double a0inv = 1.0 / (1.0 + alpha);
          b0 = a0inv * alpha;
          b1 = 0.0;
          b2 = -a0inv * alpha;
          a1 = -2.0 * a0inv * cs;
          a2 = a0inv * (1.0 - alpha);
        } break;
        case BiquadFilterType::Notch: {
          double a0inv = 1.0 / (1.0 + alpha);
          b0 = a0inv;
          b1 = -2.0 * a0inv * cs;
          b2 = a0inv;
          a1 = -2.0 * a0inv * cs;
          a2 = a0inv * (1.0 - alpha);
        } break;
        case BiquadFilterType::AllPass: {
          double a0inv = 1.0 / (1.0 + alpha);
          b0 = a0inv * (1.0 - alpha);
          b1 = -2.0 * a0inv * cs;
          b2 = a0inv * (1.0 + alpha);
          a1 = -2.0 * a0inv * cs;
          a2 = a0inv * (1.0 - alpha);
        } break;
        default:
          assert(false);
      }
      buffer[0][i] = a1;
      buffer[1][i] = a2;
      buffer[2][i] = b0;
      buffer[3][i] = b1;
      buffer[4][i] = b2;
    }
    Reset();
  }

  VecBuffer<Vec> buffer;
  std::array<double, Vec::size()> frequency;
  std::array<double, Vec::size()> quality;
  std::array<double, Vec::size()> gain;
  std::array<BiquadFilterType, Vec::size()> filterType;
};

template<typename Scalar>
class BiquadFilter final
{
  using Vec8 = typename SimdTypes<Scalar>::Vec8;
  using Vec4 = typename SimdTypes<Scalar>::Vec4;
  using Vec2 = typename SimdTypes<Scalar>::Vec2;
  static constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;
  int numChannels;
  std::vector<VecBiquadFilter<Vec8>> filters8;
  std::vector<VecBiquadFilter<Vec4>> filters4;
  std::vector<VecBiquadFilter<Vec2>> filters2;

public:
  BiquadFilter(int numChannels,
               BiquadFilterType filterType_ = BiquadFilterType::LowPass,
               double frequency_ = 0.1,
               double quality_ = 0.79,
               double gain_ = 0.0)
    : numChannels(numChannels)
  {
    int num8, num4, num2;
    if constexpr (VEC8_AVAILABLE) {
      auto d8 = std::div(numChannels, 8);
#if AVEC_MIX_VEC_SIZES
      num8 = (int)d8.quot + (d8.rem > 4 ? 1 : 0);
      num4 = (d8.rem <= 4) ? 1 : 0;
#else
      num8 = (int)d8.quot + (d8.rem > 0 ? 1 : 0);
      num4 = 0;
#endif
      num2 = 0;
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d4 = std::div(numChannels, 4);
      num8 = 0;
      num4 = (int)d4.quot + (d4.rem > 0 ? 1 : 0);
      num2 = 0;
    }
    else {
      auto d4 = std::div(numChannels, 4);
      num8 = 0;
      num2 = (int)d4.quot + (d4.rem > 0 ? 1 : 0);
      num4 = 0;
    }
    filters8.reserve(num8);
    filters4.reserve(num4);
    filters2.reserve(num2);
    for (int i = 0; i < num8; ++i) {
      filters8.push_back(
        VecBiquadFilter<Vec8>(filterType_, frequency_, quality_, gain_));
    }
    for (int i = 0; i < num4; ++i) {
      filters4.push_back(
        VecBiquadFilter<Vec4>(filterType_, frequency_, quality_, gain_));
    }
    for (int i = 0; i < num2; ++i) {
      filters2.push_back(
        VecBiquadFilter<Vec2>(filterType_, frequency_, quality_, gain_));
    }
  }

  void ProcessBlock(InterleavedBuffer<Scalar> const& input,
                    InterleavedBuffer<Scalar>& output,
                    int numSamples,
                    int numChannelsToProcess)
  {
    assert(numChannelsToProcess <= numChannels);
    assert(numSamples <= input.GetNumSamples());

    output.SetNumSamples(numSamples);
    if constexpr (VEC8_AVAILABLE) {
      for (int i = 0; i < filters8.size(); ++i) {
        filters8[i].ProcessBlock(
          input.GetBuffer8(i), output.GetBuffer8(i), numSamples);
        numChannelsToProcess -= 8;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
      if (filters4.size() > 0) {
        filters4[0].ProcessBlock(
          input.GetBuffer4(0), output.GetBuffer4(0), numSamples);
        numChannelsToProcess -= 4;
      }
    }
    else if constexpr (VEC4_AVAILABLE) {
      for (int i = 0; i < filters4.size(); ++i) {
        filters4[i].ProcessBlock(
          input.GetBuffer4(i), output.GetBuffer4(i), numSamples);
        numChannelsToProcess -= 4;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    else {
      int lastBuffers2 = output.GetNumBuffers2() - 1;
      for (int i = 0; i < filters2.size(); ++i) {
        filters2[i].ProcessBlock(
          input.GetBuffer2(i), output.GetBuffer2(i), numSamples);
        numChannelsToProcess -= 2;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
  }

  void Reset()
  {
    for (auto& f : filters8) {
      f.Reset();
    }
    for (auto& f : filters4) {
      f.Reset();
    }
    for (auto& f : filters2) {
      f.Reset();
    }
  }

  void SetFrequency(int channel, double value)
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        filters8[d.quot].SetFrequency(d.rem, value);
      }
      else {
        assert(d.quot == filters8.size());
        filters4[0].SetFrequency(d.rem, value);
      }
#else
      filters8[d.quot].SetFrequency(d.rem, value);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      filters4[d.quot].SetFrequency(d.rem, value);
    }
    else {
      auto d = std::div(channel, 2);
      filters2[d.quot].SetFrequency(d.rem, value);
    }
  }

  void SetGain(int channel, double value)
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        filters8[d.quot].SetGain(d.rem, value);
      }
      else {
        assert(d.quot == filters8.size());
        filters4[0].SetGain(d.rem, value);
      }
#else
      filters8[d.quot].SetGain(d.rem, value);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      filters4[d.quot].SetGain(d.rem, value);
    }
    else {
      auto d = std::div(channel, 2);
      filters2[d.quot].SetGain(d.rem, value);
    }
  }

  void SetQuality(int channel, double value)
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        filters8[d.quot].SetQuality(d.rem, value);
      }
      else {
        assert(d.quot == filters8.size());
        filters4[0].SetQuality(d.rem, value);
      }
#else
      filters8[d.quot].SetQuality(d.rem, value);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      filters4[d.quot].SetQuality(d.rem, value);
    }
    else {
      auto d = std::div(channel, 2);
      filters2[d.quot].SetQuality(d.rem, value);
    }
  }

  void SetBiquadFilterType(int channel, BiquadFilterType value)
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        filters8[d.quot].SetBiquadFilterType(d.rem, value);
      }
      else {
        assert(d.quot == filters8.size());
        filters4[0].SetBiquadFilterType(d.rem, value);
      }
#else
      filters8[d.quot].SetBiquadFilterType(d.rem, value);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      filters4[d.quot].SetBiquadFilterType(d.rem, value);
    }
    else {
      auto d = std::div(channel, 2);
      filters2[d.quot].SetBiquadFilterType(d.rem, value);
    }
  }

  void SetFrequency(double value)
  {
    for (auto& f : filters8) {
      f.SetFrequency(value);
    }
    for (auto& f : filters4) {
      f.SetFrequency(value);
    }
    for (auto& f : filters2) {
      f.SetFrequency(value);
    }
  }

  void SetGain(double value)
  {
    for (auto& f : filters8) {
      f.SetGain(value);
    }
    for (auto& f : filters4) {
      f.SetGain(value);
    }
    for (auto& f : filters2) {
      f.SetGain(value);
    }
  }

  void SetQuality(double value)
  {
    for (auto& f : filters8) {
      f.SetQuality(value);
    }
    for (auto& f : filters4) {
      f.SetQuality(value);
    }
    for (auto& f : filters2) {
      f.SetQuality(value);
    }
  }

  void SetBiquadFilterType(BiquadFilterType value)
  {
    for (auto& f : filters8) {
      f.SetBiquadFilterType(value);
    }
    for (auto& f : filters4) {
      f.SetBiquadFilterType(value);
    }
    for (auto& f : filters2) {
      f.SetBiquadFilterType(value);
    }
  }

  double GetFrequency(int channel) const
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        return filters8[d.quot].GetFrequency(d.rem);
      }
      else {
        assert(d.quot == filters8.size());
        return filters4[0].GetFrequency(d.rem);
      }
#else
      return filters8[d.quot].GetFrequency(d.rem);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      return filters4[d.quot].GetFrequency(d.rem);
    }
    else {
      auto d = std::div(channel, 2);
      return filters2[d.quot].GetFrequency(d.rem);
    }
  }

  double GetGain(int channel) const
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        return filters8[d.quot].GetGain(d.rem);
      }
      else {
        assert(d.quot == filters8.size());
        return filters4[0].GetGain(d.rem);
      }
#else
      return filters8[d.quot].GetGain(d.rem);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      return filters4[d.quot].GetGain(d.rem);
    }
    else {
      auto d = std::div(channel, 2);
      return filters2[d.quot].GetGain(d.rem);
    }
  }

  double GetQuality(int channel) const
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        return filters8[d.quot].GetQuality(d.rem);
      }
      else {
        assert(d.quot == filters8.size());
        return filters4[0].GetQuality(d.rem);
      }
#else
      return filters8[d.quot].GetQuality(d.rem);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      return filters4[d.quot].GetQuality(d.rem);
    }
    else {
      auto d = std::div(channel, 2);
      return filters2[d.quot].GetQuality(d.rem);
    }
  }

  BiquadFilterType GetBiquadFilterType(int channel) const
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(channel, 8);
#if AVEC_MIX_VEC_SIZES
      if (d.quot < filters8.size()) {
        return filters8[d.quot].GetBiquadFilterType(d.rem);
      }
      else {
        assert(d.quot == filters8.size());
        return filters4[0].GetBiquadFilterType(d.rem);
      }
#else
      return filters8[d.quot].GetBiquadFilterType(d.rem);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(channel, 4);
      return filters4[d.quot].GetBiquadFilterType(d.rem);
    }
    else {
      auto d = std::div(channel, 2);
      return filters2[d.quot].GetBiquadFilterType(d.rem);
    }
  }

  /**
   * @return the maximum number of channel that the generator can work with.
   */
  int GetNumChannels() const { return numChannels; }
};

} // namespace avec