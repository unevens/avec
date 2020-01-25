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

/**
 * An interface for VecBiquadFilter that abstracts over the simd register
 * size. The only methods included in this interface are those to set and get the
 * state and settings of the filter. @see VecBiquadFilter
 */
template<typename Scalar>
class VecBiquadFilterInterface
{
public:
  virtual void Reset(int channel) = 0;
  virtual void Setup(int channel, bool reset) = 0;
  virtual void SetFrequency(int channel, double value, bool update) = 0;
  virtual void SetGain(int channel, double value, bool update) = 0;
  virtual void SetQuality(int channel, double value, bool update) = 0;
  virtual void SetBiquadFilterType(int channel,
                                   BiquadFilterType value,
                                   bool update) = 0;
  virtual double GetFrequency(int channel) const = 0;
  virtual double GetGain(int channel) const = 0;
  virtual double GetQuality(int channel) const = 0;
  virtual BiquadFilterType GetBiquadFilterType(int channel) const = 0;
  virtual void SetState(int channel, Scalar state0, Scalar state1) = 0;
  virtual void GetState(int channel, Scalar& state0, Scalar& state1) const = 0;
};

/**
 * A simple biquad filter working with VecBuffers.
 */
template<class Vec>
class VecBiquadFilter final
  : public VecBiquadFilterInterface<typename ScalarTypes<Vec>::Scalar>
{
public:
  using Scalar = typename ScalarTypes<Vec>::Scalar;

  /**
   * Constructor.
   * @param filterType_ the type of filter (low pass, high shelf, ... @see
   * BiquadFilterType)
   * @param frequency_ the cutoff (angular) frequency of the filter
   * @param quality_ the quality (Q) of the filter, must be >= 0.5
   * @param gain_ the gain of the filter - only used by shelves and peaking
   * filter types
   */
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

  /**
   * Applies the filter to the input and store the result in the output.
   * @param input the input
   * @param output the output
   * @param numSamples the number of samples to process
   */
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

  /**
   * Resets the state of the filter, as if it was processing silence from an
   * eternity.
   */
  void Reset()
  {
    buffer[5] = 0.0;
    buffer[6] = 0.0;
  }

  /**
   * Resets the state of the filter for the specified channel, as if it was
   * processing silence from an eternity.
   * @param channel the channel to reset
   */
  void Reset(int channel) override
  {
    buffer[5][channel] = 0.0;
    buffer[6][channel] = 0.0;
  }

  /**
   * Sets the cutoff frequency on a specific channel
   * @param channel the channel on which to change the cutoff frequency
   * @param value the new cutoff frequency
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetFrequency(int channel, double value, bool update = true) override
  {
    frequency[channel] = value;
    if (update) {
      Setup(channel);
    }
  }

  /**
   * Sets the gain on a specific channel
   * @param channel the channel on which to change the gain
   * @param value the new gain
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetGain(int channel, double value, bool update = true) override
  {
    gain[channel] = value;
    if (update) {
      Setup(channel);
    }
  }

  /**
   * Sets the quality on a specific channel
   * @param channel the channel on which to change the quality
   * @param value the new quality
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetQuality(int channel, double value, bool update = true) override
  {
    quality[channel] = value;
    if (update) {
      Setup(channel);
    }
  }

  /**
   * Sets the filter type on a specific channel
   * @param channel the channel on which to change the filter type
   * @param value the new filter type
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetBiquadFilterType(int channel,
                           BiquadFilterType value,
                           bool update = true) override
  {
    filterType[channel] = value;
    if (update) {
      Setup(channel);
    }
  }

  /**
   * Sets the cutoff frequency on all the channels to the specified value.
   * @param value the new cutoff frequency
   */
  void SetFrequency(double value)
  {
    std::fill(frequency.begin(), frequency.end(), value);
    Setup();
  }

  /**
   * Sets the gain on all the channels to the specified value.
   * @param value the new gain
   */
  void SetGain(double value)
  {
    std::fill(gain.begin(), gain.end(), value);
    Setup();
  }

  /**
   * Sets the quality on all the channels to the specified value.
   * @param value the new quality
   */
  void SetQuality(double value)
  {
    std::fill(quality.begin(), quality.end(), value);
    Setup();
  }

  /**
   * Sets the filter type on all the channels to the specified value.
   * @param value the new filter type
   */
  void SetBiquadFilterType(BiquadFilterType value)
  {
    std::fill(filterType.begin(), filterType.end(), value);
    Setup();
  }

  /**
   * Gets the cutoff frequency used for the specified channel.
   * @param channel the channel to get the cutoff frequency from
   * @return the cutoff frequency of the specified channel.
   */
  double GetFrequency(int channel) const override { return frequency[channel]; }

  /**
   * Gets the gain used for the specified channel.
   * @param channel the channel to get the gain from
   * @return the gain of the specified channel.
   */
  double GetGain(int channel) const override { return gain[channel]; }

  /**
   * Gets the quality used for the specified channel.
   * @param channel the channel to get the quality from
   * @return the quality of the specified channel.
   */
  double GetQuality(int channel) const override { return quality[channel]; }

  /**
   * Gets the filter type used for the specified channel.
   * @param channel the channel to get the filter type from
   * @return the filter type of the specified channel.
   */
  BiquadFilterType GetBiquadFilterType(int channel) const override
  {
    return filterType[channel];
  }

  /**
   * Computes the filter coefficients for a specified channel.
   * @param channel the channel to compute the coefficients the coeffcient for
   * @param reset if true, as it is by default, resets the state of filter,
   * calling Reset().
   */
  void Setup(int channel, bool reset = true) override
  {
    double a1 = 0.0;
    double a2 = 0.0;
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double g = pow(10.0, gain[channel] / 40.0);
    double cs = cos(frequency[channel]);
    double sn = sin(frequency[channel]);
    double alpha = sn / (2.0 * quality[channel]);
    double sq = 2.0 * sqrt(g) * alpha;
    switch (filterType[channel]) {
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
    buffer[0][channel] = a1;
    buffer[1][channel] = a2;
    buffer[2][channel] = b0;
    buffer[3][channel] = b1;
    buffer[4][channel] = b2;

    if (reset) {
      Reset(channel);
    }
  }

  /**
   * Computes the filter coefficients.
   * @param reset if true, as it is by default, resets the state of filter,
   * calling Reset().
   */
  void Setup(bool reset = true)
  {
    for (int i = 0; i < Vec::size(); ++i) {
      Setup(i, false);
    }
    if (reset) {
      Reset();
    }
  }

  /**
   * Sets the state of the filter for a specified channel.
   * @param channel the channel for which to set the state of the filter
   * @param state0 the first number of the state of the filter
   * @param state1 the second number of the state of the filter
   */
  void SetState(int channel, Scalar state0, Scalar state1) override
  {
    buffer[5][channel] = state0;
    buffer[6][channel] = state1;
  }

  /**
   * Gets the state of the filter for a specified channel.
   * @param channel the channel for which to set the state of the filter
   * @param state0 the first number of the state of the filter
   * @param state1 the second number of the state of the filter
   */
  void GetState(int channel, Scalar& state0, Scalar& state1) const override
  {
    state0 = buffer[5][channel];
    state1 = buffer[6][channel];
  }

private:
  VecBuffer<Vec> buffer;
  std::array<double, Vec::size()> frequency;
  std::array<double, Vec::size()> quality;
  std::array<double, Vec::size()> gain;
  std::array<BiquadFilterType, Vec::size()> filterType;
};

/**
 * A simple biquad filter working with InterleavedBuffers.
 */
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
  /**
   * Constructor.
   * @param numChannels the number of channels to allocate resources for
   * @param filterType_ the type of filter (low pass, high shelf, ... @see
   * BiquadFilterType)
   * @param frequency_ the cutoff (angular) frequency of the filter
   * @param quality_ the quality (Q) of the filter, must be >= 0.5
   * @param gain_ the gain of the filter - only used by shelves and peaking
   * filter types
   */
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

  /**
   * Applies the filter to the input and store the result in the output.
   * @param input the input
   * @param output the output
   * @param numSamples the number of samples to process
   * @param numChannelsToProcess the number of channels to process
   */
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

  /**
   * Resets the state of the filter, as if it was processing silence from an
   * eternity.
   */
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

  /**
   * Sets the cutoff frequency on a specific channel
   * @param channel the channel on which to change the cutoff frequency
   * @param value the new cutoff frequency
   */
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

  /**
   * Sets the gain on a specific channel
   * @param channel the channel on which to change the gain
   * @param value the new gain
   */
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

  /**
   * Sets the quality on a specific channel
   * @param channel the channel on which to change the quality
   * @param value the new quality
   */
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

  /**
   * Sets the filter type on a specific channel
   * @param channel the channel on which to change the filter type
   * @param value the new filter type
   */
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

  /**
   * Sets the cutoff frequency on all the channels to the specified value.
   * @param value the new cutoff frequency
   */
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

  /**
   * Sets the gain on all the channels to the specified value.
   * @param value the new gain
   */
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

  /**
   * Sets the quality on all the channels to the specified value.
   * @param value the new quality
   */
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

  /**
   * Sets the filter type on all the channels to the specified value.
   * @param value the new filter type
   */
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

  /**
   * Gets the cutoff frequency used for the specified channel.
   * @param channel the channel to get the cutoff frequency from
   * @return the cutoff frequency of the specified channel.
   */
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

  /**
   * Gets the gain used for the specified channel.
   * @param channel the channel to get the gain from
   * @return the gain of the specified channel.
   */
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

  /**
   * Gets the quality used for the specified channel.
   * @param channel the channel to get the quality from
   * @return the quality of the specified channel.
   */
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

  /**
   * Gets the filter type used for the specified channel.
   * @param channel the channel to get the filter type from
   * @return the filter type of the specified channel.
   */
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

  /**
   * Resets the state of the filter for a sequence of channels.
   * @param srcChannelStart the first channel to be reset
   * @param numChannelsToReset the number of channels to reset
   */
  void Reset(int srcChannelStart, int numChannelsToReset)
  {
    for (int i = srcChannelStart; i < srcChannelStart + numChannelsToReset;
         ++i) {

      if constexpr (VEC8_AVAILABLE) {
        auto d8 = std::div(i, 8);

#if AVEC_MIX_VEC_SIZES

        VecBiquadFilterInterface<Scalar>* filter =
          d8.quot < filters8.size()
            ? static_cast<VecBiquadFilterInterface<Scalar>*>(&filters8[d8.quot])
            : static_cast<VecBiquadFilterInterface<Scalar>*>(&filters4[0]);

#else

        VecBiquadFilter<Vec8>* filter = &filters8[d8.quot];

#endif

        filter->Reset(d8.rem);
      }
      else if constexpr (VEC4_AVAILABLE) {
        auto d4 = std::div(i, 4);
        filters4[d4.quot].Reset(d4.rem);
      }
      else {
        auto d2 = std::div(i, 2);
        filters2[d4.quot].Reset(d2.rem);
      }
    }
  }

  /**
   * Copies the state and settings of the filter from sequence of channels to
   * an other sequence of channels, and resets the latter.
   * @param srcChannelStart the first channel to be moved
   * @param dstChannelStart the destination for the first channel to be moved
   * @param numChannelsToMove the number of channels to move
   */
  void MoveChannelStates(int srcChannelStart,
                         int dstChannelStart,
                         int numChannelsToMove)
  {
    double srcFrequency;
    double srcQuality;
    double srcGain;
    BiquadFilterType srcType;

    Scalar srcState0;
    Scalar srcState1;

    for (int i = 0; i < numChannelsToMove; ++i) {
      int srcChannel = srcChannelStart + i;
      int dstChannel = dstChannelStart + i;

      if constexpr (VEC8_AVAILABLE) {
        auto s8 = std::div(srcChannel, 8);
        auto d8 = std::div(dstChannel, 8);

#if AVEC_MIX_VEC_SIZES

        VecBiquadFilterInterface<Scalar>* srcFilter =
          s8.quot < filters8.size()
            ? static_cast<VecBiquadFilterInterface<Scalar>*>(&filters8[s8.quot])
            : static_cast<VecBiquadFilterInterface<Scalar>*>(&filters4[0]);
        VecBiquadFilterInterface<Scalar>* dstFilter =
          d8.quot < filters8.size()
            ? static_cast<VecBiquadFilterInterface<Scalar>*>(&filters8[d8.quot])
            : static_cast<VecBiquadFilterInterface<Scalar>*>(&filters4[0]);

#else

        VecBiquadFilter<Vec8>* srcFilter = &filters8[s8.quot];
        VecBiquadFilter<Vec8>* dstFilter = &filters8[d8.quot];

#endif

        int src = s8.rem;
        int dst = d8.rem;

        srcFrequency = srcFilter->GetFrequency(src);
        srcQuality = srcFilter->GetQuality(src);
        srcGain = srcFilter->GetGain(src);
        srcType = srcFilter->GetBiquadFilterType(src);

        bool needsSetup = (srcFrequency != dstFilter->GetFrequency(dst)) ||
                          (srcQuality != dstFilter->GetQuality(dst)) ||
                          (srcGain != dstFilter->GetGain(dst)) ||
                          (srcType != dstFilter->GetBiquadFilterType(dst));

        dstFilter->SetFrequency(dst, srcFrequency, false);
        dstFilter->SetQuality(dst, srcQuality, false);
        dstFilter->SetGain(dst, srcGain, false);
        dstFilter->SetBiquadFilterType(dst, srcType, false);

        if (needsSetup) {
          dstFilter->Setup(dst, false);
        }

        srcFilter->GetState(src, srcState0, srcState1);
        dstFilter->SetState(dst, srcState0, srcState1);
        srcFilter->Reset(src);
      }
      else if constexpr (VEC4_AVAILABLE) {
        auto s4 = std::div(srcChannel, 4);
        auto d4 = std::div(dstChannel, 4);
        auto& srcFilter = filters4[s4.quot];
        auto& dstFilter = filters4[d4.quot];
        int src = s4.rem;
        int dst = d4.rem;

        srcFrequency = srcFilter.GetFrequency(src);
        srcQuality = srcFilter.GetQuality(src);
        srcGain = srcFilter.GetGain(src);
        srcType = srcFilter.GetBiquadFilterType(src);

        bool needsSetup = (srcFrequency != dstFilter.GetFrequency(dst)) ||
                          (srcQuality != dstFilter.GetQuality(dst)) ||
                          (srcGain != dstFilter.GetGain(dst)) ||
                          (srcType != dstFilter.GetBiquadFilterType(dst));

        dstFilter.SetFrequency(dst, srcFrequency, false);
        dstFilter.SetQuality(dst, srcQuality, false);
        dstFilter.SetGain(dst, srcGain, false);
        dstFilter.SetBiquadFilterType(dst, srcType, false);

        if (needsSetup) {
          dstFilter.Setup(dst, false);
        }

        srcFilter.GetState(src, srcState0, srcState1);
        dstFilter.SetState(dst, srcState0, srcState1);
        srcFilter.Reset(src);
      }
      else {
        auto s2 = std::div(srcChannel, 2);
        auto d2 = std::div(dstChannel, 2);
        auto& srcFilter = filters2[s2.quot];
        auto& dstFilter = filters2[d2.quot];
        int src = s2.rem;
        int dst = d2.rem;

        srcFrequency = srcFilter.GetFrequency(src);
        srcQuality = srcFilter.GetQuality(src);
        srcGain = srcFilter.GetGain(src);
        srcType = srcFilter.GetBiquadFilterType(src);

        bool needsSetup = (srcFrequency != dstFilter.GetFrequency(dst)) ||
                          (srcQuality != dstFilter.GetQuality(dst)) ||
                          (srcGain != dstFilter.GetGain(dst)) ||
                          (srcType != dstFilter.GetBiquadFilterType(dst));

        dstFilter.SetFrequency(dst, srcFrequency, false);
        dstFilter.SetQuality(dst, srcQuality, false);
        dstFilter.SetGain(dst, srcGain, false);
        dstFilter.SetBiquadFilterType(dst, srcType, false);

        if (needsSetup) {
          dstFilter.Setup(dst, false);
        }

        srcFilter.GetState(src, srcState0, srcState1);
        dstFilter.SetState(dst, srcState0, srcState1);
        srcFilter.Reset(src);
      }
    }
  }
};

} // namespace avec