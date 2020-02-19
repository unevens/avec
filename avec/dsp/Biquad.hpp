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
 * size. The only methods included in this interface are those to set and get
 * the state and settings of the filter. @see VecBiquadFilter
 */
template<typename Scalar>
class VecBiquadFilterInterface
{
public:
  virtual void Reset(int channel) = 0;
  virtual void Setup(int channel, bool reset, bool automate = true) = 0;
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
    : buffer(12 * Vec::size())
    , isSetupNeeded(Vec::size(), 0)
  {
    std::fill(filterType.begin(), filterType.end(), filterType_);
    std::fill(frequency.begin(), frequency.end(), frequency_);
    std::fill(quality.begin(), quality.end(), quality_);
    std::fill(gain.begin(), gain.end(), gain_);
    Setup(true);
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

    if (!isAutomating) {
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
    else {
      isAutomating = false;

      Vec a1_aut = buffer[7];
      Vec a2_aut = buffer[8];
      Vec b0_aut = buffer[9];
      Vec b1_aut = buffer[10];
      Vec b2_aut = buffer[11];
      Vec prev0_aut = 0.0;
      Vec prev1_aut = 0.0;

      Vec alpha = 0.f;
      Vec inc = 1.f / (float)numSamples;

      for (int i = 0; i < numSamples; ++i) {
        Vec in = input[i];

        Vec next_buffer_0 = in - a1 * prev0 - a2 * prev1;
        Vec out = b0 * next_buffer_0 + b1 * prev0 + b2 * prev1;
        prev1 = prev0;
        prev0 = next_buffer_0;

        Vec next_buffer_0_aut = in - a1_aut * prev0_aut - a2_aut * prev1_aut;
        Vec out_aut =
          b0_aut * next_buffer_0_aut + b1_aut * prev0_aut + b2_aut * prev1_aut;
        prev1_aut = prev0_aut;
        prev0_aut = next_buffer_0_aut;

        output[i] = out + alpha * (out_aut - out);
        alpha += inc;
      }

      std::copy(&buffer(7 * Vec::size()),
                &buffer(7 * Vec::size()) + 5 * Vec::size(),
                &buffer(0));
      buffer[5] = prev0_aut;
      buffer[6] = prev1_aut;
    }
  }

  /**
   * Resets the state of the filter, as if it was processing silence from an
   * eternity.
   */
  void Reset()
  {
    buffer[5] = 0.0;
    buffer[6] = 0.0;
    if (isAutomating) {
      isAutomating = false;
      std::copy(&buffer(7 * Vec::size()),
                &buffer(7 * Vec::size()) + 5 * Vec::size(),
                &buffer(0));
    }
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
    if (isAutomating) {
      for (int i = 0; i < 5; ++i) {
        buffer[i] = buffer[7 + i];
      }
    }
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
    if (frequency[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
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
    if (gain[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
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
    if (quality[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
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
    if (filterType[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
    filterType[channel] = value;
    if (update) {
      Setup(channel);
    }
  }

  /**
   * Sets the cutoff frequency on all the channels to the specified value.
   * @param value the new cutoff frequency
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetFrequency(double value, bool update = true)
  {
    for (int i = 0; i < Vec::size(); ++i) {
      isSetupNeeded[i] = frequency[i] != value;
    }
    std::fill(frequency.begin(), frequency.end(), value);
    if (update) {
      MakeReady();
    }
  }

  /**
   * Sets the gain on all the channels to the specified value.
   * @param value the new gain
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetGain(double value, bool update = true)
  {
    for (int i = 0; i < Vec::size(); ++i) {
      isSetupNeeded[i] = gain[i] != value;
    }
    std::fill(gain.begin(), gain.end(), value);
    if (update) {
      MakeReady();
    }
  }

  /**
   * Sets the quality on all the channels to the specified value.
   * @param value the new quality
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetQuality(double value, bool update = true)
  {
    for (int i = 0; i < Vec::size(); ++i) {
      isSetupNeeded[i] = quality[i] != value;
    }
    std::fill(quality.begin(), quality.end(), value);
    if (update) {
      MakeReady();
    }
  }

  /**
   * Sets the filter type on all the channels to the specified value.
   * @param value the new filter type
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetBiquadFilterType(BiquadFilterType value, bool update = true)
  {
    for (int i = 0; i < Vec::size(); ++i) {
      isSetupNeeded[i] = filterType[i] != value;
    }
    std::fill(filterType.begin(), filterType.end(), value);
    if (update) {
      MakeReady();
    }
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
   * @param reset if true, resets the state of filter, calling Reset(). false by
   * default.
   * @param automate if true, as it is by default, the filter will use the next
   * process block call to smooth between its current state and the new state.
   */
  void Setup(int channel, bool reset = false, bool automate = true) override
  {
    isSetupNeeded[channel] = false;

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

    if (automate) {
      buffer[7][channel] = a1;
      buffer[8][channel] = a2;
      buffer[9][channel] = b0;
      buffer[10][channel] = b1;
      buffer[11][channel] = b2;
      isAutomating = true;
    }
    else {
      buffer[0][channel] = a1;
      buffer[1][channel] = a2;
      buffer[2][channel] = b0;
      buffer[3][channel] = b1;
      buffer[4][channel] = b2;
    }

    if (reset) {
      Reset(channel);
    }
  }

  /**
   * Computes the filter coefficients.
   * @param reset if true resets the state of filter, calling Reset(). false by
   * default.
   */
  void Setup(bool reset = false)
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

  /**
   * Computes any the coefficients necessary for the computation that was not
   * already computed. Usefull if you call setters with
   */
  void MakeReady()
  {
    for (int i = 0; i < Vec::size(); ++i) {
      if (isSetupNeeded[i]) {
        Setup(i);
      }
    }
  }

private:
  std::vector<int> isSetupNeeded;
  bool isAutomating = false;
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
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Scalar>::VEC2_AVAILABLE;

  int numChannels;

  std::vector<VecBiquadFilter<Vec8>> filters8;
  std::vector<VecBiquadFilter<Vec4>> filters4;
  std::vector<VecBiquadFilter<Vec2>> filters2;

  template<class Action, typename ValueType>
  ValueType OnChannel(Action action, int channel)
  {
    if constexpr (VEC8_AVAILABLE) {
      if (filters4.size() > 0) {
        if (channel < 4) {
          return action(static_cast<VecBiquadFilterInterface*>(&filters4[0]),
                        channel);
        }
        else {
          auto d8 = std::div(channel - 4, 8);
          return action(
            static_cast<VecBiquadFilterInterface*>(&filters8[d8.quot]), d8.rem);
        }
      }
      else {
        auto d8 = std::div(channel, 8);
        return action(
          static_cast<VecBiquadFilterInterface*>(&filters8[d8.quot]), d8.rem);
      }
    }
    else if constexpr (VEC4_AVAILABLE) {
      if constexpr (VEC2_AVAILABLE) {
        if (filters2.size() > 0) {
          if (channel < 2) {
            return action(static_cast<VecBiquadFilterInterface*>(&filters2[0]),
                          channel;
          }
          else {
            auto d4 = std::div(channel - 2, 4);
            return action(
              static_cast<VecBiquadFilterInterface*>(&filters4[d4.quot]),
              d4.rem);
          }
        }
        else {
          auto d4 = std::div(channel, 4);
          return action(
            static_cast<VecBiquadFilterInterface*>(&filters4[d4.quot]), d4.rem);
        }
      }
      else {
        auto d4 = std::div(channel, 4);
        return action(
          static_cast<VecBiquadFilterInterface*>(&filters4[d4.quot]), d4.rem);
      }
    }
    else {
      auto d2 = std::div(channel, 2);
      return action(static_cast<VecBiquadFilterInterface*>(&filters2[d2.quot]),
                    d2.rem);
    }
  }

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
    int num2, num4, num8;
    GetNumOfVecBuffersUsedByInterleavedBuffer<Scalar>(
      numChannels, num2, num4, num8);
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

    int channelsCount = numChannelsToProcess;

    output.SetNumSamples(numSamples);

    if constexpr (VEC2_AVAILABLE) {
      int lastBuffers2 = output.GetNumBuffers2() - 1;
      for (int i = 0; i < filters2.size(); ++i) {
        filters2[i].ProcessBlock(
          input.GetBuffer2(i), output.GetBuffer2(i), numSamples);
        channelsCount -= 2;
        if (channelsCount <= 0) {
          break;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      for (int i = 0; i < filters4.size(); ++i) {
        filters4[i].ProcessBlock(
          input.GetBuffer4(i), output.GetBuffer4(i), numSamples);
        channelsCount -= 4;
        if (channelsCount <= 0) {
          break;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      for (int i = 0; i < filters8.size(); ++i) {
        filters8[i].ProcessBlock(
          input.GetBuffer8(i), output.GetBuffer8(i), numSamples);
        channelsCount -= 8;
        if (channelsCount <= 0) {
          break;
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
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetFrequency(int channel, double value, bool update = true)
  {
    OnChannel(
      [value, update](auto* filter, int channel) {
        filter->SetFrequency(channel, value, update);
        return 0.0;
      },
      channel);
  }

  /**
   * Sets the gain on a specific channel
   * @param channel the channel on which to change the gain
   * @param value the new gain
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetGain(int channel, double value, bool update = true)
  {
    OnChannel(
      [value, update](auto* filter, int channel) {
        filter->SetGain(channel, value, update);
        return 0.0;
      },
      channel);
  }

  /**
   * Sets the quality on a specific channel
   * @param channel the channel on which to change the quality
   * @param value the new quality
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetQuality(int channel, double value, bool update = true)
  {
    OnChannel(
      [value, update](auto* filter, int channel) {
        filter->SetQuality(channel, value, update);
        return 0.0;
      },
      channel);
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
                           bool update = true)
  {
    OnChannel(
      [value, update](auto* filter, int channel) {
        filter->SetBiquadFilterType(channel, value, update);
        return 0.0;
      },
      channel);
  }

  /**
   * Sets the cutoff frequency on all the channels to the specified value.
   * @param value the new cutoff frequency
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetFrequency(double value, bool update = true)
  {
    for (auto& f : filters8) {
      f.SetFrequency(value, update);
    }
    for (auto& f : filters4) {
      f.SetFrequency(value, update);
    }
    for (auto& f : filters2) {
      f.SetFrequency(value, update);
    }
  }

  /**
   * Sets the gain on all the channels to the specified value.
   * @param value the new gain
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetGain(double value, bool update = true)
  {
    for (auto& f : filters8) {
      f.SetGain(value, update);
    }
    for (auto& f : filters4) {
      f.SetGain(value, update);
    }
    for (auto& f : filters2) {
      f.SetGain(value, update);
    }
  }

  /**
   * Sets the quality on all the channels to the specified value.
   * @param value the new quality
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetQuality(double value, bool update = true)
  {
    for (auto& f : filters8) {
      f.SetQuality(value, update);
    }
    for (auto& f : filters4) {
      f.SetQuality(value, update);
    }
    for (auto& f : filters2) {
      f.SetQuality(value, update);
    }
  }

  /**
   * Sets the filter type on all the channels to the specified value.
   * @param value the new filter type
   * @param update if true, as it is by default, the new filter coefficients
   * will be computed. Otherwise they will not be computed.
   */
  void SetBiquadFilterType(BiquadFilterType value, bool update = true)
  {
    for (auto& f : filters8) {
      f.SetBiquadFilterType(value, update);
    }
    for (auto& f : filters4) {
      f.SetBiquadFilterType(value, update);
    }
    for (auto& f : filters2) {
      f.SetBiquadFilterType(value, update);
    }
  }

  /**
   * Gets the cutoff frequency used for the specified channel.
   * @param channel the channel to get the cutoff frequency from
   * @return the cutoff frequency of the specified channel.
   */
  double GetFrequency(int channel) const
  {
    OnChannel(
      [](auto* filter, int channel) { return filter->GetFrequency(channel); },
      channel);
  }

  /**
   * Gets the gain used for the specified channel.
   * @param channel the channel to get the gain from
   * @return the gain of the specified channel.
   */
  double GetGain(int channel) const
  {
    OnChannel(
      [](auto* filter, int channel) { return filter->GetGain(channel); },
      channel);
  }

  /**
   * Gets the quality used for the specified channel.
   * @param channel the channel to get the quality from
   * @return the quality of the specified channel.
   */
  double GetQuality(int channel) const
  {
    OnChannel(
      [](auto* filter, int channel) { return filter->GetQuality(channel); },
      channel);
  }

  /**
   * Gets the filter type used for the specified channel.
   * @param channel the channel to get the filter type from
   * @return the filter type of the specified channel.
   */
  BiquadFilterType GetBiquadFilterType(int channel) const
  {
    OnChannel([](auto* filter,
                 int channel) { return filter->GetBiquadFilterType(channel); },
              channel);
  }

  /**
   * @return the maximum number of channel that the generator can work with.
   */
  int GetNumChannels() const { return numChannels; }

  /**
   * Computes any the coefficients necessary for the computation that was not
   * already computed. Usefull if you call setters with
   */
  void MakeReady()
  {
    for (auto& f : filters8) {
      f.MakeReady();
    }
    for (auto& f : filters4) {
      f.MakeReady();
    }
    for (auto& f : filters2) {
      f.MakeReady();
    }
  }
};

} // namespace avec