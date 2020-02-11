/*
Copyright 2019-2020 Dario Mambro

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

#include "avec/ScalarBuffer.hpp"
#include "avec/VecBuffer.hpp"
#include <algorithm>

namespace avec {

/**
 * A multi channel buffer holding interleaved data to be used with simd
 * vector functions from vectorclass.
 * @tparam Scalar float or double
 */

template<typename Scalar>
class InterleavedBuffer final
{
  using Vec8 = typename SimdTypes<Scalar>::Vec8;
  using Vec4 = typename SimdTypes<Scalar>::Vec4;
  using Vec2 = typename SimdTypes<Scalar>::Vec2;

  static constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Scalar>::VEC2_AVAILABLE;

  std::vector<VecBuffer<Vec8>> buffers8;
  std::vector<VecBuffer<Vec4>> buffers4;

  // fallback when VEC4_AVAILABLE == false (Scalar=double, and no AVX)
  // otherwise unused
  std::vector<VecBuffer<Vec2>> buffers2;

  int numChannels;
  int capacity;
  int numSamples;
  bool isUsingBuffer4;

public:
  /**
   * @return the i-th VecBuffer of 8 channel, by reference
   */
  VecBuffer<Vec8>& GetBuffer8(int i) { return buffers8[i]; }

  /**
   * @return the i-th VecBuffer of 4 channel, by reference
   */
  VecBuffer<Vec4>& GetBuffer4(int i) { return buffers4[i]; }

  /**
   * @return the i-th VecBuffer of 2 channel, by reference
   */
  VecBuffer<Vec2>& GetBuffer2(int i) { return buffers2[i]; }

  /**
   * @return the i-th VecBuffer of 8 channel, by const reference
   */
  VecBuffer<Vec8> const& GetBuffer8(int i) const { return buffers8[i]; }

  /**
   * @return the i-th VecBuffer of 4 channel, by const reference
   */
  VecBuffer<Vec4> const& GetBuffer4(int i) const { return buffers4[i]; }

  /**
   * @return the i-th VecBuffer of 2 channel, by const reference
   */
  VecBuffer<Vec2> const& GetBuffer2(int i) const { return buffers2[i]; }

  /**
   * @return the number of 8 channels VecBuffers
   */
  int GetNumBuffers8() const { return (int)buffers8.size(); }

  /**
   * @return the number of 4 channels VecBuffers
   */
  int GetNumBuffers4() const { return (int)buffers4.size(); }

  /**
   * @return the number of 2 channels VecBuffers
   */
  int GetNumBuffers2() const { return (int)buffers2.size(); }

  /**
   * @return the numSamples of each VecBuffer
   */
  int GetNumSamples() const { return numSamples; }

  /**
   * @return the number of channels
   */
  int GetNumChannels() const { return numChannels; }

  /**
   * Sets the numSamples of each VecBuffer
   * @param value the new numSamples
   */
  void SetNumSamples(int value);

  /**
   * Sets the number of channels
   * @param value the new number of channels
   */
  void SetNumChannels(int value);

  /**
   * @return the allocated capacity of each VecBuffer
   */
  int GetCapacity() const { return capacity; }

  /**
   * Reserves memory to store up to maxNumSamples samples on each channel
   * @param maxNumSamples the number of samples to allocate memory for
   */
  void Reserve(int maxNumSamples);

  /**
   * Constructor.
   * @param numChannels the new number of channels
   * @param numSamples the number of samples to to allocate memory with for each
   * channel
   */
  InterleavedBuffer(int numChannels = 2, int numSamples = 256)
    : numSamples(numSamples)
    , capacity(numSamples)
  {
    SetNumChannels(numChannels);
  }

  /**
   * Fills each buffer with the supplied value
   * @param value value to set all the elements of the buffers to.
   */
  void Fill(Scalar value = 0.f);

  /**
   * Deinterleaves the data to an output.
   * @param output pointer to the memory in which to store the deinterleaved
   * data.
   * @param numOutputChannels number of channels to deinterleave, should be less
   * or equal to the numChannel of the InterleavedBuffer
   * @param numSamples number of samples of each channel of the output
   * @return true if deinterleaving was successfull, false if numOutputChannels
   * is greater to the numChannel of the InterleavedBuffer
   */
  bool Deinterleave(Scalar** output,
                    int numOutputChannels,
                    int numSamples) const;

  /**
   * Deinterleaves the data to an output.
   * @param output ScalarBuffer in which to store the deinterleaved data.
   * @return true if deinterleaving was successfull, false if the number of
   * channel of the output is greater to the numChannel of the InterleavedBuffer
   */
  bool Deinterleave(ScalarBuffer<Scalar>& output) const
  {
    return Deinterleave(
      output.Get(), output.GetNumChannels(), output.GetSize());
  }

  /**
   * Interleaves input data to the VecBuffers.
   * @param input pointer to the data to interleave.
   * @param numInputChannels number of channels to interleave, should be less
   * or equal to the numChannel of the InterleavedBuffer
   * @param numInputSamples number of samples to interleave of each channel of
   * the input
   * @return true if interleaving was successfull, false if numInputChannels
   * is greater to the numChannel of the InterleavedBuffer
   */
  bool Interleave(Scalar* const* input,
                  int numInputChannels,
                  int numInputSamples);

  /**
   * Interleaves input data to the VecBuffers.
   * @param input ScalarBuffer holding the data to interleave.
   * @param numInputChannels number of channels to interleave, should be less
   * or equal to the numChannel of the InterleavedBuffer and of the input
   * @return true if interleaving was successfull, false if numInputChannels
   * is greater to the numChannel of the InterleavedBuffer
   */
  bool Interleave(ScalarBuffer<Scalar> const& input, int numInputChannels)
  {
    if (numInputChannels > input.GetNumChannels()) {
      return false;
    }
    return Interleave(input.Get(), numInputChannels, input.GetSize());
  }

  /**
   * Returns the value of a a specific sample of a specific channel of the
   * buffer. cosnt version
   * @param channel
   * @param sample
   * @return a pointer to the const value of the sample of the channel, same as
   * doing &scalarBuffer[channel][sample] on a ScalarBuffer or a Scalar**
   */
  Scalar const* At(int channel, int sample) const;

  /**
   * Returns the value of a a specific sample of a specific channel of the
   * buffer.
   * @param channel
   * @param sample
   * @return a pointer to the value of the sample of the channel, same as doing
   * &scalarBuffer[channel][sample] on a ScalarBuffer or a Scalar**
   */
  Scalar* At(int channel, int sample);

  /**
   * Copies the first numSamples of an other interleaved buffer, optionally up
   * to a specified channel.
   * @param numChannels the number of channels to copy. If negative, all
   * channels will be copied.
   */
  void CopyFrom(InterleavedBuffer const& other,
                int numSamples,
                int numChannels = -1);
};

static_assert(
  std::is_nothrow_move_constructible<InterleavedBuffer<float>>::value,
  "InterleavedBuffer should be noexcept move constructible");

static_assert(std::is_nothrow_move_assignable<InterleavedBuffer<float>>::value,
              "InterleavedBuffer should be noexcept move assignable");

// implementation

template<typename Scalar>
void
InterleavedBuffer<Scalar>::Reserve(int value)
{
  if (capacity >= value) {
    return;
  }
  capacity = value;
  for (auto& b8 : buffers8) {
    b8.SetCapacityAsVec(value);
  }
  for (auto& b4 : buffers4) {
    b4.SetCapacityAsVec(value);
  }
  for (auto& b2 : buffers2) {
    b2.SetCapacityAsVec(value);
  }
}

template<typename Scalar>
inline void
InterleavedBuffer<Scalar>::SetNumSamples(int value)
{
  numSamples = value;
  Reserve(value);
  for (auto& b8 : buffers8) {
    b8.SetSizeAsVec(value);
  }
  for (auto& b4 : buffers4) {
    b4.SetSizeAsVec(value);
  }
  for (auto& b2 : buffers2) {
    b2.SetSizeAsVec(value);
  }
}

template<typename Scalar>
void
InterleavedBuffer<Scalar>::SetNumChannels(int value)
{
  numChannels = value;
  if constexpr (VEC8_AVAILABLE) {
    if (numChannels <= 4) {
      buffers4.resize(1);
      buffers8.resize(0);
      buffers2.resize(0);
    }
    else {
      auto d8 = std::div(numChannels, 8);
      buffers8.resize((std::size_t)d8.quot + (d8.rem > 4 ? 1 : 0));
      buffers4.resize((d8.rem <= 4 && d8.rem > 0) ? 1 : 0);
      buffers2.resize(0);
    }
  }
  else if constexpr (VEC4_AVAILABLE) {
    buffers8.resize(0);
    auto d4 = std::div(numChannels, 4);
    if constexpr (VEC2_AVAILABLE) {
      if (numChannels <= 2) {
        buffers2.resize(1);
        buffers4.resize(0);
      }
      else {
        buffers4.resize((std::size_t)d4.quot + (d4.rem > 2 ? 1 : 0));
        buffers2.resize((d4.rem <= 2 && d4.rem > 0) ? 1 : 0);
      }
    }
    else {
      buffers4.resize((std::size_t)d4.quot + (d4.rem > 0 ? 1 : 0));
      buffers2.resize(0);
    }
  }
  else {
    auto d2 = std::div(numChannels, 2);
    buffers2.resize((std::size_t)d2.quot + (d2.rem > 0 ? 1 : 0));
    buffers8.resize(0);
    buffers4.resize(0);
  }
  Reserve(capacity);
  SetNumSamples(numSamples);
}

template<typename Scalar>
void
InterleavedBuffer<Scalar>::Fill(Scalar value)
{
  for (auto& b8 : buffers8) {
    b8.Fill(value);
  }
  for (auto& b4 : buffers4) {
    b4.Fill(value);
  }
  for (auto& b2 : buffers2) {
    b2.Fill(value);
  }
}

template<typename Scalar>
bool
InterleavedBuffer<Scalar>::Deinterleave(Scalar** output,
                                        int numOutputChannels,
                                        int numOutputSamples) const
{
  if (numOutputChannels > numChannels || numOutputSamples > numSamples) {
    return false;
  }

  int processedChannels = 0;

  if constexpr (VEC2_AVAILABLE) {
    if (buffers2.size() > 0) {
      auto d2 = std::div(numOutputChannels, 2);
      for (int b = 0;
           b < std::min(d2.quot + (d2.rem > 0 ? 1 : 0), (int)buffers2.size());
           ++b) {
        int r = std::min(2, numOutputChannels - processedChannels);
        for (int i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (int j = 0; j < numOutputSamples; ++j) {
            output[c][j] = buffers2[b](j * 2 + i);
          }
        }
        processedChannels += r;
        assert(processedChannels <= numOutputChannels);
        if (processedChannels == numOutputChannels) {
          return true;
        }
      }
    }
  }
  if constexpr (VEC4_AVAILABLE) {
    if (buffers4.size() > 0) {
      auto d4 = std::div(numOutputChannels, 4);
      for (int b = 0;
           b < std::min(d4.quot + (d4.rem > 0 ? 1 : 0), (int)buffers4.size());
           ++b) {
        int r = std::min(4, numOutputChannels - processedChannels);
        for (int i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (int j = 0; j < numOutputSamples; ++j) {
            output[c][j] = buffers4[b](j * 4 + i);
          }
        }
        processedChannels += r;
        assert(processedChannels <= numOutputChannels);
        if (processedChannels == numOutputChannels) {
          return true;
        }
      }
    }
  }
  if constexpr (VEC8_AVAILABLE) {
    if (buffers8.size() > 0) {
      auto d8 = std::div(numOutputChannels, 8);
      for (int b = 0;
           b < std::min(d8.quot + (d8.rem > 0), (int)buffers8.size());
           ++b) {
        int r = std::min(8, numOutputChannels - processedChannels);
        for (int i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (int j = 0; j < numOutputSamples; ++j) {
            output[c][j] = buffers8[b](j * 8 + i);
          }
        }
        processedChannels += r;
        assert(processedChannels <= numOutputChannels);
        if (processedChannels == numOutputChannels) {
          return true;
        }
      }
    }
  }
  assert(false);
  return false;
}

template<typename Scalar>
bool
InterleavedBuffer<Scalar>::Interleave(Scalar* const* input,
                                      int numInputChannels,
                                      int numInputSamples)
{

  if (numInputChannels > numChannels) {
    SetNumChannels(numInputChannels);
  }
  SetNumSamples(numInputSamples);

  if (VEC8_AVAILABLE && buffers8.size() > 0) {
    if (numInputChannels % 8 != 0) {
      Fill(0.f);
    }
  }
  else if (VEC4_AVAILABLE && buffers4.size() > 0) {
    if (numInputChannels % 4 != 0) {
      Fill(0.f);
    }
  }
  else {
    if (numInputChannels % 2 != 0) {
      Fill(0.f);
    }
  }

  int processedChannels = 0;

  if constexpr (VEC2_AVAILABLE) {
    if (buffers2.size() > 0) {
      auto d2 = std::div(numInputChannels, 2);
      for (int b = 0;
           b < std::min(d2.quot + (d2.rem > 0 ? 1 : 0), (int)buffers2.size());
           ++b) {
        int r = std::min(2, numInputChannels - processedChannels);
        for (int i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (int j = 0; j < numInputSamples; ++j) {
            buffers2[b](j * 2 + i) = input[c][j];
          }
        }
        processedChannels += r;
        assert(processedChannels <= numInputChannels);
        if (processedChannels == numInputChannels) {
          return true;
        }
      }
    }
  }
  if constexpr (VEC4_AVAILABLE) {
    if (buffers4.size() > 0) {
      auto d4 = std::div(numInputChannels, 4);
      for (int b = 0;
           b < std::min(d4.quot + (d4.rem > 0 ? 1 : 0), (int)buffers4.size());
           ++b) {
        int r = std::min(4, numInputChannels - processedChannels);
        for (int i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (int j = 0; j < numInputSamples; ++j) {
            buffers4[b](j * 4 + i) = input[c][j];
          }
        }
        processedChannels += r;
        assert(processedChannels <= numInputChannels);
        if (processedChannels == numInputChannels) {
          return true;
        }
      }
    }
  }
  if constexpr (VEC8_AVAILABLE) {
    if (buffers8.size() > 0) {
      auto d8 = std::div(numInputChannels, 8);
      for (int b = 0;
           b < std::min(d8.quot + (d8.rem > 0), (int)buffers8.size());
           ++b) {
        int r = std::min(8, numInputChannels - processedChannels);
        for (int i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (int j = 0; j < numInputSamples; ++j) {
            buffers8[b](j * 8 + i) = input[c][j];
          }
        }
        processedChannels += r;
        assert(processedChannels <= numInputChannels);
        if (processedChannels == numInputChannels) {
          return true;
        }
      }
    }
  }
  assert(false);
  return false;
}

template<typename Scalar>
Scalar const*
InterleavedBuffer<Scalar>::At(int channel, int sample) const
{
  if constexpr (VEC8_AVAILABLE) {
    if (buffers4.size() > 0) {
      if (channel < 4) {
        return &buffers4[0](4 * sample + channel);
      }
      else {
        auto d8 = std::div(channel - 4, 8);
        return &buffers8[d8.quot](8 * sample + d8.rem);
      }
    }
    else {
      auto d8 = std::div(channel, 8);
      return &buffers8[d8.quot](8 * sample + d8.rem);
    }
  }
  else if constexpr (VEC4_AVAILABLE) {
    if constexpr (VEC2_AVAILABLE) {
      if (buffers2.size() > 0) {
        if (channel < 2) {
          return &buffers2[0](2 * sample + channel);
        }
        else {
          auto d4 = std::div(channel - 2, 4);
          return &buffers4[d4.quot](4 * sample + d4.rem);
        }
      }
      else {
        auto d4 = std::div(channel, 4);
        return &buffers4[d4.quot](4 * sample + d4.rem);
      }
    }
    else {
      auto d4 = std::div(channel, 4);
      return &buffers4[d4.quot](4 * sample + d4.rem);
    }
  }
  else {
    auto d2 = std::div(channel, 2);
    return &buffers2[d2.quot](2 * sample + d2.rem);
  }
}

template<typename Scalar>
Scalar*
InterleavedBuffer<Scalar>::At(int channel, int sample)
{
  return const_cast<Scalar*>(
    const_cast<InterleavedBuffer<Scalar> const*>(this)->At(channel, sample));
}

template<typename Scalar>
inline void
InterleavedBuffer<Scalar>::CopyFrom(InterleavedBuffer const& other,
                                    int numSamples,
                                    int numChannelsToCopy)
{
  if (numChannelsToCopy < 0) {
    numChannelsToCopy = other.GetNumChannels();
  }
  if (numChannels < numChannelsToCopy) {
    SetNumChannels(other.GetNumChannels());
  }
  assert(numSamples <= other.GetNumSamples());
  SetNumSamples(numSamples);
  if constexpr (VEC8_AVAILABLE) {
    for (int i = 0; i < buffers8.size(); ++i) {
      std::copy(&other.buffers8[i](0),
                &other.buffers8[i](0) + 8 * numSamples,
                &buffers8[i](0));
      numChannelsToCopy -= 8;
      if (numChannelsToCopy <= 0) {
        return;
      }
    }
  }
  if constexpr (VEC4_AVAILABLE) {
    for (int i = 0; i < buffers4.size(); ++i) {
      std::copy(&other.buffers4[i](0),
                &other.buffers4[i](0) + 4 * numSamples,
                &buffers4[i](0));
      numChannelsToCopy -= 4;
      if (numChannelsToCopy <= 0) {
        return;
      }
    }
  }
  else {
    for (int i = 0; i < buffers2.size(); ++i) {
      std::copy(&other.buffers2[i](0),
                &other.buffers2[i](0) + 2 * numSamples,
                &buffers2[i](0));
      numChannelsToCopy -= 2;
      if (numChannelsToCopy <= 0) {
        return;
      }
    }
  }
}

} // namespace avec