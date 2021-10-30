/*
Copyright 2019-2021 Dario Mambro

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

#include "avec/Buffer.hpp"
#include "avec/VecBuffer.hpp"
#include <algorithm>

namespace avec {

/**
 * A multi channel buffer holding interleaved data to be used with simd
 * vector functions from vectorclass.
 * @tparam Float float or double
 */

template<typename Float>
class InterleavedBuffer final
{
  using Vec8 = typename SimdTypes<Float>::Vec8;
  using Vec4 = typename SimdTypes<Float>::Vec4;
  using Vec2 = typename SimdTypes<Float>::Vec2;

  static constexpr bool VEC8_AVAILABLE = SimdTypes<Float>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Float>::VEC4_AVAILABLE;
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Float>::VEC2_AVAILABLE;

  std::vector<VecBuffer<Vec8>> buffers8;
  std::vector<VecBuffer<Vec4>> buffers4;
  std::vector<VecBuffer<Vec2>> buffers2;

  uint32_t numChannels = 0;
  uint32_t capacity = 0;
  uint32_t numSamples = 0;

public:
  /**
   * @return the i-th VecBuffer of 8 channel, by reference
   */
  VecBuffer<Vec8>& getBuffer8(uint32_t i) { return buffers8[i]; }

  /**
   * @return the i-th VecBuffer of 4 channel, by reference
   */
  VecBuffer<Vec4>& getBuffer4(uint32_t i) { return buffers4[i]; }

  /**
   * @return the i-th VecBuffer of 2 channel, by reference
   */
  VecBuffer<Vec2>& getBuffer2(uint32_t i) { return buffers2[i]; }

  /**
   * @return the i-th VecBuffer of 8 channel, by const reference
   */
  VecBuffer<Vec8> const& getBuffer8(uint32_t i) const { return buffers8[i]; }

  /**
   * @return the i-th VecBuffer of 4 channel, by const reference
   */
  VecBuffer<Vec4> const& getBuffer4(uint32_t i) const { return buffers4[i]; }

  /**
   * @return the i-th VecBuffer of 2 channel, by const reference
   */
  VecBuffer<Vec2> const& getBuffer2(uint32_t i) const { return buffers2[i]; }

  /**
   * @return the number of 8 channels VecBuffers
   */
  uint32_t getNumBuffers8() const { return (uint32_t)buffers8.size(); }

  /**
   * @return the number of 4 channels VecBuffers
   */
  uint32_t getNumBuffers4() const { return (uint32_t)buffers4.size(); }

  /**
   * @return the number of 2 channels VecBuffers
   */
  uint32_t getNumBuffers2() const { return (uint32_t)buffers2.size(); }

  /**
   * @return the numSamples of each VecBuffer
   */
  uint32_t getNumSamples() const { return numSamples; }

  /**
   * @return the number of channels
   */
  uint32_t getNumChannels() const { return numChannels; }

  /**
   * Sets the numSamples of each VecBuffer
   * @param value the new numSamples
   */
  void setNumSamples(uint32_t value);

  /**
   * Sets the number of channels
   * @param value the new number of channels
   */
  void setNumChannels(uint32_t value);

  /**
   * @return the allocated capacity of each VecBuffer
   */
  uint32_t getCapacity() const { return capacity; }

  /**
   * Reserves memory to store up to maxNumSamples samples on each channel
   * @param maxNumSamples the number of samples to allocate memory for
   */
  void reserve(uint32_t maxNumSamples);

  /**
   * Constructor.
   * @param numChannels the new number of channels
   * @param numSamples the number of samples to to allocate memory with for each
   * channel
   */
  InterleavedBuffer(uint32_t numChannels = 2, uint32_t numSamples = 256)
    : numSamples(numSamples)
    , capacity(numSamples)
  {
    setNumChannels(numChannels);
  }

  /**
   * Fills each buffer with the supplied value
   * @param value value to set all the elements of the buffers to.
   */
  void fill(Float value = 0.f);

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
  bool deinterleave(Float** output,
                    uint32_t numOutputChannels,
                    uint32_t numSamples) const;

  /**
   * Deinterleaves the data to an output.
   * @param output Buffer in which to store the deinterleaved data.
   * @return true if deinterleaving was successfull, false if the number of
   * channel of the output is greater to the numChannel of the InterleavedBuffer
   */
  bool deinterleave(Buffer<Float>& output) const
  {
    return deinterleave(
      output.get(), output.getNumChannels(), output.getNumSamples());
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
  bool interleave(Float* const* input,
                  uint32_t numInputChannels,
                  uint32_t numInputSamples);

  /**
   * Interleaves input data to the VecBuffers.
   * @param input Buffer holding the data to interleave.
   * @param numInputChannels number of channels to interleave, should be less
   * or equal to the numChannel of the InterleavedBuffer and of the input
   * @return true if interleaving was successful, false if numInputChannels
   * is greater to the numChannel of the InterleavedBuffer
   */
  bool interleave(Buffer<Float> const& input, uint32_t numInputChannels)
  {
    if (numInputChannels > input.getNumChannels()) {
      return false;
    }
    return interleave(input.get(), numInputChannels, input.getNumSamples());
  }

  /**
   * Interleaves input data to the VecBuffers.
   * @param input Buffer holding the data to interleave.
   * @return true if interleaving was successful, false if numInputChannels
   * is greater to the numChannel of the InterleavedBuffer
   */
  bool interleave(Buffer<Float> const& input)
  {
    return interleave(
      input.get(), input.getNumChannels(), input.getNumSamples());
  }

  /**
   * Returns the value of a a specific sample of a specific channel of the
   * buffer. cosnt version
   * @param channel
   * @param sample
   * @return a pointer to the const value of the sample of the channel, same as
   * doing &scalarBuffer[channel][sample] on a Buffer or a Float**
   */
  Float const* at(uint32_t channel, uint32_t sample) const;

  /**
   * Returns the value of a a specific sample of a specific channel of the
   * buffer.
   * @param channel
   * @param sample
   * @return a pointer to the value of the sample of the channel, same as doing
   * &scalarBuffer[channel][sample] on a Buffer or a Float**
   */
  Float* at(uint32_t channel, uint32_t sample);

  /**
   * Copies the first numSamples of an other interleaved buffer, optionally up
   * to a specified channel.
   * @param numSamplesToCopy the number of samples to copy
   * @param numChannels the number of channels to copy. If negative, all
   * channels will be copied.
   */
  void copyFrom(InterleavedBuffer const& other,
                uint32_t numSamplesToCopy,
                uint32_t numChannels);

  void copyFrom(InterleavedBuffer const& other, uint32_t numSamplesToCopy)
  {
    copyFrom(other, numSamplesToCopy, other.getNumChannels());
  }

  void copyFrom(InterleavedBuffer const& other)
  {
    copyFrom(other, other.getNumSamples(), other.getNumChannels());
  }
};

static_assert(
  std::is_nothrow_move_constructible<InterleavedBuffer<float>>::value,
  "InterleavedBuffer should be noexcept move constructible");

static_assert(std::is_nothrow_move_assignable<InterleavedBuffer<float>>::value,
              "InterleavedBuffer should be noexcept move assignable");

/**
 * Computes the number of VecBuffer of sizes 2, 4, and 8 used by an
 * InterleavedBuffer the supplied number of channels.
 * @param numChannels the total number of channels
 * @param num2 the number of VecBuffer<Vec2> used
 * @param num4 the number of VecBuffer<Vec4> used
 * @param num8 the number of VecBuffer<Vec8> used
 */
template<typename Float>
inline void
getNumOfVecBuffersUsedByInterleavedBuffer(uint32_t numChannels,
                                          uint32_t& num2,
                                          uint32_t& num4,
                                          uint32_t& num8)
{
  constexpr bool VEC8_AVAILABLE = SimdTypes<Float>::VEC8_AVAILABLE;
  constexpr bool VEC4_AVAILABLE = SimdTypes<Float>::VEC4_AVAILABLE;
  constexpr bool VEC2_AVAILABLE = SimdTypes<Float>::VEC2_AVAILABLE;
  if constexpr (VEC8_AVAILABLE) {
    if (numChannels <= 4) {
      num4 = 1;
      num8 = num2 = 0;
    }
    else {
      auto const quot = numChannels / 8;
      auto const rem = numChannels % 8;
      num8 = (uint32_t)quot + (rem > 4 ? 1 : 0);
      num4 = (rem > 0 && rem <= 4) ? 1 : 0;
      num2 = 0;
    }
  }
  else if constexpr (VEC4_AVAILABLE) {
    auto const quot = numChannels / 4;
    auto const rem = numChannels % 4;
    num8 = 0;
    if constexpr (VEC2_AVAILABLE) {
      if (numChannels <= 2) {
        num2 = 1;
        num4 = 0;
      }
      else {
        num4 = (uint32_t)quot + (rem > 2 ? 1 : 0);
        num2 = (rem > 0 && rem <= 2) ? 1 : 0;
      }
    }
    else {
      num4 = (uint32_t)quot + (rem > 0 ? 1 : 0);
      num2 = 0;
    }
  }
  else {
    auto const quot = numChannels / 2;
    auto const rem = numChannels % 2;
    num8 = 0;
    num2 = (uint32_t)quot + (rem > 0 ? 1 : 0);
    num4 = 0;
  }
}

/**
 * Consider the at(uint32_t channel, uint32_t sample) method of the
 * InterleavedBuffer. It has to find in what VecBuffer the speficied channel is
 * stored, and what to what channel of the VecBuffer it is mapped. This class
 * provides the logic necessary to get the buffer and the relative channel,
 * abstracted from the InterleavedBuffer so that it can be used by other classes
 * that use the same memory layout of the InterleavedBuffer.
 */
template<typename Float>
struct InterleavedChannel final
{
  /**
   * Executes a functor on a specific channel of a structure layed out as the
   * InterleavedBuffer.
   * @param the channel to execute the functor on
   * @param v2 the container of the Vec2 objects
   * @param v4 the container of the Vec4 objects
   * @param v8 the container of the Vec8 objects
   * @param action the functor to execute
   */
  template<class Action, class T2, class T4, class T8>
  static auto doAtChannel(uint32_t channel,
                          T2& v2,
                          T4& v4,
                          T8& v8,
                          Action action)
  {
    constexpr bool VEC8_AVAILABLE = SimdTypes<Float>::VEC8_AVAILABLE;
    constexpr bool VEC4_AVAILABLE = SimdTypes<Float>::VEC4_AVAILABLE;
    constexpr bool VEC2_AVAILABLE = SimdTypes<Float>::VEC2_AVAILABLE;

    if constexpr (VEC8_AVAILABLE) {
      if (v4.size() > 0) {
        if (channel < 4) {
          return action(v4[0], channel, 4);
        }
        else {
          auto const channelsLeft = channel - 4;
          auto const quot = channelsLeft / 8;
          auto const rem = channelsLeft % 8;
          return action(v8[quot], rem, 8);
        }
      }
      else {
        auto const quot = channel / 8;
        auto const rem = channel % 8;
        return action(v8[quot], rem, 8);
      }
    }
    else if constexpr (VEC4_AVAILABLE) {
      if constexpr (VEC2_AVAILABLE) {
        if (v2.size() > 0) {
          if (channel < 2) {
            return action(v2[0], channel, 2);
          }
          else {
            auto const channelsLeft = channel - 2;
            auto const quot = channelsLeft / 4;
            auto const rem = channelsLeft % 4;
            return action(v4[quot], rem, 4);
          }
        }
        else {
          auto const quot = channel / 4;
          auto const rem = channel % 4;
          return action(v4[quot], rem, 4);
        }
      }
      else {
        auto const quot = channel / 4;
        auto const rem = channel % 4;
        return action(v4[quot], rem, 4);
      }
    }
    else {
      auto const quot = channel / 2;
      auto const rem = channel % 2;
      return action(v2[quot], rem, 2);
    }
  }
};

// implementation

template<typename Float>
void
InterleavedBuffer<Float>::reserve(uint32_t value)
{
  if (capacity >= value) {
    return;
  }
  capacity = value;
  for (auto& b8 : buffers8) {
    b8.reserveVec(value);
  }
  for (auto& b4 : buffers4) {
    b4.reserveVec(value);
  }
  for (auto& b2 : buffers2) {
    b2.reserveVec(value);
  }
}

template<typename Float>
inline void
InterleavedBuffer<Float>::setNumSamples(uint32_t value)
{
  numSamples = value;
  reserve(value);
  for (auto& b8 : buffers8) {
    b8.setNumSamples(value);
  }
  for (auto& b4 : buffers4) {
    b4.setNumSamples(value);
  }
  for (auto& b2 : buffers2) {
    b2.setNumSamples(value);
  }
}

template<typename Float>
void
InterleavedBuffer<Float>::setNumChannels(uint32_t value)
{
  if (numChannels == value)
    return;
  numChannels = value;
  uint32_t num2, num4, num8;
  getNumOfVecBuffersUsedByInterleavedBuffer<Float>(
    numChannels, num2, num4, num8);
  buffers8.resize(num8);
  buffers4.resize(num4);
  buffers2.resize(num2);
  reserve(capacity);
  setNumSamples(numSamples);
}

template<typename Float>
void
InterleavedBuffer<Float>::fill(Float value)
{
  for (auto& b8 : buffers8) {
    b8.fill(value);
  }
  for (auto& b4 : buffers4) {
    b4.fill(value);
  }
  for (auto& b2 : buffers2) {
    b2.fill(value);
  }
}

template<typename Float>
bool
InterleavedBuffer<Float>::deinterleave(Float** output,
                                        uint32_t numOutputChannels,
                                        uint32_t numOutputSamples) const
{
  if (numOutputChannels > numChannels || numOutputSamples > numSamples) {
    return false;
  }

  uint32_t processedChannels = 0;

  if constexpr (VEC2_AVAILABLE) {
    if (buffers2.size() > 0) {
      auto const quot = numOutputChannels / 2;
      auto const rem = numOutputChannels % 2;
      for (uint32_t b = 0;
           b < std::min(quot + (rem > 0 ? 1 : 0), (uint32_t)buffers2.size());
           ++b) {
        auto const r =
          std::min(numOutputChannels - processedChannels, (uint32_t)2);
        for (uint32_t i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (uint32_t j = 0; j < numOutputSamples; ++j) {
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
      auto const quot = numOutputChannels / 4;
      auto const rem = numOutputChannels % 4;
      for (uint32_t b = 0;
           b < std::min(quot + (rem > 0 ? 1 : 0), (uint32_t)buffers4.size());
           ++b) {
        auto const r =
          std::min((uint32_t)4, numOutputChannels - processedChannels);
        for (uint32_t i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (uint32_t j = 0; j < numOutputSamples; ++j) {
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
      auto const quot = numOutputChannels / 8;
      auto const rem = numOutputChannels % 8;
      for (uint32_t b = 0;
           b < std::min(quot + (rem > 0), (uint32_t)buffers8.size());
           ++b) {
        auto const r =
          std::min((uint32_t)8, numOutputChannels - processedChannels);
        for (uint32_t i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (uint32_t j = 0; j < numOutputSamples; ++j) {
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

template<typename Float>
bool
InterleavedBuffer<Float>::interleave(Float* const* input,
                                      uint32_t numInputChannels,
                                      uint32_t numInputSamples)
{
  assert(numInputChannels <= numChannels);
  assert(numInputSamples <= numSamples);

  if (VEC8_AVAILABLE && buffers8.size() > 0) {
    if (numInputChannels % 8 != 0) {
      fill(0.f);
    }
  }
  else if (VEC4_AVAILABLE && buffers4.size() > 0) {
    if (numInputChannels % 4 != 0) {
      fill(0.f);
    }
  }
  else {
    if (numInputChannels % 2 != 0) {
      fill(0.f);
    }
  }

  uint32_t processedChannels = 0;

  if constexpr (VEC2_AVAILABLE) {
    if (buffers2.size() > 0) {
      auto const quot = numInputChannels / 2;
      auto const rem = numInputChannels % 2;
      for (uint32_t b = 0;
           b < std::min(quot + (rem > 0 ? 1 : 0), (uint32_t)buffers2.size());
           ++b) {
        auto const r =
          std::min((uint32_t)2, numInputChannels - processedChannels);
        for (uint32_t i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (uint32_t j = 0; j < numInputSamples; ++j) {
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
      auto const quot = numInputChannels / 4;
      auto const rem = numInputChannels % 4;
      for (uint32_t b = 0;
           b < std::min(quot + (rem > 0 ? 1 : 0), (uint32_t)buffers4.size());
           ++b) {
        auto const r =
          std::min((uint32_t)4, numInputChannels - processedChannels);
        for (uint32_t i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (uint32_t j = 0; j < numInputSamples; ++j) {
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
      auto const quot = numInputChannels / 8;
      auto const rem = numInputChannels % 8;
      for (uint32_t b = 0;
           b < std::min(quot + (rem > 0), (uint32_t)buffers8.size());
           ++b) {
        auto const r =
          std::min((uint32_t)8, numInputChannels - processedChannels);
        for (uint32_t i = 0; i < r; ++i) {
          auto c = i + processedChannels;
          for (uint32_t j = 0; j < numInputSamples; ++j) {
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

template<typename Float>
Float const*
InterleavedBuffer<Float>::at(uint32_t channel, uint32_t sample) const
{
  return const_cast<Float const*>(
    const_cast<InterleavedBuffer<Float>*>(this)->at(channel, sample));
}

template<typename Float>
Float*
InterleavedBuffer<Float>::at(uint32_t channel, uint32_t sample)
{
  return InterleavedChannel<Float>::doAtChannel(
    channel,
    buffers2,
    buffers4,
    buffers8,
    [sample](auto& buffer, uint32_t channel, uint32_t numChannels) {
      return &buffer(numChannels * sample + channel);
    });
}

template<typename Float>
inline void
InterleavedBuffer<Float>::copyFrom(InterleavedBuffer const& other,
                                    uint32_t numSamplesToCopy,
                                    uint32_t numChannelsToCopy)
{
  if (numChannelsToCopy < 0) {
    numChannelsToCopy = other.getNumChannels();
  }
  assert (numChannels >= numChannelsToCopy);
  assert(numSamplesToCopy <= other.getNumSamples());
  assert(numSamplesToCopy <= getNumSamples());
  if constexpr (VEC8_AVAILABLE) {
    for (std::size_t i = 0; i < buffers8.size(); ++i) {
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
    for (std::size_t i = 0; i < buffers4.size(); ++i) {
      std::copy(&other.buffers4[i](0),
                &other.buffers4[i](0) + 4 * numSamples,
                &buffers4[i](0));
      numChannelsToCopy -= 4;
      if (numChannelsToCopy <= 0) {
        return;
      }
    }
  }
  if constexpr (VEC2_AVAILABLE) {
    for (std::size_t i = 0; i < buffers2.size(); ++i) {
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
