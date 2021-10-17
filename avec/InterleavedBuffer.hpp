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
  std::vector<VecBuffer<Vec2>> buffers2;

  int numChannels = 0;
  int capacity = 0;
  int numSamples = 0;

public:
  /**
   * @return the i-th VecBuffer of 8 channel, by reference
   */
  VecBuffer<Vec8>& getBuffer8(int i) { return buffers8[i]; }

  /**
   * @return the i-th VecBuffer of 4 channel, by reference
   */
  VecBuffer<Vec4>& getBuffer4(int i) { return buffers4[i]; }

  /**
   * @return the i-th VecBuffer of 2 channel, by reference
   */
  VecBuffer<Vec2>& getBuffer2(int i) { return buffers2[i]; }

  /**
   * @return the i-th VecBuffer of 8 channel, by const reference
   */
  VecBuffer<Vec8> const& getBuffer8(int i) const { return buffers8[i]; }

  /**
   * @return the i-th VecBuffer of 4 channel, by const reference
   */
  VecBuffer<Vec4> const& getBuffer4(int i) const { return buffers4[i]; }

  /**
   * @return the i-th VecBuffer of 2 channel, by const reference
   */
  VecBuffer<Vec2> const& getBuffer2(int i) const { return buffers2[i]; }

  /**
   * @return the number of 8 channels VecBuffers
   */
  int getNumBuffers8() const { return (int)buffers8.size(); }

  /**
   * @return the number of 4 channels VecBuffers
   */
  int getNumBuffers4() const { return (int)buffers4.size(); }

  /**
   * @return the number of 2 channels VecBuffers
   */
  int getNumBuffers2() const { return (int)buffers2.size(); }

  /**
   * @return the numSamples of each VecBuffer
   */
  int getNumSamples() const { return numSamples; }

  /**
   * @return the number of channels
   */
  int getNumChannels() const { return numChannels; }

  /**
   * Sets the numSamples of each VecBuffer
   * @param value the new numSamples
   */
  void setNumSamples(int value);

  /**
   * Sets the number of channels
   * @param value the new number of channels
   */
  void setNumChannels(int value);

  /**
   * @return the allocated capacity of each VecBuffer
   */
  int getCapacity() const { return capacity; }

  /**
   * Reserves memory to store up to maxNumSamples samples on each channel
   * @param maxNumSamples the number of samples to allocate memory for
   */
  void reserve(int maxNumSamples);

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
    setNumChannels(numChannels);
  }

  /**
   * Fills each buffer with the supplied value
   * @param value value to set all the elements of the buffers to.
   */
  void fill(Scalar value = 0.f);

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
  bool deinterleave(Scalar** output,
                    int numOutputChannels,
                    int numSamples) const;

  /**
   * Deinterleaves the data to an output.
   * @param output ScalarBuffer in which to store the deinterleaved data.
   * @return true if deinterleaving was successfull, false if the number of
   * channel of the output is greater to the numChannel of the InterleavedBuffer
   */
  bool deinterleave(ScalarBuffer<Scalar>& output) const
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
  bool interleave(Scalar* const* input,
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
  bool interleave(ScalarBuffer<Scalar> const& input, int numInputChannels)
  {
    if (numInputChannels > input.getNumChannels()) {
      return false;
    }
    return interleave(input.get(), numInputChannels, input.getNumSamples());
  }

  /**
   * Returns the value of a a specific sample of a specific channel of the
   * buffer. cosnt version
   * @param channel
   * @param sample
   * @return a pointer to the const value of the sample of the channel, same as
   * doing &scalarBuffer[channel][sample] on a ScalarBuffer or a Scalar**
   */
  Scalar const* at(int channel, int sample) const;

  /**
   * Returns the value of a a specific sample of a specific channel of the
   * buffer.
   * @param channel
   * @param sample
   * @return a pointer to the value of the sample of the channel, same as doing
   * &scalarBuffer[channel][sample] on a ScalarBuffer or a Scalar**
   */
  Scalar* at(int channel, int sample);

  /**
   * Copies the first numSamples of an other interleaved buffer, optionally up
   * to a specified channel.
   * @param numSamplesToCopy the number of samples to copy
   * @param numChannels the number of channels to copy. If negative, all
   * channels will be copied.
   */
  void copyFrom(InterleavedBuffer const& other,
                int numSamplesToCopy,
                int numChannels = -1);
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
template<typename Scalar>
inline void
getNumOfVecBuffersUsedByInterleavedBuffer(int numChannels,
                                          int& num2,
                                          int& num4,
                                          int& num8)
{
  constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;
  constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;
  constexpr bool VEC2_AVAILABLE = SimdTypes<Scalar>::VEC2_AVAILABLE;
  if constexpr (VEC8_AVAILABLE) {
    if (numChannels <= 4) {
      num4 = 1;
      num8 = num2 = 0;
    }
    else {
      auto d8 = std::div(numChannels, 8);
      num8 = (int)d8.quot + (d8.rem > 4 ? 1 : 0);
      num4 = (d8.rem > 0 && d8.rem <= 4) ? 1 : 0;
      num2 = 0;
    }
  }
  else if constexpr (VEC4_AVAILABLE) {
    auto d4 = std::div(numChannels, 4);
    num8 = 0;
    if constexpr (VEC2_AVAILABLE) {
      if (numChannels <= 2) {
        num2 = 1;
        num4 = 0;
      }
      else {
        num4 = (int)d4.quot + (d4.rem > 2 ? 1 : 0);
        num2 = (d4.rem > 0 && d4.rem <= 2) ? 1 : 0;
      }
    }
    else {
      num4 = (int)d4.quot + (d4.rem > 0 ? 1 : 0);
      num2 = 0;
    }
  }
  else {
    auto d2 = std::div(numChannels, 2);
    num8 = 0;
    num2 = (int)d2.quot + (d2.rem > 0 ? 1 : 0);
    num4 = 0;
  }
}

/**
 * Consider the at(int channel, int sample) method of the InterleavedBuffer.
 * It has to find in what VecBuffer the speficied channel is stored, and what
 * to what channel of the VecBuffer it is mapped. This class provides the
 * logic necessary to get the buffer and the relative channel, abstracted from
 * the InterleavedBuffer so that it can be used by other classes that use the
 * same memory layout of the InterleavedBuffer.
 */
template<typename Scalar>
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
  static auto doAtChannel(int channel, T2& v2, T4& v4, T8& v8, Action action)
  {
    constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;
    constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;
    constexpr bool VEC2_AVAILABLE = SimdTypes<Scalar>::VEC2_AVAILABLE;

    if constexpr (VEC8_AVAILABLE) {
      if (v4.size() > 0) {
        if (channel < 4) {
          return action(v4[0], channel, 4);
        }
        else {
          auto d8 = std::div(channel - 4, 8);
          return action(v8[d8.quot], d8.rem, 8);
        }
      }
      else {
        auto d8 = std::div(channel, 8);
        return action(v8[d8.quot], d8.rem, 8);
      }
    }
    else if constexpr (VEC4_AVAILABLE) {
      if constexpr (VEC2_AVAILABLE) {
        if (v2.size() > 0) {
          if (channel < 2) {
            return action(v2[0], channel, 2);
          }
          else {
            auto d4 = std::div(channel - 2, 4);
            return action(v4[d4.quot], d4.rem, 4);
          }
        }
        else {
          auto d4 = std::div(channel, 4);
          return action(v4[d4.quot], d4.rem, 4);
        }
      }
      else {
        auto d4 = std::div(channel, 4);
        return action(v4[d4.quot], d4.rem, 4);
      }
    }
    else {
      auto d2 = std::div(channel, 2);
      return action(v2[d2.quot], d2.rem, 2);
    }
  }
};

// implementation

template<typename Scalar>
void
InterleavedBuffer<Scalar>::reserve(int value)
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

template<typename Scalar>
inline void
InterleavedBuffer<Scalar>::setNumSamples(int value)
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

template<typename Scalar>
void
InterleavedBuffer<Scalar>::setNumChannels(int value)
{
  if (numChannels == value)
    return;
  numChannels = value;
  int num2, num4, num8;
  getNumOfVecBuffersUsedByInterleavedBuffer<Scalar>(
    numChannels, num2, num4, num8);
  buffers8.resize(num8);
  buffers4.resize(num4);
  buffers2.resize(num2);
  reserve(capacity);
  setNumSamples(numSamples);
}

template<typename Scalar>
void
InterleavedBuffer<Scalar>::fill(Scalar value)
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

template<typename Scalar>
bool
InterleavedBuffer<Scalar>::deinterleave(Scalar** output,
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
InterleavedBuffer<Scalar>::interleave(Scalar* const* input,
                                      int numInputChannels,
                                      int numInputSamples)
{

  if (numInputChannels > numChannels) {
    setNumChannels(numInputChannels);
  }
  setNumSamples(numInputSamples);

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
InterleavedBuffer<Scalar>::at(int channel, int sample) const
{
  return const_cast<Scalar const*>(
    const_cast<InterleavedBuffer<Scalar>*>(this)->at(channel, sample));
}

template<typename Scalar>
Scalar*
InterleavedBuffer<Scalar>::at(int channel, int sample)
{
  return InterleavedChannel<Scalar>::doAtChannel(
    channel,
    buffers2,
    buffers4,
    buffers8,
    [sample](auto& buffer, int channel, int numChannels) {
      return &buffer(numChannels * sample + channel);
    });
}

template<typename Scalar>
inline void
InterleavedBuffer<Scalar>::copyFrom(InterleavedBuffer const& other,
                                    int numSamplesToCopy,
                                    int numChannelsToCopy)
{
  if (numChannelsToCopy < 0) {
    numChannelsToCopy = other.getNumChannels();
  }
  if (numChannels < numChannelsToCopy) {
    setNumChannels(other.getNumChannels());
  }
  assert(numSamplesToCopy <= other.getNumSamples());
  setNumSamples(numSamplesToCopy);
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
  if constexpr (VEC2_AVAILABLE) {
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
