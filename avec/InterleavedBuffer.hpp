/*
Copyright 2019 Dario Mambro

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
   * @return the i-th 8 channel interleaved VecBuffer, by reference
   */
  VecBuffer<Vec8>& GetBuffer8(int i) { return buffers8[i]; }

  /**
   * @return the i-th 4 channel interleaved VecBuffer, by reference
   */
  VecBuffer<Vec4>& GetBuffer4(int i) { return buffers4[i]; }

  /**
   * @return the i-th 2 channel interleaved VecBuffer, by reference
   */
  VecBuffer<Vec2>& GetBuffer2(int i) { return buffers2[i]; }

  /**
   * @return the i-th 8 channel interleaved VecBuffer, by const reference
   */
  VecBuffer<Vec8> const& GetBuffer8(int i) const { return buffers8[i]; }

  /**
   * @return the i-th 4 channel interleaved VecBuffer, by const reference
   */
  VecBuffer<Vec4> const& GetBuffer4(int i) const { return buffers4[i]; }

  /**
   * @return the i-th 2 channel interleaved VecBuffer, by const reference
   */
  VecBuffer<Vec2> const& GetBuffer2(int i) const { return buffers2[i]; }

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
   * Deinterleaves the data to an output of numOutputChannels channels, treating
   * the InterleavedBuffer data as a sequence of numTracks parallel tracks,
   * which are summed together into the output.
   * @param output pointer to the memory in which to store the deinterleaved
   * data.
   * @param numOutputChannels number of channels to deinterleave, should be less
   * or equal to the numChannel of the InterleavedBuffer
   * @param numInputSamples number of samples of each channel of the output
   * @param numTracks number of tracks
   * @return true if deinterleaving was successfull, false if numOutputChannels
   * * numTracks is greater than the numChannel of the InterleavedBuffer
   */
  bool DeinterleaveTracks(Scalar** output,
                          int numOutputChannels,
                          int numInputSamples,
                          int numTracks) const;

  /**
   * Deinterleaves the data to an output of numOutputChannels channels, treating
   * the InterleavedBuffer data as a sequence of numTracks parallel tracks,
   * which are summed together into the output.
   * @param output ScalarBuffer in which to store the deinterleaved data.
   * @param numTracks number of tracks
   * @return true if deinterleaving was successfull, false if the number of
   * channels of output multiplied by numTracks is greater than the
   * numChannel of the InterleavedBuffer
   */
  bool DeinterleaveTracks(ScalarBuffer<Scalar>& output,
                          int numOutputChannels,
                          int numTracks) const
  {
    return DeinterleaveTracks(
      output.Get(), numOutputChannels, output.GetSize(), numTracks);
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
   * buffer.
   * @param channel
   * @param sample
   * @return the value of the sample of the channel, same as doing
   * scalarBuffer[channel][sample] on a ScalarBuffer or a Scalar**
   */
  Scalar& At(int channel, int sample);
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
    auto d8 = std::div(numChannels, 8);
    buffers8.resize((std::size_t)d8.quot + (d8.rem > 4 ? 1 : 0));
    buffers4.resize((d8.rem <= 4) ? 1 : 0);
  }
  else if constexpr (VEC4_AVAILABLE) {
    auto d4 = std::div(numChannels, 4);
    buffers8.resize(0);
    buffers4.resize((std::size_t)d4.quot + (d4.rem > 0 ? 1 : 0));
  }
  else {
    auto d2 = std::div(numChannels, 2);
    buffers2.resize((std::size_t)d2.quot + (d2.rem > 0 ? 1 : 0));
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
  int b = 0;
  for (int c = 0; c < numOutputChannels; ++c) {
    if (VEC8_AVAILABLE && buffers8.size() > 0) {
      auto d = std::div(b, 8);
      assert(d.quot <= buffers8.size());
      if (d.quot == buffers8.size()) {
        assert(buffers4.size() > 0);
        for (int i = 0; i < numOutputSamples; ++i) {
          output[c][i] = buffers4[0](4 * i + d.rem);
        }
      }
      else {
        for (int i = 0; i < numOutputSamples; ++i) {
          output[c][i] = buffers8[d.quot](8 * i + d.rem);
        }
      }
    }
    else if (VEC4_AVAILABLE && buffers4.size()) {
      auto d = std::div(b, 4);
      assert(d.quot < buffers4.size());
      for (int i = 0; i < numOutputSamples; ++i) {
        output[c][i] = buffers4[d.quot](4 * i + d.rem);
      }
    }
    else {
      auto d = std::div(b, 2);
      assert(d.quot < buffers2.size());
      for (int i = 0; i < numOutputSamples; ++i) {
        output[c][i] = buffers2[d.quot](2 * i + d.rem);
      }
    }
    ++b;
  }
  return true;
}

template<typename Scalar>
bool
InterleavedBuffer<Scalar>::DeinterleaveTracks(Scalar** output,
                                              int numOutputChannels,
                                              int numOutputSamples,
                                              int numTracks) const
{
  if (numOutputChannels * numTracks > numChannels ||
      numOutputSamples > numSamples) {
    return false;
  }
  for (int i = 0; i < numOutputChannels; ++i) {
    std::fill_n(output[i], numOutputSamples, 0.f);
  }
  int b = 0;
  for (int n = 0; n < numTracks; ++n) {
    for (int c = 0; c < numOutputChannels; ++c) {
      if (VEC8_AVAILABLE && buffers8.size() > 0) {
        auto d = std::div(b, 8);
        assert(d.quot <= buffers8.size());
        if (d.quot == buffers8.size()) {
          assert(buffers4.size() > 0);
          for (int i = 0; i < numOutputSamples; ++i) {
            output[c][i] += buffers4[0](4 * i + d.rem);
          }
        }
        else {
          for (int i = 0; i < numOutputSamples; ++i) {
            output[c][i] += buffers8[d.quot](8 * i + d.rem);
          }
        }
      }
      else if (VEC4_AVAILABLE && buffers4.size() > 0) {
        auto d = std::div(b, 4);
        assert(d.quot < buffers4.size());
        for (int i = 0; i < numOutputSamples; ++i) {
          output[c][i] += buffers4[d.quot](4 * i + d.rem);
        }
      }
      else {
        auto d = std::div(b, 2);
        assert(d.quot < buffers2.size());
        for (int i = 0; i < numOutputSamples; ++i) {
          output[c][i] += buffers2[d.quot](2 * i + d.rem);
        }
      }
      ++b;
    }
  }
  return true;
}

template<typename Scalar>
bool
InterleavedBuffer<Scalar>::Interleave(Scalar* const* input,
                                      int numInputChannels,
                                      int numInputSamples)
{
  SetNumChannels(numInputChannels);
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
  if constexpr (VEC8_AVAILABLE) {
    auto d8 = std::div(numInputChannels, 8);
    for (int b = 0; b < std::min(d8.quot + (d8.rem > 4), (int)buffers8.size());
         ++b) {
      int r = std::min(8, numInputChannels - processedChannels);
      for (int i = 0; i < r; ++i) {
        auto c = b * 8 + i;
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

  if constexpr (VEC4_AVAILABLE) {
    auto d4 = std::div(numInputChannels, 4);
    for (int b = 0;
         b < std::min(d4.quot + (d4.rem > 0 ? 1 : 0), (int)buffers4.size());
         ++b) {
      int r = std::min(4, numInputChannels - processedChannels);
      for (int i = 0; i < r; ++i) {
        auto c = b * 4 + i;
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

  auto d2 = std::div(numInputChannels, 2);
  for (int b = 0;
       b < std::min(d2.quot + (d2.rem > 0 ? 1 : 0), (int)buffers2.size());
       ++b) {
    int r = std::min(2, numInputChannels - processedChannels);
    for (int i = 0; i < r; ++i) {
      auto c = b * 2 + i;
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

  assert(false);
  return false;
}

template<typename Scalar>
Scalar&
InterleavedBuffer<Scalar>::At(int channel, int sample)
{
  if (VEC8_AVAILABLE && buffers8.size() > 0) {
    auto d8 = std::div(channel, 8);
    if (d8.quot < buffers8.size()) {
      return buffers8[d8.quot](8 * sample + d8.rem);
    }
    else {
      return buffers4[0](4 * sample + d8.rem);
    }
  }
  else if constexpr (VEC4_AVAILABLE) {
    auto d4 = std::div(channel, 4);
    return buffers4[d4.quot](4 * sample + d4.rem);
  }
  else {
    auto d2 = std::div(channel, 2);
    return buffers2[d2.quot](2 * sample + d2.rem);
  }
}

} // namespace avec