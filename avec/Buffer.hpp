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

#include "avec/Alignment.hpp"

namespace avec {

// just a wrapper around std::vector<aligned_vector<Float>> to manage a multi
// channel buffer

/**
 * Multi channel buffer, holding an aligned_vector<Float> for each
 * channel
 * @tparam Float the sample type, float or double.
 */
template<class Float>
class Buffer final
{
  std::vector<aligned_vector<Float>> data;
  std::vector<Float*> pointers;
  uint32_t size = 0;
  uint32_t capacity = 0;

  void updatePointers()
  {
    pointers.resize(data.size());
    for (uint32_t i = 0; i < pointers.size(); ++i) {
      pointers[i] = data[i].size() > 0 ? &data[i][0] : nullptr;
    }
  }

public:
  /**
   * Gets a reference to an element of the buffer.
   * @param i the index of the element to retrieve
   * @return a reference to the i-th element of buffer.
   */
  aligned_vector<Float>& operator[](uint32_t i) { return data[i]; }
  /**
   * Gets a const reference to an element of the buffer.
   * @param i the index of the element to retrieve
   * @return a const reference to the i-th element of buffer.
   */
  aligned_vector<Float> const& operator[](uint32_t i) const { return data[i]; }

  /**
   * @return a Float** to the buffer.
   */
  Float** get() { return &pointers[0]; }

  /**
   * @return a Float* const* to the buffer.
   */
  Float* const* get() const { return &pointers[0]; }

  /**
   * Fills the buffer with the supplied value
   * @param value value to set all the elements of the buffer to.
   */
  void fill(Float value)
  {
    for (auto& channel : data) {
      std::fill(channel.begin(), channel.end(), value);
    }
  }

  /**
   * @return the size of each channel of the buffer.
   */
  uint32_t getNumSamples() const { return size; }

  /**
   * @return the capacity of each channel of the buffer - capacity = size of
   * allocated memory, measured in sizeof(Float).
   */
  uint32_t getCapacity() const { return capacity; }

  /**
   * @return the number of channels of the buffer.
   */
  uint32_t getNumChannels() const { return static_cast<uint32_t>(data.size()); }

  /**
   * Sets the number of channels of the buffer.
   * @param numRequiredChannels the new number of channel.
   */
  void setNumChannels(uint32_t numRequiredChannels)
  {
    if (numRequiredChannels == data.size())
      return;
    data.resize(numRequiredChannels);
    for (auto& d : data) {
      d.reserve(capacity);
      d.resize(data[0].size());
    }
    updatePointers();
  }

  /**
   * Preallocates memory for each channel of the buffer.
   * @param numSamples the amount of samples to allocate memory for.
   */
  void reserve(uint32_t numSamples)
  {
    if (capacity >= numSamples) {
      return;
    }
    capacity = numSamples;
    for (uint32_t i = 0; i < data.size(); ++i) {
      data[i].reserve(numSamples);
    }
    updatePointers();
  }

  /**
   * Set the size of each channel of the buffer.
   * @param requiredSize the amount of samples to set the size to.
   * @param shrinkIfSmaller if true, the buffer tells each std::vector to
   * release any previously allocated memory that is no longer neeeded.
   */
  void setNumSamples(uint32_t numSamples, bool shrinkIfSmaller = false)
  {
    if (numSamples == size && !shrinkIfSmaller)
      return;
    reserve(numSamples);
    size = numSamples;
    capacity = std::max(capacity, size);
    for (auto& d : data) {
      d.resize(numSamples, 0.0);
    }
    if (shrinkIfSmaller) {
      shrink();
    }
    updatePointers();
  }

  /**
   * Set the number of channels and the size of each channel.
   * @param numRequiredChannels the new number of channels.
   * @param numRequiredSamples the amount of samples to set the size of each
   * channel to.
   * @param shrink if true, the buffer tells each std::vector to release any
   * previously allocated memory that is no longer neeeded.
   */
  void setNumChannelsAndSamples(uint32_t numRequiredChannels,
                                uint32_t numRequiredSamples,
                                bool shrink = false)
  {
    setNumChannels(numRequiredChannels);
    setNumSamples(numRequiredSamples, shrink);
  }

  /**
   * Asks each std::vector to release any previously allocated memory that is no
   * longer neeeded.
   */
  void shrink()
  {
    data.shrink_to_fit();
    for (auto& d : data) {
      d.shrink_to_fit();
    }
    updatePointers();
    pointers.shrink_to_fit();
    capacity = size;
  }

  /**
   * Constructor.
   * @param numChannels the number of channels to allocate.
   * @param size the amount of samples to set the size of each channel
   * to.
   */
  Buffer(uint32_t numChannels = 2, uint32_t size = 256)
  {
    capacity = size;
    setNumChannelsAndSamples(numChannels, size);
  }
};

template<typename InScalar, typename OutScalar>
inline void
copyScalarBuffer(Buffer<InScalar> const& input,
                 Buffer<OutScalar>& output,
                 uint32_t numChannels)
{
  if (numChannels < 0) {
    numChannels = input.getNumChannels();
  }
  output.setNumChannelsAndSamples(numChannels, input.getNumSamples());
  for (uint32_t c = 0; c < numChannels; ++c) {
    std::copy(
      &input[c][0], &input[c][0] + input.getNumSamples(), &output[c][0]);
  }
}

template<typename InScalar, typename OutScalar>
inline void
copyScalarBuffer(Buffer<InScalar> const& input, Buffer<OutScalar>& output)
{
  auto const numChannels = input.getNumChannels();
  output.setNumChannelsAndSamples(numChannels, input.getNumSamples());
  for (uint32_t c = 0; c < numChannels; ++c) {
    std::copy(
      &input[c][0], &input[c][0] + input.getNumSamples(), &output[c][0]);
  }
}

static_assert(std::is_nothrow_move_constructible<Buffer<float>>::value,
              "Buffer should be noexcept move constructable");

static_assert(std::is_nothrow_move_assignable<Buffer<float>>::value,
              "Buffer should be noexcept move assignable");

static_assert(std::is_copy_constructible<Buffer<float>>::value,
              "Buffer should be move constructable");

static_assert(std::is_copy_assignable<Buffer<float>>::value,
              "Buffer should be move assignable");

} // namespace avec
