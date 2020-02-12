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

#include "avec/Alignment.hpp"

namespace avec {

// just a wrapper around std::vector<aligned_vector<Scalar>> to manage a multi
// channel buffer

/**
 * Multi channel buffer, holding an aligned_vector<Scalar> for each
 * channel
 * @tparam Scalar the sample type, float or double.
 */
template<class Scalar>
class ScalarBuffer final
{
  std::vector<aligned_vector<Scalar>> data;
  std::vector<Scalar*> pointers;
  int size;
  int capacity;

  void UpdatePointers()
  {
    pointers.resize(data.size());
    for (int i = 0; i < pointers.size(); ++i) {
      pointers[i] = data[i].size() > 0 ? &data[i][0] : nullptr;
    }
  }

public:
  /**
   * Gets a reference to an element of the buffer.
   * @param i the index of the element to retrieve
   * @return a reference to the i-th element of buffer.
   */
  aligned_vector<Scalar>& operator[](int i) { return data[i]; }
  /**
   * Gets a const reference to an element of the buffer.
   * @param i the index of the element to retrieve
   * @return a const reference to the i-th element of buffer.
   */
  aligned_vector<Scalar> const& operator[](int i) const { return data[i]; }

  /**
   * @return a Scalar** to the buffer.
   */
  Scalar** Get() { return &pointers[0]; }

  /**
   * @return a Scalar* const* to the buffer.
   */
  Scalar* const* Get() const { return &pointers[0]; }

  /**
   * Fills the buffer with the supplied value
   * @param value value to set all the elements of the buffer to.
   */
  void Fill(Scalar value)
  {
    for (auto& channel : data) {
      std::fill(channel.begin(), channel.end(), value);
    }
  }

  /**
   * @return the size of each channel of the buffer.
   */
  int GetSize() const { return size; }

  /**
   * @return the capacity of each channel of the buffer - capacity = size of
   * allocated memory, measured in sizeof(Scalar).
   */
  int GetCapacity() const { return capacity; }

  /**
   * @return the number of channels of the buffer.
   */
  int GetNumChannels() const { return data.size(); }

  /**
   * Sets the number of channels of the buffer.
   * @param numRequiredChannels the new number of channel.
   */
  void SetNumChannels(int numRequiredChannels)
  {
    data.resize(numRequiredChannels);
    for (int i = 0; i < data.size(); ++i) {
      data[i].reserve(data[0].capacity());
      data[i].resize(data[0].size());
    }
    UpdatePointers();
  }

  /**
   * Preallocates memory for each channel of the buffer.
   * @param requiredSize the amount of samples to allocate memory for.
   */
  void Reserve(int requiredSize)
  {
    if (capacity >= requiredSize) {
      return;
    }
    capacity = requiredSize;
    for (int i = 0; i < data.size(); ++i) {
      data[i].reserve(requiredSize);
    }
    UpdatePointers();
  }

  /**
   * Set the size of each channel of the buffer.
   * @param requiredSize the amount of samples to set the size to.
   * @param shrink if true, the buffer asks each std::vector to release any
   * previously allocated memory that is no longer neeeded.
   */
  void SetSize(int requiredSize, bool shrink = false)
  {
    Reserve(requiredSize);
    size = requiredSize;
    for (int i = 0; i < data.size(); ++i) {
      data[i].resize(requiredSize, 0.0);
    }
    if (shrink) {
      Shrink();
    }
    UpdatePointers();
  }

  /**
   * Set the size of each channel of the buffer, only if there already is enough
   * memory allocated to do so.
   * @param requiredSize the amount of samples to set the size to.
   * @return true if the resizing was succesfull, false if there was not enough
   * allocated memory
   * @see GetCapacity
   * @see Reserve
   */
  bool SetSizeIfPreallocated(int requiredSize)
  {
    if (requiredSize <= GetCapacity()) {
      SetSize(requiredSize, false);
      return true;
    }
    return false;
  }

  /**
   * Set the number of channels and the size of each channel.
   * @param numRequiredChannels the new number of channels.
   * @param requiredSize the amount of samples to set the size of each channel
   * to.
   * @param shrink if true, the buffer asks each std::vector to release any
   * previously allocated memory that is no longer neeeded.
   */
  void SetNumChannelsAndSize(int numRequiredChannels,
                             int requiredSize,
                             bool shrink = false)
  {
    SetNumChannels(numRequiredChannels);
    SetSize(requiredSize, shrink);
  }

  /**
   * Asks each std::vector to release any previously allocated memory that is no
   * longer neeeded.
   */
  void Shrink()
  {
    data.shrink_to_fit();
    for (int i = 0; i < data.size(); ++i) {
      data[i].shrink_to_fit();
    }
    UpdatePointers();
    pointers.shrink_to_fit();
    capacity = size;
  }

  /**
   * Constructor.
   * @param numChannels the number of channels to allocate.
   * @param size the amount of samples to set the size of each channel
   * to.
   */
  ScalarBuffer(int numChannels = 2, int size = 256)
  {
    SetNumChannelsAndSize(numChannels, size);
  }
};

template<typename InScalar, typename OutScalar>
inline void
CopyScalarBuffer(ScalarBuffer<InScalar> const& input,
                 ScalarBuffer<OutScalar>& output,
                 int numChannels = -1)
{
  if (numChannels < 0) {
    numChannels = input.GetNumChannels();
  }
  output.SetNumChannelsAndSize(numChannels, input.GetSize());
  for (int c = 0; c < numChannels; ++c) {
    std::copy(&input[c][0], &input[c][0] + input.GetSize(), &output[c][0]);
  }
}

static_assert(std::is_nothrow_move_constructible<ScalarBuffer<float>>::value,
              "Buffer should be noexcept move constructible");

static_assert(std::is_nothrow_move_assignable<ScalarBuffer<float>>::value,
              "Buffer should be noexcept move assignable");

static_assert(std::is_copy_constructible<ScalarBuffer<float>>::value,
              "Buffer should be move constructible");

static_assert(std::is_copy_assignable<ScalarBuffer<float>>::value,
              "Buffer should be move assignable");

} // namespace avec
