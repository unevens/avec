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
#include "avec/VecView.hpp"

namespace avec {

/**
 * A wrapper on aligned_vector used to manage interleaved memory that can be
 * mapped to simd vector objects.
 * @tparam Vec the simd vector object type that the VecBuffer can be mapped
 * with.
 */
template<class Vec>
class VecBuffer final
{
public:
  /**
   * The scalar type deduced from the simd vector type.
   */
  using Scalar = typename ScalarTypes<Vec>::Scalar;

private:
  aligned_vector<Scalar> data;

public:
  /**
   * Constructor
   * @param numSamples the number of samples to initialize the buffer with
   * @param value value to initialize the memory to
   */
  VecBuffer(uint32_t numSamples = 0, Scalar value = 0.f)
  {
    setNumSamples(numSamples);
    fill(value);
  }

  /**
   * @return the size of the buffer measured in number of Scalar elements
   */
  inline uint32_t getScalarSize() const { return (uint32_t)data.size(); }

  /**
   * @return the size of the buffer measured in number of Vec elements
   */
  inline uint32_t getNumSamples() const { return (uint32_t)data.size() / size<Vec>(); }

  /**
   * @return the capacity of the buffer measured in number of Scalar elements
   */
  inline uint32_t getScalarCapacity() const { return data.capacity(); }

  /**
   * @return the capacity of the buffer measured in number of Vec elements
   */
  inline uint32_t getVecCapacity() const
  {
    return getScalarCapacity() / size<Vec>();
  }

  /**
   * Resize the buffer
   * @param newSize the new size measured in number of Scalar elements
   */
  void setScalarSize(uint32_t newSize) { data.resize(newSize); }

  /**
   * Resize the buffer
   * @param newSize the new size measured in number of Vec elements
   */
  void setNumSamples(uint32_t newSize) { setScalarSize(newSize * size<Vec>()); }

  /**
   * Set the capacity of the buffer
   * @param newCapacity the new capacity measured in number of Scalar elements
   */
  void reserveScalar(uint32_t newCapacity) { data.reserve(newCapacity); }

  /**
   * Resize the buffer
   * @param newCapacity the new size measured in number of Vec elements
   */
  void reserveVec(uint32_t newCapacity)
  {
    reserveScalar(newCapacity * size<Vec>());
  }

  /**
   * Fills the buffer with the supplied value
   * @param value value to set all the elements of the buffer to.
   */
  void fill(Scalar value = 0.f) { std::fill(data.begin(), data.end(), value); }

  /**
   * @return a reference to the i-th Scalar elements of the buffer.
   */
  Scalar& operator()(uint32_t i = 0) { return data[i]; }

  /**
   * @return a reference to the i-th Scalar elements of the buffer.
   */
  Scalar const& operator()(uint32_t i = 0) const { return data[i]; }

  /**
   * @return a VecView to the memory corresponding to the i-th vecotr elements
   * of the buffer.
   */
  VecView<Vec> operator[](uint32_t i)
  {
    assert(i < data.size() / size<Vec>());
    return VecView<Vec>(&data[i * size<Vec>()]);
  }

  /**
   * @return a VecView to the memory corresponding to the i-th vecotr elements
   * of the buffer.
   */
  VecView<Vec> const operator[](uint32_t i) const
  {
    assert(i < data.size() / size<Vec>());
    return VecView<Vec>(const_cast<Scalar*>(&data[i * size<Vec>()]));
  }

  /**
   * Implicit conversion to Scalar*
   * @return a pointer to the buffer's memory.
   */
  operator Scalar*() { return &data[0]; }

  /**
   * Implicit conversion to Scalar*
   * @return a pointer to the buffer's memory.
   */
  operator Scalar const *() const { return &data[0]; }
};

// static asserts for paranoid me

static_assert(std::is_nothrow_move_constructible<VecBuffer<Vec8f>>::value,
              "VecBuffer<Vec8f> should be noexcept move constructible");

static_assert(std::is_nothrow_move_assignable<VecBuffer<Vec8f>>::value,
              "VecBuffer<Vec8f> should be noexcept move assignable");

static_assert(std::is_nothrow_move_constructible<VecBuffer<Vec4f>>::value,
              "VecBuffer<Vec4f> should be noexcept move constructible");

static_assert(std::is_nothrow_move_assignable<VecBuffer<Vec4f>>::value,
              "VecBuffer<Vec4f> should be noexcept move assignable");

} // namespace avec