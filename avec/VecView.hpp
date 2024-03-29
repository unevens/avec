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
#include "avec/Traits.hpp"

namespace avec {

/**
 * A view over a simd sized and aligned piece of a vector/array, with operators
 * overloaded to convert it and/or use it as a vector type from vectorclass.
 * @tparam Vec the vector type.
 */

template<class Vec>
class VecView final
{
  template<class TVec>
  friend class VecView;

public:
  /**
   * The scalar type.
   */
  using Float = typename ScalarTypes<Vec>::Float;

private:
  Float* ptr;

public:
  /**
   * Constructor.
   * @param ptr pointer to the memory to view. It must be aligned
   * tosize<Vec>()
   * * sizeof(Float).
   */
  VecView(Float* ptr) { setPointer(ptr); }

  /**
   * Resets the view to point to a different address.
   * @param ptr_ pointer to the memory to view. It must be aligned to
   *size<Vec>()*sizeof(Float).
   */
  void setPointer(Float* ptr_)
  {
    AVEC_ASSERT_ALIGNMENT(ptr_, Vec);
    ptr = ptr_;
    AVEC_ASSUME_ALIGNMENT(ptr, Vec);
  }

  /**
   * Copies the memory viewed by a VecView<Vec4f>. Can only be called if the
   * VecView's Float is also float.
   * @param other the view to copy from.
   */
  VecView& operator=(VecView<Vec4f> const other)
  {
    if constexpr (!std::is_same<Float, float>::value) {
      // Can't assign a VecView<Vec*f> to a VecView<Vec*d> or viceversa.
      assert(false);
      return *this;
    }
    std::copy(other.ptr, other.ptr + Vec4f::size(), ptr);
    return *this;
  }

  /**
   * Copies the memory viewed by a VecView<Vec8f>. Can only be called if the
   * VecView's Float is also float.
   * @param other the view to copy from.
   */
  VecView& operator=(VecView<Vec8f> const x)
  {
    if constexpr (!std::is_same<Float, float>::value) {
      // Can't assign a VecView<Vec*f> to a VecView<Vec*d> or viceversa.
      assert(false);
      return *this;
    }
    std::copy(x.ptr, x.ptr + size<Vec>(), ptr);
    return *this;
  }

  /**
   * Copies the memory viewed by a VecView<Vec2d>. Can only be called if the
   * VecView's Float is also double.
   * @param other the view to copy from.
   */
  VecView& operator=(VecView<Vec2d> const other)
  {
    if constexpr (!std::is_same<Float, double>::value) {
      // Can't assign a VecView<Vec*f> to a VecView<Vec*d> or viceversa.
      assert(false);
      return *this;
    }
    std::copy(other.ptr, other.ptr + size<Vec>(), ptr);
    return *this;
  }

  /**
   * Copies the memory viewed by a VecView<Vec4d>. Can only be called if the
   * VecView's Float is also double.
   * @param other the view to copy from.
   */
  VecView& operator=(VecView<Vec4d> const other)
  {
    if constexpr (!std::is_same<Float, double>::value) {
      // Can't assign a VecView<Vec*f> to a VecView<Vec*d> or viceversa.
      assert(false);
      return *this;
    }
    std::copy(other.ptr, other.ptr + Vec4d::size(), ptr);
    return *this;
  }

  /**
   * Copies the memory viewed by a VecView<Vec8d>. Can only be called if the
   * VecView's Float is also double.
   * @param other the view to copy from.
   */
  VecView& operator=(VecView<Vec8d> const other)
  {
    if constexpr (!std::is_same<Float, double>::value) {
      // Can't assign a VecView<Vec*f> to a VecView<Vec*d> or viceversa.
      assert(false);
      return *this;
    }
    std::copy(other.ptr, other.ptr + size<Vec>(), ptr);
    return *this;
  }

  /**
   * Set all elements of the viewed memory to a Float value.
   * @param value the value to set the elements to.
   */
  VecView& operator=(Float value)
  {
    for (uint32_t i = 0; i < size<Vec>(); ++i) {
      ptr[i] = value;
    }
    return *this;
  }

  /**
   * Copies the memory pointed to by the argument to the viewed memory.
   * @param src pointer to the memory to copy from.
   */
  VecView& operator=(Float const* src)
  {
    std::copy(src, src + size<Vec>(), ptr);
    return *this;
  }

  /**
   * Copies from the a vectorclass simd vector object.
   * @param v the simd vector object.
   */
  VecView& operator=(Vec const& v)
  {
    v.store_a(ptr);
    return *this;
  }

  /**
   * Implicit conversion to Float*
   * @returns the pointer to the viewed memory.
   */
  operator Float*() { return ptr; }

  /**
   * Implicit conversion to Float const*
   * @returns the pointer to the viewed memory.
   */
  operator Float const *() const { return ptr; }

  /**
   * Implicit conversion to a simd vector object.
   * @returns simd vector object initialized with the viewed memory.
   */
  operator Vec() const
  {
    Vec v;
    v.load_a(ptr);
    return v;
  }

  /**
   * Explicit Conversion to Float*
   * @returns the pointer to the viewed memory.
   */
  Float* getPtr() { return ptr; }

  /**
   * Explicit Conversion to Float*
   * @returns the pointer to the viewed memory.
   */
  Float const* getPtr() const { return ptr; }

  /**
   * A nullptr with VecView type.
   * @returns a VecView VecView pointing to nullptr.
   */
  static VecView null() { return VecView(nullptr); }
};

template<class Vec>
inline Vec
operator-(VecView<Vec> const lhs, VecView<Vec> const rhs)
{
  Vec x;
  x.load_a(lhs);
  return x - rhs;
}

} // namespace avec
