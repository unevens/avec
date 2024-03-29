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

#include "avec/Simd.hpp"
#include <type_traits>

namespace avec {

/**
 * Static template class with aliases for the available vectorclass types for a
 * given Float type (float or double).
 * @tparam Float the scalar type.
 */
template<typename Float>
struct SimdTypes
{
  static_assert(std::is_same<Float, float>::value ||
                  std::is_same<Float, double>::value,
                "Only floating point types are allowed here.");
  /**
   * 8 elements vectorclass type.
   */
  using Vec8 = typename std::
    conditional<std::is_same<Float, float>::value, Vec8f, Vec8d>::type;
  /**
   * 4 elements vectorclass type.
   */
  using Vec4 = typename std::
    conditional<std::is_same<Float, float>::value, Vec4f, Vec4d>::type;
  /**
   * 2 elements vectorclass type.
   */
  using Vec2 = typename std::
    conditional<std::is_same<Float, float>::value, Vec4f, Vec2d>::type;
  /**
   * bool constexpr, true if 8 elements vector are not emulated.
   */
  static constexpr bool VEC8_AVAILABLE = std::is_same<Float, float>::value
                                           ? has256bitSimdRegisters
                                           : has512bitSimdRegisters;
  /**
   * bool constexpr, true if 4 elements vector are not emulated.
   */
  static constexpr bool VEC4_AVAILABLE =
    std::is_same<Float, float>::value ? true : has256bitSimdRegisters;
  /**
   * bool constexpr, true if 2 elements vector are available.
   */
  static constexpr bool VEC2_AVAILABLE =
    std::is_same<Float, double>::value ? true : false;
};

/**
 * Static template class with an alias to deduce the underlying Float type from
 * a vectorclass type.
 * @tparam Vec the simd vector type.
 */
template<typename Vec>
class ScalarTypes
{
private:
  using ErrorType = bool; // any type different from float or double is ok

  using Scalar8f = typename std::
    conditional<std::is_same<Vec, Vec8f>::value, float, ErrorType>::type;
  using Scalar4f = typename std::
    conditional<std::is_same<Vec, Vec4f>::value, float, ErrorType>::type;
  using Scalar8d = typename std::
    conditional<std::is_same<Vec, Vec8d>::value, double, ErrorType>::type;
  using Scalar4d = typename std::
    conditional<std::is_same<Vec, Vec4d>::value, double, ErrorType>::type;
  using Scalar2d = typename std::
    conditional<std::is_same<Vec, Vec2d>::value, double, ErrorType>::type;

  using MaybeMaybeDouble = typename std::
    conditional<std::is_same<Scalar8d, double>::value, double, Scalar4d>::type;
  using MaybeDouble =
    typename std::conditional<std::is_same<MaybeMaybeDouble, double>::value,
                              double,
                              Scalar2d>::type;
  using MaybeFloat = typename std::
    conditional<std::is_same<Scalar8f, float>::value, float, Scalar4f>::type;

public:
  /**
   * The scalar type deduced from Vec.
   */
  using Float =
    typename std::conditional<std::is_same<MaybeDouble, double>::value,
                              double,
                              MaybeFloat>::type;

  static_assert(std::is_same<Float, float>::value ||
                  std::is_same<Float, double>::value,
                "Only Vec8f Vec4f Vec8d Vec4d and Vec2d are allowed here.");
};

/**
 * Static template class with an alias to deduce the mask type from
 * vectorclass type.
 * @tparam Vec the simd vector type.
 */
template<typename Vec>
class MaskTypes
{
public:
  /**
   * The Mask type deduced from Vec.
   */
  using Mask = typename std::conditional<
    std::is_same<Vec, Vec8f>::value,
    Vec8fb,
    typename std::conditional<
      std::is_same<Vec, Vec4f>::value,
      Vec4fb,
      typename std::conditional<
        std::is_same<Vec, Vec8d>::value,
        Vec8db,
        typename std::conditional<
          std::is_same<Vec, Vec4d>::value,
          Vec4db,
          typename std::conditional<std::is_same<Vec, Vec2d>::value,
                                    Vec2db,
                                    bool>::type>::type>::type>::type>::type;

  static_assert(!std::is_same<Mask, bool>::value,
                "Only Vec8f Vec4f Vec8d Vec4d and Vec2d are allowed here.");
};

} // namespace avec