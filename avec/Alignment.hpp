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
#include "BoostAlign.hpp"
#include <cassert>
#include <climits>
#include <cmath>
#include <vector>

#define AVEC_ASSERT_ALIGNMENT(ptr, Vec)                                        \
  assert(boost::alignment::is_aligned(                                         \
    ptr, Vec::size() * sizeof(typename ScalarTypes<Vec>::Scalar)));

#define AVEC_ASSUME_ALIGNMENT(ptr, Vec)                                        \
  BOOST_ALIGN_ASSUME_ALIGNED(                                                  \
    ptr, Vec::size() * sizeof(typename ScalarTypes<Vec>::Scalar));

namespace avec {

constexpr int ALIGNMENT = 64; // cache line

/**
 * std::vector aligned to the width of a cache line using
 * boost::alignment::aligned_allocator.
 * @tparam T type of elements held by the std::vector
 */
template<class T>
using aligned_vector =
  std::vector<T, boost::alignment::aligned_allocator<T, ALIGNMENT>>;

/**
 * Deleter for unique_ptr holding memory allocated using
 * boost::alignment::aligned_alloc
 */
struct AlignedDeleter
{
  void operator()(void* ptr) { boost::alignment::aligned_free(ptr); }
};

/**
 * Alias for unique_ptr using AlignedDeleter
 */
template<class T>
using aligned_ptr = std::unique_ptr<T, AlignedDeleter>;

/**
 *Template class that provides static methods to construct aligned unique_ptr or
 *aligned_vector of the class specified as its template argument.
 */
template<class Class>
class Aligned final
{
public:
  template<class HolderClass = Class>
  static aligned_ptr<HolderClass> make()
  {
    auto ptr = boost::alignment::aligned_alloc(ALIGNMENT, sizeof(Class));
    return aligned_ptr<HolderClass>(new (ptr) Class);
  }

  static aligned_vector<Class> make(int num)
  {
    return aligned_vector<Class>(num);
  }
};

} // namespace avec