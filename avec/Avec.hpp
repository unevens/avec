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
#include "avec/InterleavedBuffer.hpp"

template<class T>
using aligned_vector = avec::aligned_vector<T>;

template<class T>
using aligned_ptr = avec::aligned_ptr<T>;

template<class T>
using Aligned = avec::Aligned<T>;

template<class Number>
using Buffer = avec::Buffer<Number>;

template<class Vec>
using VecBuffer = avec::VecBuffer<Vec>;

template<class Vec>
using VecView = avec::VecView<Vec>;

template<typename Number>
using InterleavedBuffer = avec::InterleavedBuffer<Number>;

template<typename Number>
using SimdTypes = avec::SimdTypes<Number>;

template<typename Vec>
using ScalarTypes = avec::ScalarTypes<Vec>;

template<typename Vec>
using MaskTypes = avec::MaskTypes<Vec>;

constexpr bool has256bitSimdRegisters = avec::has256bitSimdRegisters;
constexpr bool has512bitSimdRegisters = avec::has512bitSimdRegisters;
constexpr bool has128bitSimdRegisters = avec::has128bitSimdRegisters;
