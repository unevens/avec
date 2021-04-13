/*
Copyright 2021 Dario Mambro

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

#if (defined(__arm__) || defined(__aarch64__) || defined(__arm64__))

#define AVEC_ARM 1
#define AVEC_X86 0

namespace avec {

#ifdef __ARM_NEON

#define AVEC_NEON 1
constexpr bool has128bitSimdRegisters = true;

#else

constexpr bool has128bitSimdRegisters = false;

#endif

constexpr bool has256bitSimdRegisters = false;
constexpr bool has512bitSimdRegisters = false;

#if (defined(__aarch64__) || defined(__arm64__))

#define AVEC_NEON 1
#define AVEC_NEON_64 1
constexpr bool supportsDoublePrecision = true;

#else

#define AVEC_NEON_64 0
constexpr bool supportsDoublePrecision = false;

#endif

static_assert(has128bitSimdRegisters, "NEON not supported.");

} // namespace avec

#include "NeonMath.hpp"

#else

#define AVEC_X86 1
#define AVEC_ARM 0

#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_hyp.h"
#include "vectormath_trig.h"

namespace avec {

#define AVEC_SSE2 (INSTRSET >= 2)
#define AVEC_AVX (INSTRSET >= 7)
#define AVEC_AVX512 (INSTRSET >= 9)

constexpr bool has128bitSimdRegisters = AVEC_SSE2;
constexpr bool supportsDoublePrecision = AVEC_SSE2;
constexpr bool has256bitSimdRegisters = AVEC_AVX;
constexpr bool has512bitSimdRegisters = AVEC_AVX512;
static_assert(has128bitSimdRegisters,
              "The minimum supported instruction set is SSE2.");
} // namespace avec

#endif

#ifndef AVEC_ARM
#define AVEC_ARM 0
#endif
#ifndef AVEC_NEON
#define AVEC_NEON 0
#endif
#ifndef AVEC_NEON_64
#define AVEC_NEON_64 0
#endif
#ifndef AVEC_X86
#define AVEC_X86 0
#endif
#ifndef AVEC_SSE2
#define AVEC_SSE2 0
#endif
#ifndef AVEC_AVX
#define AVEC_AVX 0
#endif
#ifndef AVEC_AVX512
#define AVEC_AVX512 0
#endif

namespace avec {

template<class Vec>
constexpr int
size()
{
  return Vec::size();
}

template<>
constexpr int
size<float>()
{
  return 1;
}

template<>
constexpr int
size<double>()
{
  return 1;
}

} // namespace avec