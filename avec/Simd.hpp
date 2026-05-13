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
#include <cstdint>

#if (defined(__arm__) || defined(__aarch64__) || defined(__arm64__))

#define AVEC_ARM 1
#define AVEC_X86 0

namespace avec {

#ifdef __ARM_NEON

#define AVEC_NEON 1
constexpr bool has128bitSimdRegisters = true;

#else

#define AVEC_NEON 0
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

constexpr bool hasSimd = AVEC_NEON;

} // namespace avec

#include "NeonMath.hpp"

#else

#define AVEC_X86 1
#define AVEC_ARM 0

// Put vectorclass into its own namespace so its free functions (notably
// `extend` and friends, declared at global scope when MAX_VECTOR_SIZE >= 256)
// don't collide with system headers. On macOS in particular, any TU that
// pulls in <Cocoa/Cocoa.h> via iPlug2's IControls.h transitively gets
// <MacTypes.h>, whose anonymous enum declares a global enumerator
// `extend = 0x40`. With VCL_NAMESPACE set, vectorclass's `extend` lives at
// `vcl::extend` and the collision goes away. Consumers of avec who reference
// vectorclass types unqualified (audio-dsp, Curvessor's CurvessorDsp.cpp,
// Overdraw, …) keep working via the using-declarations below.
#ifndef VCL_NAMESPACE
#define VCL_NAMESPACE vcl
#endif
#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_hyp.h"
#include "vectormath_trig.h"

// Re-export the vectorclass identifiers that avec and its consumers
// reference without a `vcl::` qualifier. Operator overloads and free
// functions found via ADL (e.g. `exp(Vec2d)`) don't need re-exporting —
// argument-dependent lookup finds them in `vcl::` automatically. Template
// functions invoked with explicit template arguments (e.g. `permute2<1,0>(v)`)
// do need to be visible by name, so they go below.
using vcl::Vec2d;
using vcl::Vec4d;
using vcl::Vec8d;
using vcl::Vec4f;
using vcl::Vec8f;
using vcl::Vec16f;
using vcl::Vec2db;
using vcl::Vec4db;
using vcl::Vec8db;
using vcl::Vec4fb;
using vcl::Vec8fb;
using vcl::Vec16fb;
using vcl::permute2;
using vcl::permute4;
using vcl::permute8;
using vcl::select;

namespace avec {

#define AVEC_SSE (INSTRSET >= 1)
#define AVEC_SSE2 (INSTRSET >= 2)
#define AVEC_AVX (INSTRSET >= 7)
#define AVEC_AVX512 (INSTRSET >= 9)

constexpr bool has128bitSimdRegisters = AVEC_SSE2;
constexpr bool supportsDoublePrecision = AVEC_SSE2;
constexpr bool has256bitSimdRegisters = AVEC_AVX;
constexpr bool has512bitSimdRegisters = AVEC_AVX512;
constexpr bool hasSimd = AVEC_SSE;

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
constexpr uint32_t
size()
{
  return Vec::size();
}

template<>
constexpr uint32_t
size<float>()
{
  return 1;
}

template<>
constexpr uint32_t
size<double>()
{
  return 1;
}

} // namespace avec
