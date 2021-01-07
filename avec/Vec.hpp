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

#ifdef __ARM_NEON
constexpr bool has128bitSimdRegisters = true;
#else
constexpr bool has128bitSimdRegisters = false;
#endif

constexpr bool has256bitSimdRegisters = false;
constexpr bool has512bitSimdRegisters = false;

#if (defined(__aarch64__) || defined(__arm64__))
#define AVEC_NEON_64
constexpr bool supportsDoublePrecision = true;
#else
constexpr bool supportsDoublePrecision = false;
#endif

static_assert(has128bitSimdRegisters, "NEON not supported.");

#include "NeonMath.hpp"

#else

#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_hyp.h"
#include "vectormath_trig.h"

// see vectorclass/instrset.h
constexpr bool has256bitSimdRegisters = INSTRSET >= 7;
constexpr bool has128bitSimdRegisters = INSTRSET >= 2;
constexpr bool supportsDoublePrecision = INSTRSET >= 2;
constexpr bool has512bitSimdRegisters = INSTRSET >= 9;
static_assert(has128bitSimdRegisters,
              "The minimum supported instruction set is SSE2.");

#endif