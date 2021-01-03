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

constexpr bool has128bitSimdRegisters = true;
constexpr bool has256bitSimdRegisters = false;
constexpr bool has512bitSimdRegisters = false;

#if (defined(__aarch64__) || defined(__arm64__))
constexpr bool supportsDoublePrecision = true;
#else
constexpr bool supportsDoublePrecision = false;
#endif

#include "NeonVec.h"

#else

#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_hyp.h"
#include "vectormath_trig.h"

#endif