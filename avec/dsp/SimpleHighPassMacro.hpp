/*
Copyright 2020 Dario Mambro

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
#include "avec/dsp/SimpleHighPass.hpp"

// simple high pass

#define LOAD_SIMPLE_HIGH_PASS(filter, Vec)                                     \
  Vec filter##_in_mem = Vec().load_a(filter->inputMemory);                     \
  Vec filter##_out_mem = Vec().load_a(filter->outputMemory);                   \
  Vec filter##_alpha = Vec().load_a(filter->alpha);

#define STORE_SIMPLE_HIGH_PASS(filter)                                         \
  filter##_in_mem.store_a(filter->inputMemory);                                \
  filter##_out_mem.store_a(filter->outputMemory);

#define APPLY_SIMPLE_HIGH_PASS(filter, in, out)                                \
  filter##_out_mem =                                                           \
    filter##_alpha * (filter##_out_mem + in - filter##_in_mem);                \
  filter##_in_mem = in;                                                        \
  out = filter##_out_mem;
