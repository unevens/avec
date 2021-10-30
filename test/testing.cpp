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

#include "avec/InterleavedBuffer.hpp"

#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

// macro-paranoia macro
#ifdef _MSC_VER
#ifdef _DEBUG
#ifndef CHECK_MEMORY
#define _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC_NEW
#include <crtdbg.h>
#define CHECK_MEMORY assert(_CrtCheckMemory());
#endif
#else
#define CHECK_MEMORY /*nothing*/
#endif
#else
#define CHECK_MEMORY /*nothing*/
#endif

using namespace avec;
using namespace std;

std::string
n2s(double x)
{
  stringstream stream;
  stream << fixed << showpos << setprecision(6) << x;
  return stream.str();
}

std::string
i2s(uint32_t x)
{
  stringstream stream;
  stream << setw(4) << setfill(' ') << x;
  return stream.str();
}

void
verify(bool condition, string description)
{
  if (!condition) {
    cout << "FAILURE: " << description << "\n";
  }
}

template<typename Float>
void
testInterleavedBuffer(uint32_t numChannels, uint32_t samplesPerBlock)
{
  cout << "Testing InterleavedBuffer with " << numChannels << " channels and "
       << (typeid(Float) == typeid(float) ? "single" : "double")
       << " precision\n";
  // prepare data
  uint32_t v = 0;
  Float** inout = new Float*[numChannels];
  for (uint32_t i = 0; i < numChannels; ++i) {
    inout[i] = new Float[samplesPerBlock];
    for (uint32_t s = 0; s < samplesPerBlock; ++s) {
      inout[i][s] = (Float)v++;
    }
  }
  // interleaver test
  auto buffer = InterleavedBuffer<Float>(numChannels, samplesPerBlock);
  buffer.interleave(inout, numChannels, samplesPerBlock);
  for (uint32_t i = 0; i < numChannels; ++i) {
    for (uint32_t s = 0; s < samplesPerBlock; ++s) {
      // cout << inout[i][s] << "==" << buffer.at(i, s) << "\n";
      verify(inout[i][s] == buffer.at(i, s)[0],
             "checking InterleaverBuffer::at\n");
    }
  }
  cout << "interleaving test complete\n";
  CHECK_MEMORY;
  for (uint32_t i = 0; i < numChannels; ++i) {
    for (uint32_t s = 0; s < samplesPerBlock; ++s) {
      inout[i][s] = -1.f;
    }
  }
  buffer.deinterleave(inout, numChannels, samplesPerBlock);
  // check
  v = 0;
  for (uint32_t i = 0; i < numChannels; ++i) {
    for (uint32_t s = 0; s < samplesPerBlock; ++s) {
      // cout << inout[i][s] << "==" << s << "\n";
      verify(inout[i][s] == (Float)v++, "checking deinterleaving\n");
    }
  }
  cout << "deinterleaving test completed\n";
  CHECK_MEMORY;
  // cleanup
  for (uint32_t i = 0; i < numChannels; ++i) {
    delete[] inout[i];
  }
  delete[] inout;
  cout << "completed testing InterleavedBuffer with " << numChannels
       << " channels and "
       << (typeid(Float) == typeid(float) ? "single" : "double")
       << " precision\n\n";
}

int
main()
{
  cout << "are 256 bit simd registers available? " << (has256bitSimdRegisters ? "yes" : "no") << "\n";
  cout << "are 64 bit floating point simd operations supported? " << (supportsDoublePrecision? "yes" : "no") << "\n";
  cout << "sizeof(void*) " << sizeof(void*) << "\n";

  for (uint32_t c = 1; c < 32; ++c) {
    testInterleavedBuffer<float>(c, 128);
    testInterleavedBuffer<double>(c, 128);
  }
  return 0;
}
