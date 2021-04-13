/*
Copyright 2019 Dario Mambro

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
i2s(int x)
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

template<typename Scalar>
void
testInterleavedBuffer(int numChannels, int samplesPerBlock)
{
  cout << "Testing InterleavedBuffer with " << numChannels << "channels and "
       << (typeid(Scalar) == typeid(float) ? "single" : "double")
       << " precision\n";
  // prepare data
  int v = 0;
  Scalar** inout = new Scalar*[numChannels];
  for (int i = 0; i < numChannels; ++i) {
    inout[i] = new Scalar[samplesPerBlock];
    for (int s = 0; s < samplesPerBlock; ++s) {
      inout[i][s] = (Scalar)v++;
    }
  }
  // interleaver test
  auto buffer = InterleavedBuffer<Scalar>(numChannels, samplesPerBlock);
  buffer.interleave(inout, numChannels, samplesPerBlock);
  for (int i = 0; i < numChannels; ++i) {
    for (int s = 0; s < samplesPerBlock; ++s) {
      // cout << inout[i][s] << "==" << buffer.at(i, s) << "\n";
      verify(inout[i][s] == buffer.at(i, s)[0],
             "checking InterleaverBuffer::at\n");
    }
  }
  cout << "interleaving test complete\n";
  CHECK_MEMORY;
  for (int i = 0; i < numChannels; ++i) {
    for (int s = 0; s < samplesPerBlock; ++s) {
      inout[i][s] = -1.f;
    }
  }
  buffer.deinterleave(inout, numChannels, samplesPerBlock);
  // check
  v = 0;
  for (int i = 0; i < numChannels; ++i) {
    for (int s = 0; s < samplesPerBlock; ++s) {
      // cout << inout[i][s] << "==" << s << "\n";
      verify(inout[i][s] == (Scalar)v++, "checking deinterleaving\n");
    }
  }
  cout << "deinterleaving test completed\n";
  CHECK_MEMORY;
  // cleanup
  for (int i = 0; i < numChannels; ++i) {
    delete[] inout[i];
  }
  delete[] inout;
  cout << "completed testing InterleavedBuffer with " << numChannels
       << " channels and "
       << (typeid(Scalar) == typeid(float) ? "single" : "double")
       << " precision\n\n";
}

int
main()
{
  cout << "are 256 bit simd registers available? " << (has256bitSimdRegisters ? "yes" : "no") << "\n";
  cout << "are 64 bit floating point simd operations supported? " << (supportsDoublePrecision? "yes" : "no") << "\n";
  cout << "sizeof(void*) " << sizeof(void*) << "\n";

  for (int c = 1; c < 32; ++c) {
    testInterleavedBuffer<float>(c, 128);
    testInterleavedBuffer<double>(c, 128);
  }
  return 0;
}
