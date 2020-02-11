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
#include "avec/InterleavedBuffer.hpp"
#include "avec/xorshift32_16bit_simd.h"

namespace avec {

/**
 * Abstract template for white noise generators that generate noise into a
 * VecBuffer<Vec>.
 */
template<class Vec>
class VecNoiseGenerator final
{};

/**
 * White noise generator that generates 8 channels of white
 * noise in a VecBuffer<Vec8f>.
 */
template<>
class VecNoiseGenerator<Vec8f> final
{
  aligned_vector<uint16_t> state;

public:
  /**
   * Constructor.
   * @param seed an aligned vector containing the seeds, one for each channel.
   */
  VecNoiseGenerator(
    aligned_vector<uint16_t> const& seed = { 1, 2, 3, 4, 5, 6, 7, 8 })
  {
    assert(seed.size() == 8);
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin(), seed.begin() + 4);
    }
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin() + 4, seed.end());
    }
  }

  /**
   * Generates noise.
   * @param output the VecBuffer in which to generate the noise.
   * @param numSamples the number of samples to generate in each channel.
   */
  void Generate(VecBuffer<Vec8f>& output, int numSamples)
  {
    xorshift32_16bit_simd_f8(&state[0], &output(0), numSamples);
  }

  /**
   * Sets the state/seed of a channel.
   * @param channel the channel whose seed should be set.
   * @param value the new state.
   */
  void SetState(int channel, uint16_t value) { state[channel] = value; }

  /**
   * Gets the state/seed of a channel.
   * @param channel the channel whose state should be returned.
   * @return the state of the channel.
   */
  uint16_t GetState(int channel) const { return state[channel]; }
};

/**
 * White noise generator that generates 4 channels of white
 * noise in a VecBuffer<Vec4f>.
 */
template<>
class VecNoiseGenerator<Vec4f> final
{
  aligned_vector<uint16_t> state;

public:
  /**
   * Constructor.
   * @param seed an aligned vector containing the seeds, one for each channel.
   */
  VecNoiseGenerator(aligned_vector<uint16_t> const& seed = { 1, 2, 3, 4 })
  {
    assert(seed.size() == 4);
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin(), seed.end());
    }
  }

  /**
   * Generates noise.
   * @param output the VecBuffer in which to generate the noise.
   * @param numSamples the number of samples to generate in each channel.
   */
  void Generate(VecBuffer<Vec4f>& output, int numSamples)
  {
    xorshift32_16bit_simd_f4(&state[0], &output(0), numSamples);
  }

  /**
   * Sets the state/seed of a channel.
   * @param channel the channel whose seed should be set.
   * @param value the new state.
   */
  void SetState(int channel, uint16_t value) { state[channel] = value; }

  /**
   * Gets the state/seed of a channel.
   * @param channel the channel whose state should be returned.
   * @return the state of the channel.
   */
  uint16_t GetState(int channel) const { return state[channel]; }
};

/**
 * White noise generator that generates 8 channels of white
 * noise in a VecBuffer<Vec8d>.
 */
template<>
class VecNoiseGenerator<Vec8d> final
{
  aligned_vector<uint16_t> state;

public:
  /**
   * Constructor.
   * @param seed an aligned vector containing the seeds, one for each channel.
   */
  VecNoiseGenerator(
    aligned_vector<uint16_t> const& seed = { 1, 2, 3, 4, 5, 6, 7, 8 })
  {
    assert(seed.size() == 8);
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin(), seed.begin() + 4);
    }
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin() + 4, seed.end());
    }
  }

  /**
   * Generates noise.
   * @param output the VecBuffer in which to generate the noise.
   * @param numSamples the number of samples to generate in each channel.
   */
  void Generate(VecBuffer<Vec8d>& output, int numSamples)
  {
    float* asFloats = (float*)&output(0);
    xorshift32_16bit_simd_f8(&state[0], asFloats, numSamples);
    for (int i = numSamples - 1; i > -1; --i) {
      output(i) = asFloats[i];
    }
  }

  /**
   * Sets the state/seed of a channel.
   * @param channel the channel whose seed should be set.
   * @param value the new state.
   */
  void SetState(int channel, uint16_t value) { state[channel] = value; }

  /**
   * Gets the state/seed of a channel.
   * @param channel the channel whose state should be returned.
   * @return the state of the channel.
   */
  uint16_t GetState(int channel) const { return state[channel]; }
};

/**
 * White noise generator that generates 4 channels of white
 * noise in a VecBuffer<Vec4d>.
 */
template<>
class VecNoiseGenerator<Vec4d> final
{
  aligned_vector<uint16_t> state;

public:
  /**
   * Constructor.
   * @param seed an aligned vector containing the seeds, one for each channel.
   */
  VecNoiseGenerator(aligned_vector<uint16_t> const& seed = { 1, 2, 3, 4 })
  {
    assert(seed.size() == 4);
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin(), seed.end());
    }
  }

  /**
   * Generates noise.
   * @param output the VecBuffer in which to generate the noise.
   * @param numSamples the number of samples to generate in each channel.
   */
  void Generate(VecBuffer<Vec4d>& output, int numSamples)
  {
    float* asFloats = (float*)&output(0);
    xorshift32_16bit_simd_f4(&state[0], asFloats, numSamples);
    for (int i = numSamples - 1; i > -1; --i) {
      output(i) = asFloats[i];
    }
  }

  /**
   * Sets the state/seed of a channel.
   * @param channel the channel whose seed should be set.
   * @param value the new state.
   */
  void SetState(int channel, uint16_t value) { state[channel] = value; }

  /**
   * Gets the state/seed of a channel.
   * @param channel the channel whose state should be returned.
   * @return the state of the channel.
   */
  uint16_t GetState(int channel) const { return state[channel]; }
};

/**
 * White noise generator that generates 2 or 4 channels of white
 * noise in 1 or 2 VecBuffer<Vec2d>.
 */
template<>
class VecNoiseGenerator<Vec2d> final
{
  aligned_vector<uint16_t> state;

public:
  /**
   * Constructor.
   * @param seed an aligned vector containing the seeds, one for each channel.
   */
  VecNoiseGenerator(aligned_vector<uint16_t> const& seed = { 1, 2, 3, 4 })
  {
    assert(seed.size() == 4);
    for (int i = 0; i < 2; ++i) {
      state.insert(state.end(), seed.begin(), seed.end());
    }
  }

  /**
   * Generates noise.
   * @param output the VecBuffer in which to generate the noise.
   * @param numSamples the number of samples to generate in each channel.
   * @param output2 a pointer to an optional VecBuffer in which to generate two
   * additional channels of noise. Ignored if nullptr
   */
  void Generate(VecBuffer<Vec2d>& output,
                int numSamples,
                VecBuffer<Vec2d>* output2 = nullptr)
  {
    float* asFloats = (float*)&output(0);
    xorshift32_16bit_simd_f4(&state[0], asFloats, numSamples);
    if (output2) {
      for (int i = numSamples - 1; i > -1; i -= 4) {
        (*output2)(i) = asFloats[i];
        (*output2)(i) = asFloats[i - 1];
      }
    }
    for (int i = numSamples - 1; i > -1; i -= 4) {
      output(i) = asFloats[i - 2];
      output(i) = asFloats[i - 3];
    }
  }

  /**
   * Sets the state/seed of a channel.
   * @param channel the channel whose seed should be set.
   * @param value the new state.
   */
  void SetState(int channel, uint16_t value) { state[channel] = value; }

  /**
   * Gets the state/seed of a channel.
   * @param channel the channel whose state should be returned.
   * @return the state of the channel.
   */
  uint16_t GetState(int channel) const { return state[channel]; }
};

/**
 * White noise generator that generates a custom number of channels of white
 * noise in an interleaved buffer.
 */

template<typename Scalar>
class NoiseGenerator final
{
  using Vec8 = typename SimdTypes<Scalar>::Vec8;
  using Vec4 = typename SimdTypes<Scalar>::Vec4;
  using Vec2 = typename SimdTypes<Scalar>::Vec2;
  static constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Scalar>::VEC2_AVAILABLE;
  int numChannels;
  std::vector<VecNoiseGenerator<Vec8>> generators8;
  std::vector<VecNoiseGenerator<Vec4>> generators4;
  std::vector<VecNoiseGenerator<Vec2>> generators2;

public:
  /**
   * Constructor.
   * @param numChannels the number of channels to allocate resources for.
   * @param seed the seed for the first channel. The seed of the n-th channel
   * will be seed + n.
   */
  NoiseGenerator(int numChannels, uint16_t seed = 1);

  /**
   * Resets the state/seed of all channels.
   * @param state the seed for the first channel. The state of the n-th channel
   * will be state + n.
   */
  void SetState(uint16_t state);
  
  /**
   * Generates noise.
   * @param outputBuffer the InterleavedBuffer in which to generate the noise.
   * @param numSamples the number of samples to generate in each channel.
   * @param numChannelsToGenerate the number of channels to generate.
   */
  void Generate(InterleavedBuffer<Scalar>& outputBuffer,
                int numSamples,
                int numChannelsToGenerate);

  /**
   * @return the maximum number of channel that the generator can work with.
   */
  int GetNumChannels() const { return numChannels; }
};

// implementation

template<typename Scalar>
inline NoiseGenerator<Scalar>::NoiseGenerator(int numChannels, uint16_t seed)
  : numChannels(numChannels)
{
  int num8, num4, num2;
  if constexpr (VEC8_AVAILABLE) {
    if (numChannels <= 4) {
      num4 = 1;
      num8 = num2 = 0;
    }
    else {
      auto d8 = std::div(numChannels, 8);
      num8 = (int)d8.quot + (d8.rem > 4 ? 1 : 0);
      num4 = (d8.rem > 0 && d8.rem <= 4) ? 1 : 0;
      num2 = 0;
    }
  }
  else if constexpr (VEC4_AVAILABLE) {
    auto d4 = std::div(numChannels, 4);
    num8 = 0;
    if constexpr (VEC2_AVAILABLE) {
      if (numChannels <= 2) {
        num2 = 1;
        num4 = 0;
      }
      else {
        num4 = (int)d4.quot + (d4.rem > 2 ? 1 : 0);
        num2 = (d4.rem > 0 d4.rem <= 2) ? 1 : 0;
      }
    }
    else {
      num4 = (int)d4.quot + (d4.rem > 0 ? 1 : 0);
      num2 = 0;
    }
  }
  else {
    auto d4 = std::div(numChannels, 4);
    num8 = 0;
    num2 = (int)d4.quot + (d4.rem > 0 ? 1 : 0);
    num4 = 0;
  }
  generators8.reserve(num8);
  generators4.reserve(num4);
  generators2.reserve(num2);
  uint16_t s = seed;
  for (int i = 0; i < num8; ++i) {
    generators8.push_back(
      VecNoiseGenerator<Vec8>({ s++, s++, s++, s++, s++, s++, s++, s++ }));
  }
  for (int i = 0; i < num4; ++i) {
    generators4.push_back(VecNoiseGenerator<Vec4>({ s++, s++, s++, s++ }));
  }
  for (int i = 0; i < num2; ++i) {
    generators2.push_back(VecNoiseGenerator<Vec2>({ s++, s++, s++, s++ }));
  }
}

template<typename Scalar>
inline void
NoiseGenerator<Scalar>::SetState(uint16_t state)
{
  for (auto& gen : generators8) {
    for (int i = 0; i < 8; ++i) {
      gen.SetState(i, state++);
    }
  }
  for (auto& gen : generators4) {
    for (int i = 0; i < 4; ++i) {
      gen.SetState(i, state++);
    }
  }
  for (auto& gen : generators2) {
    for (int i = 0; i < 4; ++i) {
      gen.SetState(i, state++);
    }
  }
}

template<typename Scalar>
inline void
NoiseGenerator<Scalar>::Generate(InterleavedBuffer<Scalar>& outputBuffer,
                                 int numSamples,
                                 int numChannelsToGenerate)
{
  assert(numChannelsToGenerate <= numChannels);

  if constexpr (VEC2_AVAILABLE) {
    int lastBuffers2 = outputBuffer.GetNumBuffers2() - 1;
    for (int i = 0; i < generators2.size(); ++i) {
      VecBuffer<Vec2>* next =
        (i < lastBuffers2) ? &outputBuffer.GetBuffer2(i + 1) : nullptr;
      generators2[i].Generate(outputBuffer.GetBuffer2(i), numSamples, next);
      numChannelsToGenerate -= 4;
      if (numChannelsToGenerate <= 0) {
        return;
      }
    }
  }

  if constexpr (VEC4_AVAILABLE) {
    for (int i = 0; i < generators4.size(); ++i) {
      generators4[i].Generate(outputBuffer.GetBuffer4(i), numSamples);
      numChannelsToGenerate -= 4;
      if (numChannelsToGenerate <= 0) {
        return;
      }
    }
  }

  outputBuffer.SetNumSamples(numSamples);
  if constexpr (VEC8_AVAILABLE) {
    for (int i = 0; i < generators8.size(); ++i) {
      generators8[i].Generate(outputBuffer.GetBuffer8(i), numSamples);
      numChannelsToGenerate -= 8;
      if (numChannelsToGenerate <= 0) {
        return;
      }
    }
  }

  assert(numChannelsToGenerate <= 0);
}

} // namespace avec