/*
16 bit xorshift32 rng,
see https://b2d-f9r.blogspot.com/2010/08/16-bit-xorshift-rng-now-with-more.html

Intended usage: white noise generation in audio application

This implementation uses SIMD instructions to compute 4 independent random
numbers at the same time, and store them into a buffer of interleaved channels,
in order to generate 4 channels of white noise, each with its own seed.

Author: Dario Mambro @ https://github.com/unevens/xorshift32_16bit_simd
*/

/*
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
For more information, please refer to <http://unlicense.org/>
*/

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Generates 4 interleaved channels of random floating point numbers in the
   * range [-1,1].
   * @param output must be aligned to a 16 byte boundary, it must hold 4 *
   * numSamples * sizeof(float) bytes.
   * @param state pointer to the state/seed of the rng, it musts contain 8
   * uint16_t values, it must be aligned to a 16 byte boundary. See the comment
   * about seeding.
   * @param numSamples the number of samples to generate for each channel
   */

  void xorshift32_16bit_simd_f4(uint16_t* state, float* output, int numSamples);

  /**
   * Generates 8 interleaved channels of random floating point numbers in the
   * range [-1,1].
   * @param output must be aligned to a 16 byte boundary, it must hold 8 *
   * numSamples * sizeof(float) bytes.
   * @param state pointer to the state/seed of the rng, it musts contain 16
   * uint16_t values, it must be aligned to a 16 byte boundary. See the comment
   * about seeding.
   * @param numSamples the number of samples to generate for each channel
   */
  void xorshift32_16bit_simd_f8(uint16_t* state, float* output, int numSamples);

  /**
   * Generates 4 interleaved channels of random unsigned 16 bit integers.
   * @param output it must be aligned to a 16 byte boundary, it must hold 4 *
   * numSamples * sizeof(int) bytes.
   * @param state pointer to the state/seed of the rng, it musts contain 8
   * uint16_t values, it must be aligned to a 16 byte boundary.See the comment
   * about seeding.
   * @param numSamples the number of samples to generate for each channel
   */
  void xorshift32_16bit_simd_i4(uint16_t* state, int* output, int numSamples);

  /**
   * SEEDING:
   * To generate 4 channels at the same time, we need 4 seeds.
   *
   * The state should be initialized with the seeds, repeated twice.
   *
   * For example, allocate a state with:
   *
   * uint16_t* state = alligned_malloc(8*sizeof(uint15_t));
   *
   * Set the seeds with:
   *
   * state[0] = state[4] = first_channel_seed;
   * state[1] = state[5] = second_channel_seed;
   * state[2] = state[6] = third_channel_seed;
   * state[3] = state[7] = fourth_channel_seed;
   *
   * And use it with either xorshift32_16bit_simd_f4(state, ...) or
   * xorshift32_16bit_simd_i4(state, ...)
   *
   * To generate 8 channels, we need 8 seeds, so we allocate with:
   *
   * uint16_t* state = alligned_malloc(16 * sizeof(uint15_t));
   *
   * Set the seeds with:
   *
   * state[0] = state[8] = first_channel_seed;
   * state[1] = state[9] = second_channel_seed;
   * state[2] = state[10] = third_channel_seed;
   * state[3] = state[11] = fourth_channel_seed;
   * state[4] = state[12] = fift_channel_seed;
   * state[5] = state[13] = sixth_channel_seed;
   * state[6] = state[14] = seventh_channel_seed;
   * state[7] = state[15] = eighth_channel_seed;
   *
   * And use it with xorshift32_16bit_simd_f8(state, ...)
   */

#ifdef __cplusplus
}
#endif