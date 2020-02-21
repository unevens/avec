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

#include "avec/xorshift32_16bit_simd.h"
#include <immintrin.h>

#define PXORSHIFT3216(x)                                                       \
  __m128i t = _mm_sll_epi16(x, _mm_set1_epi64x(5));                            \
  t = _mm_xor_si128(x, t);                                                     \
  t = _mm_shuffle_epi32(t, _MM_SHUFFLE(3, 2, 3, 2));                           \
  x = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 1, 0));                           \
  __m128i y = _mm_xor_si128(x, _mm_srl_epi16(x, _mm_set1_epi64x(1)));          \
  y =                                                                          \
    _mm_xor_si128(y, _mm_xor_si128(t, _mm_srl_epi16(t, _mm_set1_epi64x(3))));  \
  /*store state*/                                                              \
  __m128 xf = _mm_castsi128_ps(x);                                             \
  __m128 yf = _mm_castsi128_ps(y);                                             \
  xf = _mm_shuffle_ps(yf, xf, _MM_SHUFFLE(3, 2, 1, 0));                        \
  x = _mm_castps_si128(xf);

#define PXORSHIFT3216I(x, xs)                                                  \
  {                                                                            \
    PXORSHIFT3216(x)                                                           \
    /*convert to int*/                                                         \
    *xs = _mm_cvtepu16_epi32(x);                                               \
  }

#define PXORSHIFT3216F(x, xs)                                                  \
  {                                                                            \
    PXORSHIFT3216(x)                                                           \
    /*convert to float*/                                                       \
    xf = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(x));                               \
    *xs = _mm_add_ps(_mm_mul_ps(xf, _mm_set1_ps(2.0f / 65535.f)),              \
                     _mm_set1_ps(-1.f));                                       \
  }

void
xorshift32_16bit_simd_f4(uint16_t* state, float* output, int numSamples)
{
  __m128i x = _mm_load_si128((__m128i const*)state);
  __m128* xs = (__m128*)output;
  for (int i = 0; i < numSamples; ++i) {
    PXORSHIFT3216F(x, xs);
    ++xs;
  }
  _mm_store_si128((__m128i*)state, x);
}

void
xorshift32_16bit_simd_f8(uint16_t* state, float* output, int numSamples)
{
  __m128i x0 = _mm_load_si128((__m128i const*)state);
  __m128i x1 = _mm_load_si128((__m128i const*)(state + 8));
  __m128* xs = (__m128*)output;
#if 0
  for (int i = 0; i < numSamples; ++i) {
    PXORSHIFT3216F(x0, xs);
    ++xs;
    PXORSHIFT3216F(x1, xs);
    ++xs;
  }
#else
  for (int i = 0; i < numSamples; ++i) {
    PXORSHIFT3216F(x0, xs);
    xs += 2;
  }
  xs = (__m128*)output;
  ++xs;
  for (int i = 0; i < numSamples; ++i) {
    PXORSHIFT3216F(x1, xs);
    xs += 2;
  }
#endif
  _mm_store_si128((__m128i*)state, x0);
  _mm_store_si128((__m128i*)(state + 8), x1);
}

void
xorshift32_16bit_simd_i4(uint16_t* state, int* output, int numSamples)
{
  __m128i x = _mm_load_si128((__m128i const*)state);
  __m128i* xs = (__m128i*)output;
  for (int i = 0; i < numSamples; ++i) {
    PXORSHIFT3216I(x, xs);
    ++xs;
  }
  _mm_store_si128((__m128i*)state, x);
}

// reference
// https://b2d-f9r.blogspot.com/2010/08/16-bit-xorshift-rng-now-with-more.html
static inline uint16_t
rnd_xorshift_32()
{
  static uint16_t x = 1, y = 1;
  uint16_t t = (x ^ (x << 5));
  x = y;
  return y = (y ^ (y >> 1)) ^ (t ^ (t >> 3));
}