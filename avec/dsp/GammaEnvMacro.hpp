/**
 * Simd implementation of Aleksey Vaneev's https://github.com/avaneev/gammaenv
 *
 * Differences from the original:
 * - output is in dB,
 * - does peak/rms computation
 * - IsIverse is not supported
 * - Attack and Release are not in seconds but in angular frequency units
 * (2*pi/(samplerate*seconds))
 *
 * By Dario Mambro @ https://github.com/unevens/avec
 */

/**
 * Copyright (c) 2016 Aleksey Vaneev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once
#include "GammaEnv.hpp"

#define LOAD_GAMMAENV_STATE(gamma, Vec)                                        \
  Vec gamma##_env[16];                                                         \
  Vec gamma##_enva[4];                                                         \
  Vec gamma##_envb[4];                                                         \
  Vec gamma##_envr[16];                                                        \
  Vec gamma##_env5 = Vec().load_a(gamma->env5_);                               \
  Vec gamma##_enva5 = Vec().load_a(gamma->enva5_);                             \
  Vec gamma##_envb5 = Vec().load_a(gamma->envb5_);                             \
  Vec gamma##_envr5 = Vec().load_a(gamma->envr5_);                             \
  Vec gamma##_prevr = Vec().load_a(gamma->prevr_);                             \
  auto const gamma##_rms = Vec().load_a(gamma->useRms) != 0.0;                 \
  Vec const gamma##_to_db_coef = select(gamma##_rms,                           \
                                        10.0 / 2.30258509299404568402,         \
                                        20.0 / 2.30258509299404568402);        \
  for (int i = 0; i < 4; ++i) {                                                \
    gamma##_enva[i] = Vec().load_a(gamma->enva_ + i * Vec::size());            \
    gamma##_envb[i] = Vec().load_a(gamma->envb_ + i * Vec::size());            \
  }                                                                            \
  for (int i = 0; i < 16; ++i) {                                               \
    gamma##_env[i] = Vec().load_a(gamma->env_ + i * Vec::size());              \
    gamma##_envr[i] = Vec().load_a(gamma->envr_ + i * Vec::size());            \
  }

#define COMPUTE_GAMMAENV(gamma, Vec, in, out)                                  \
  {                                                                            \
    Vec v = select(gamma##_rms, in * in, abs(in));                             \
    gamma##_env[0] += (v - gamma##_env[0]) * gamma##_enva[0];                  \
    gamma##_env[1] += (gamma##_env5 - gamma##_env[1]) * gamma##_enva[1];       \
    gamma##_env[2] +=                                                          \
      (gamma##_env[4 * 3 + 1] - gamma##_env[2]) * gamma##_enva[2];             \
    gamma##_env[3] +=                                                          \
      (gamma##_env[4 * 3 + 0] - gamma##_env[3]) * gamma##_enva[3];             \
    gamma##_env5 += (gamma##_env[4 * 3 + 0] - gamma##_env5) * gamma##_enva5;   \
    int i;                                                                     \
    for (i = 4; i < 16; i += 4) {                                              \
      gamma##_env[i + 0] +=                                                    \
        (gamma##_env[i - 4] - gamma##_env[i + 0]) * gamma##_enva[0];           \
      gamma##_env[i + 1] +=                                                    \
        (gamma##_env[i - 3] - gamma##_env[i + 1]) * gamma##_enva[1];           \
      gamma##_env[i + 2] +=                                                    \
        (gamma##_env[i - 2] - gamma##_env[i + 2]) * gamma##_enva[2];           \
      gamma##_env[i + 3] +=                                                    \
        (gamma##_env[i - 1] - gamma##_env[i + 3]) * gamma##_enva[3];           \
    }                                                                          \
    Vec resa = (gamma##_env[i - 4] + gamma##_env[i - 3] + gamma##_env[i - 2] - \
                gamma##_env[i - 1] - gamma##_env5);                            \
    auto const increasing = resa >= gamma##_prevr;                             \
    gamma##_envr[0] += (resa - gamma##_envr[0]) * gamma##_envb[0];             \
    gamma##_envr[1] += (gamma##_envr5 - gamma##_envr[1]) * gamma##_envb[1];    \
    gamma##_envr[2] +=                                                         \
      (gamma##_envr[4 * 3 + 1] - gamma##_envr[2]) * gamma##_envb[2];           \
    gamma##_envr[3] +=                                                         \
      (gamma##_envr[4 * 3 + 0] - gamma##_envr[3]) * gamma##_envb[3];           \
    gamma##_envr5 +=                                                           \
      (gamma##_envr[4 * 3 + 0] - gamma##_envr5) * gamma##_envb5;               \
    for (i = 4; i < 16; i += 4) {                                              \
      gamma##_envr[i + 0] +=                                                   \
        (gamma##_envr[i - 4] - gamma##_envr[i + 0]) * gamma##_envb[0];         \
      gamma##_envr[i + 1] +=                                                   \
        (gamma##_envr[i - 3] - gamma##_envr[i + 1]) * gamma##_envb[1];         \
      gamma##_envr[i + 2] +=                                                   \
        (gamma##_envr[i - 2] - gamma##_envr[i + 2]) * gamma##_envb[2];         \
      gamma##_envr[i + 3] +=                                                   \
        (gamma##_envr[i - 1] - gamma##_envr[i + 3]) * gamma##_envb[3];         \
    }                                                                          \
    gamma##_prevr = gamma##_envr[i - 4] + gamma##_envr[i - 3] +                \
                    gamma##_envr[i - 2] - gamma##_envr[i - 1] - gamma##_envr5; \
    for (i = 0; i < 16; i += 4) {                                              \
      gamma##_envr[i + 0] = select(increasing, resa, gamma##_envr[i + 0]);     \
      gamma##_envr[i + 1] = select(increasing, resa, gamma##_envr[i + 1]);     \
      gamma##_envr[i + 2] = select(increasing, resa, gamma##_envr[i + 2]);     \
      gamma##_envr[i + 3] = select(increasing, resa, gamma##_envr[i + 3]);     \
    }                                                                          \
                                                                               \
    gamma##_envr5 = select(increasing, resa, gamma##_envr5);                   \
    gamma##_prevr = select(increasing, resa, gamma##_prevr);                   \
    out = gamma##_to_db_coef *                                                 \
          log(gamma##_prevr + std::numeric_limits<float>::min());              \
  }

#define STORE_GAMMAENV_STATE(gamma, Vec)                                       \
  for (int i = 0; i < 4; ++i) {                                                \
    gamma##_enva[i].store_a(gamma->enva_ + i * Vec::size());                   \
    gamma##_envb[i].store_a(gamma->envb_ + i * Vec::size());                   \
  }                                                                            \
  for (int i = 0; i < 16; ++i) {                                               \
    gamma##_env[i].store_a(gamma->env_ + i * Vec::size());                     \
    gamma##_envr[i].store_a(gamma->envr_ + i * Vec::size());                   \
  }                                                                            \
  gamma##_env5.store_a(gamma->env5_);                                          \
  gamma##_enva5.store_a(gamma->enva5_);                                        \
  gamma##_envb5.store_a(gamma->envb5_);                                        \
  gamma##_envr5.store_a(gamma->envr5_);                                        \
  gamma##_prevr.store_a(gamma->prevr_);
