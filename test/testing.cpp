/*
Copyright 2019-2026 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// doctest-based test suite for avec. Replaces the earlier "print SNR / no
// pass-fail" harness with proper REQUIRE / CHECK assertions so a regression
// turns into a non-zero exit code that CI can detect.
//
// Coverage:
//   - SIMD math (exp, log, sin, cos, tan, sqrt) vs std:: per lane, with a
//     per-function ULP tolerance.
//   - Arithmetic operators (+ - * / unary -) on Vec2d / Vec4f.
//   - load / store / load_a / store_a round-trip for aligned + unaligned
//     buffers.
//   - permute2<i, j>: every {0,1}x{0,1} permutation.
//   - InterleavedBuffer<Float>::interleave + deinterleave round-trip,
//     parameterised over channel counts (the old test).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "avec/InterleavedBuffer.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace avec;

// ============================================================================
// Helpers
// ============================================================================

// ULP distance between two finite floats — see Bruce Dawson's writeup. NaN
// produces "infinitely far" (INT_MAX) so a NaN-vs-finite mismatch always
// fails the tolerance gate.
template<class Float>
static int ulp_distance(Float a, Float b)
{
  if (a == b) return 0;
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<int>::max();
  if (std::isinf(a) != std::isinf(b)) return std::numeric_limits<int>::max();

  using UInt = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  UInt ua, ub;
  std::memcpy(&ua, &a, sizeof(Float));
  std::memcpy(&ub, &b, sizeof(Float));

  constexpr UInt signBit = UInt(1) << (sizeof(Float) * 8 - 1);
  // Two's-complement-ish remap so monotone-increasing UInts correspond to
  // monotone-increasing floats. Negative-zero and negative-NaN excluded by
  // the early-out above.
  if (ua & signBit) ua = signBit - ua;
  if (ub & signBit) ub = signBit - ub;
  return ua > ub ? int(ua - ub) : int(ub - ua);
}

// ============================================================================
// SIMD math vs std::
// ============================================================================
//
// Strategy: lift a scalar Float to Vec, call the SIMD math function, compare
// every lane against std::. ULP tolerance is per-function; values come from
// table-driven test inputs covering 0, small, large, near singularities.

template<class Vec, class StdFn, class SimdFn>
static void check_unary(const char* name,
                        const std::vector<typename ScalarTypes<Vec>::Float>& inputs,
                        StdFn stdFn, SimdFn simdFn, int maxUlp)
{
  using Float = typename ScalarTypes<Vec>::Float;
  for (Float x : inputs) {
    Vec v(x);
    Vec r = simdFn(v);
    Float expected = stdFn(x);
    for (int lane = 0; lane < Vec::size(); ++lane) {
      Float got = r[lane];
      int ulps = ulp_distance(got, expected);
      INFO(name << "(" << x << ") lane=" << lane
                << ": simd=" << got << " std=" << expected << " ulps=" << ulps);
      CHECK(ulps <= maxUlp);
    }
  }
}

TEST_CASE("Vec2d math vs std::")
{
  using F = double;
  std::vector<F> generic = { 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 7.0, -7.0,
                             100.0, -100.0, 1e-3, -1e-3, 1e3, -1e3 };
  std::vector<F> positive = { 1e-8, 1e-3, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0,
                              1e3, 1e6, 1e9 };
  // exp blows up beyond ~700 for double; keep inputs in range.
  std::vector<F> expSafe  = { -700.0, -100.0, -10.0, -1.0, -0.5, 0.0,
                              0.5, 1.0, 10.0, 100.0, 700.0 };

  SUBCASE("exp")  { check_unary<Vec2d>("exp",  expSafe,
                                       [](F x){ return std::exp(x);  },
                                       [](Vec2d v){ return exp(v);   }, 4); }
  SUBCASE("log")  { check_unary<Vec2d>("log",  positive,
                                       [](F x){ return std::log(x);  },
                                       [](Vec2d v){ return log(v);   }, 4); }
  SUBCASE("sin")  { check_unary<Vec2d>("sin",  generic,
                                       [](F x){ return std::sin(x);  },
                                       [](Vec2d v){ return sin(v);   }, 4); }
  SUBCASE("cos")  { check_unary<Vec2d>("cos",  generic,
                                       [](F x){ return std::cos(x);  },
                                       [](Vec2d v){ return cos(v);   }, 4); }
  SUBCASE("tan")  {
    // Stay away from ±π/2 + kπ where std::tan blows up — comparing two
    // approximations near a pole is noisy and doesn't measure anything useful.
    std::vector<F> tanSafe = { 0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.4, -1.4 };
    check_unary<Vec2d>("tan",  tanSafe,
                       [](F x){ return std::tan(x);  },
                       [](Vec2d v){ return tan(v);   }, 6);
  }
  SUBCASE("sqrt") { check_unary<Vec2d>("sqrt", positive,
                                       [](F x){ return std::sqrt(x); },
                                       [](Vec2d v){ return sqrt(v);  }, 2); }
}

// ============================================================================
// Arithmetic operators
// ============================================================================

TEST_CASE_TEMPLATE("Vec arithmetic operators", Vec, Vec2d, Vec4f)
{
  using F = typename ScalarTypes<Vec>::Float;
  const F a = F(3.5);
  const F b = F(1.25);

  Vec va(a);
  Vec vb(b);

  SUBCASE("addition") {
    Vec r = va + vb;
    for (int i = 0; i < Vec::size(); ++i) CHECK(r[i] == doctest::Approx(a + b));
  }
  SUBCASE("subtraction") {
    Vec r = va - vb;
    for (int i = 0; i < Vec::size(); ++i) CHECK(r[i] == doctest::Approx(a - b));
  }
  SUBCASE("multiplication") {
    Vec r = va * vb;
    for (int i = 0; i < Vec::size(); ++i) CHECK(r[i] == doctest::Approx(a * b));
  }
  SUBCASE("division") {
    Vec r = va / vb;
    for (int i = 0; i < Vec::size(); ++i) CHECK(r[i] == doctest::Approx(a / b));
  }
  SUBCASE("unary negate") {
    Vec r = -va;
    for (int i = 0; i < Vec::size(); ++i) CHECK(r[i] == doctest::Approx(-a));
  }
  SUBCASE("(a + b) * c == a*c + b*c (FP-tolerant)") {
    const F c = F(0.75);
    Vec vc(c);
    Vec lhs = (va + vb) * vc;
    Vec rhs = va * vc + vb * vc;
    for (int i = 0; i < Vec::size(); ++i)
      CHECK(lhs[i] == doctest::Approx(rhs[i]).epsilon(1e-6));
  }
}

// ============================================================================
// load / store round-trips
// ============================================================================
//
// The DSP path does aligned loads (`load_a`) on aligned-ptr backed state
// (mDsp->wetAmount etc.). Bugs here are subtle — usually only show up at
// specific block sizes or under LTO — so explicit coverage helps.

TEST_CASE("Vec2d aligned load/store round-trip")
{
  alignas(16) double buf[2] = { 3.25, -7.5 };
  Vec2d v;
  v.load_a(buf);
  CHECK(v[0] == 3.25);
  CHECK(v[1] == -7.5);

  alignas(16) double out[2] = { 0.0, 0.0 };
  v.store_a(out);
  CHECK(out[0] == 3.25);
  CHECK(out[1] == -7.5);
}

TEST_CASE("Vec2d unaligned load/store round-trip")
{
  double buf[2] = { 1e-6, -1e6 };
  Vec2d v;
  v.load(buf);
  CHECK(v[0] == 1e-6);
  CHECK(v[1] == -1e6);

  double out[2] = { 0.0, 0.0 };
  v.store(out);
  CHECK(out[0] == 1e-6);
  CHECK(out[1] == -1e6);
}

// ============================================================================
// permute2
// ============================================================================
//
// Curvessor's stereo-link path calls permute2<1, 0> to swap the two channels
// before averaging. Exercise all four combinations to make sure both the
// vectorclass path and the NEON shim agree on the indexing convention.

TEST_CASE("permute2")
{
  Vec2d v(10.0, 20.0);

  Vec2d p00 = permute2<0, 0>(v);
  Vec2d p01 = permute2<0, 1>(v);   // identity
  Vec2d p10 = permute2<1, 0>(v);   // swap
  Vec2d p11 = permute2<1, 1>(v);

  CHECK(p00[0] == 10.0); CHECK(p00[1] == 10.0);
  CHECK(p01[0] == 10.0); CHECK(p01[1] == 20.0);
  CHECK(p10[0] == 20.0); CHECK(p10[1] == 10.0);
  CHECK(p11[0] == 20.0); CHECK(p11[1] == 20.0);
}

// ============================================================================
// InterleavedBuffer round-trip — ported from the original test.
// ============================================================================

template<class Float>
static void run_interleaved_round_trip(uint32_t numChannels, uint32_t samplesPerBlock)
{
  std::vector<std::vector<Float>> channels(numChannels,
                                           std::vector<Float>(samplesPerBlock));
  Float v = 0;
  std::vector<Float*> ptrs(numChannels);
  for (uint32_t c = 0; c < numChannels; ++c) {
    ptrs[c] = channels[c].data();
    for (uint32_t s = 0; s < samplesPerBlock; ++s)
      channels[c][s] = v++;
  }

  InterleavedBuffer<Float> buffer(numChannels, samplesPerBlock);
  buffer.interleave(ptrs.data(), numChannels, samplesPerBlock);

  // Verify per-element via at(c, s).
  for (uint32_t c = 0; c < numChannels; ++c) {
    for (uint32_t s = 0; s < samplesPerBlock; ++s) {
      Float orig = channels[c][s];
      INFO("ch=" << c << " s=" << s);
      CHECK(buffer.at(c, s)[0] == orig);
    }
  }

  // Round-trip through deinterleave.
  for (uint32_t c = 0; c < numChannels; ++c)
    std::fill(channels[c].begin(), channels[c].end(), Float(-1));

  buffer.deinterleave(ptrs.data(), numChannels, samplesPerBlock);

  v = 0;
  for (uint32_t c = 0; c < numChannels; ++c)
    for (uint32_t s = 0; s < samplesPerBlock; ++s) {
      INFO("ch=" << c << " s=" << s);
      CHECK(channels[c][s] == v++);
    }
}

TEST_CASE_TEMPLATE("InterleavedBuffer round-trip", Float, float, double)
{
  // Cover channel counts that exercise both the Vec-aligned path (multiples
  // of Vec width) and the tail path (odd counts).
  for (uint32_t c : { 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 16u }) {
    SUBCASE("") {
      run_interleaved_round_trip<Float>(c, 128);
    }
  }
}
