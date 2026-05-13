cmake_minimum_required(VERSION 3.16)
project(avec-tests CXX)

# Driven by the run-tests.sh script per-arch. The script invokes cmake with
# CMAKE_OSX_ARCHITECTURES set explicitly (e.g. "arm64" or "x86_64") so the
# build dir is single-arch and tests can actually run via `arch -<slice>`.
# If invoked manually without CMAKE_OSX_ARCHITECTURES the host arch is used.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_policy(SET CMP0074 NEW)

include_directories(../)
include_directories(../vectorclass)

if(UNIX AND NOT APPLE)
  # On non-Apple UNIX we keep the old defaults: ASan in Debug + -march=native.
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  add_executable(avec-test-native testing.cpp)
  target_compile_options(avec-test-native PUBLIC -march=native)
endif()

add_executable(avec-test testing.cpp)

if(WIN32)
  add_executable(avec-test-avx  testing.cpp)
  add_executable(avec-test-sse2 testing.cpp)
  target_compile_options(avec-test-avx  PRIVATE /arch:AVX)
  target_compile_options(avec-test-sse2 PRIVATE /arch:SSE2)
endif()

enable_testing()
add_test(NAME avec-test COMMAND avec-test)
