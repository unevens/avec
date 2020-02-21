cmake_minimum_required(VERSION 3.0.0)

project(avec-tests)

set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0074 NEW)
find_package(Boost)
include_directories ( ${Boost_INCLUDE_DIRS} )

include_directories(../)
include_directories(../vectorclass)

if(WIN32)

add_executable(avec-test-avx testing.cpp)
add_executable(avec-test-sse2 testing.cpp)

target_compile_options (avec-test-avx PRIVATE /arch:AVX)
target_compile_options (avec-test-sse2 PRIVATE /arch:SSE2)


endif(WIN32)

if(UNIX)

add_executable(avec-test-native testing.cpp)
add_executable(avec-test testing.cpp)

target_compile_options (avec-test-native PRIVATE -march=native)

endif(UNIX)
