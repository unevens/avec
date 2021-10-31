cmake_minimum_required(VERSION 3.0.0)

project(avec-tests)

set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0074 NEW)

include_directories(../)
include_directories(../vectorclass)

if (WIN32)

    add_executable(avec-test-avx testing.cpp)
    add_executable(avec-test-sse2 testing.cpp)

    target_compile_options(avec-test-avx PRIVATE /arch:AVX)
    target_compile_options(avec-test-sse2 PRIVATE /arch:SSE2)


endif (WIN32)

if (UNIX)

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


    #universal binary if building on arm64
    if (APPLE)

        # on MacOS, by default unplug will build an universal binary. you can set avec_override_macos_arch to the architecture
        # you want to build by uncommenting one of the next two lines. It can be useful because Compiler Explorer does not support
        # universal binaries.
        # Note: CMake sometimes does not update this when you reload the project without deleting the build folder manually.
        set(avec_override_macos_arch "")

        if (avec_override_macos_arch STREQUAL "arm64")
            set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
            message(STATUS "Supported architectures: arm64 (forced by user)")
        elseif (avec_override_macos_arch STREQUAL "x86_64")
            set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "")
            message(STATUS "Supported architectures: x86_64 (forced by user)")
        else ()
            execute_process(
                    COMMAND uname -m
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE MACHINE_ARCHITECTURE
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if (MACHINE_ARCHITECTURE STREQUAL "arm64")
                set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "")
                message(STATUS "Supported architectures: x86_64, arm64")
            else ()
                message(STATUS "Supported architectures: ${MACHINE_ARCHITECTURE}")
            endif ()
        endif ()

        add_executable(avec-test testing.cpp)

    else ()
        add_executable(avec-test-native testing.cpp)
        add_executable(avec-test testing.cpp)
        target_compile_options(avec-test-native PUBLIC -march=native)
    endif ()


endif (UNIX)
