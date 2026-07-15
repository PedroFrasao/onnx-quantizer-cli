cmake_minimum_required(VERSION 3.15)

# ---------------------------------------------------------------------------
# vcpkg toolchain (Windows only)
#
# CMAKE_TOOLCHAIN_FILE must be set BEFORE project() to have any effect.
# The previous version of this file called project() first and only
# then set CMAKE_TOOLCHAIN_FILE -- that line had no effect at all; it
# only appeared to work because users were separately passing
# -DCMAKE_TOOLCHAIN_FILE=... on the cmake command line, which is set
# before this file is even parsed and therefore took priority.
#
# This block only runs on Windows, and only if the user hasn't already
# provided a toolchain file, so Linux/macOS builds (which normally rely
# on system packages, or their own vcpkg triplet) are never forced into
# a Windows-specific vcpkg path.
# ---------------------------------------------------------------------------
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    elseif(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

if(WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static")
endif()

project(ONNX_OPTIMIZER CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# /MT (static CRT) only makes sense for MSVC; GCC/Clang have no
# equivalent setting and would simply error out on this line.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(onnxruntime CONFIG REQUIRED)
find_package(onnx CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Core library
#
# Everything except main.cpp lives in a static library so the unit test
# binary (tests/test_quant_math.cpp) can link against the real
# quant_math/quantizer code instead of recompiling or reimplementing it.
# ---------------------------------------------------------------------------
add_library(onnx_optimizer_core STATIC
    src/model_loader.cpp
    src/optimizer.cpp
    src/model_info.cpp
    src/quantizer.cpp
    src/quant_math.cpp
)

target_include_directories(onnx_optimizer_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(onnx_optimizer_core PUBLIC cxx_std_20)
target_link_libraries(onnx_optimizer_core PUBLIC onnxruntime::onnxruntime ${ONNX_LIBRARIES})

if(MSVC)
    target_compile_options(onnx_optimizer_core PRIVATE /W4)
else()
    target_compile_options(onnx_optimizer_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE onnx_optimizer_core)

# The CMake target is still called "main" (so existing scripts/build
# references keep working), but the produced binary is named
# "onnx_optimizer" -- a clearer name for something meant to be
# distributed/run directly.
set_target_properties(main PROPERTIES OUTPUT_NAME onnx_optimizer)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build the unit test suite" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
