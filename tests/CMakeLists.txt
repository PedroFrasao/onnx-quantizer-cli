# Downloads and builds GoogleTest automatically on first configure.
# This only requires internet access once; if your build machine is
# offline, install gtest via vcpkg/apt/brew instead and replace this
# FetchContent block with find_package(GTest CONFIG REQUIRED).
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Match the parent project's runtime library setting on Windows so the
# test binary and onnx_optimizer_core link against the same CRT.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests test_quant_math.cpp)
target_link_libraries(unit_tests PRIVATE onnx_optimizer_core GTest::gtest_main)
target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(unit_tests)
