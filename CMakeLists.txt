cmake_minimum_required(VERSION 3.15)

project(ONNX_OPTIMIZER)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(VCPKG_TARGET_TRIPLET "x64-windows-static")

#  ESSENCIAL: forçar /MT corretamente (sem gambiarra)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

find_package(onnxruntime CONFIG REQUIRED)

add_executable(main
    src/model_loader.cpp
    src/optimizer.cpp
    src/model_info.cpp
    src/main.cpp
    src/quantizer.cpp
)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_features(main PRIVATE cxx_std_20)

target_link_libraries(main PRIVATE onnxruntime::onnxruntime)

find_package(onnx CONFIG REQUIRED)
target_link_libraries(main PRIVATE onnx::onnx)