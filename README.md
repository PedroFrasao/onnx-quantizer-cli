# ONNX Quantizer CLI

A command-line tool written in **C++20** for inspecting, optimizing, and quantizing ONNX models.
Built on top of [ONNX Runtime](https://onnxruntime.ai/) and the [ONNX protobuf API](https://onnx.ai/), it operates directly at the graph level — allowing weight manipulation that a standard inference session cannot expose.

> **Phase 2 in progress**
> Core refactor, cross-platform support, more operators, and real automated tests have landed. Per-channel quantization and dataset-based calibration are still on the roadmap.

---

## Features

- **`info`** — Inspect model inputs and outputs (names, shapes, types)
- **`optimize`** — Apply ONNX Runtime graph-level optimizations (operator fusion, constant folding, dead node elimination, etc.)
- **`quantize`** — Quantize `Conv`, `Gemm`, and `MatMul` weights from FLOAT32 → INT8 using per-tensor static quantization with the QDQ pattern (QuantizeLinear + DequantizeLinear nodes)

---

## Requirements

| Dependency | Version / Note |
|---|---|
| Operating System | Windows, Linux, macOS |
| Compiler | MSVC 2022 (Windows), GCC ≥ 11 or Clang ≥ 14 (Linux/macOS) — needs C++20 |
| CMake | ≥ 3.15 |
| onnxruntime | via vcpkg, or your system's package manager |
| onnx | via vcpkg, or your system's package manager |

### Windows

Uses [vcpkg](https://github.com/microsoft/vcpkg) in static-triplet mode, same as before:

```bash
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install onnxruntime:x64-windows-static onnx:x64-windows-static
```

`CMakeLists.txt` will auto-detect vcpkg via the `VCPKG_ROOT` environment variable, or fall back to `C:/vcpkg` if that's not set. Set `VCPKG_ROOT` if you installed it elsewhere.

### Linux / macOS

Install onnxruntime and onnx via vcpkg (recommended, matches Windows exactly):

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
~/vcpkg/vcpkg install onnxruntime onnx
export VCPKG_ROOT=~/vcpkg
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

Or, if your distro/Homebrew already ships these packages, just point CMake at them directly (no toolchain file needed) — `find_package(onnxruntime CONFIG REQUIRED)` will pick them up as long as they're discoverable (e.g. installed under a standard prefix, or with `CMAKE_PREFIX_PATH` set).

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The executable is named **`onnx_optimizer`** (previously just `main`):
- Windows: `build\Release\onnx_optimizer.exe`
- Linux/macOS: `build/onnx_optimizer`

The first configure will also download GoogleTest (see [Testing](#testing) below). If your build machine has no internet access, see the note in `tests/CMakeLists.txt` for the offline alternative.

---

## Usage

```
onnx_optimizer <subcommand> [options]
```

**Inspect a model**
```bash
onnx_optimizer info -i models/mobilenetv2-12.onnx
```

**Apply graph optimizations**
```bash
onnx_optimizer optimize -i models/mobilenetv2-12.onnx -o models/mobilenetv2_optimized.onnx
```

**Quantize model weights**
```bash
onnx_optimizer quantize -i models/mobilenetv2-12.onnx -o models/mobilenetv2_quantized.onnx
```

---

## Testing

There are two layers of tests:

### 1. Unit tests (the quantization math)

`src/quant_math.hpp/.cpp` contains the actual quantization formulas (scale, zero-point, clamping) with **zero dependency on ONNX/protobuf**, specifically so it can be unit-tested in isolation. Tests live in `tests/test_quant_math.cpp` and run via CTest:

```bash
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

or run the test binary directly:
```bash
./build/tests/unit_tests        # Linux/macOS
build\tests\Release\unit_tests.exe   # Windows
```

These tests cover, among other things, a regression test for the original signed/unsigned zero-point bug, the constant-tensor (min == max) edge case, and that quantized codes always stay within `[-128, 127]`.

### 2. End-to-end smoke test (the actual CLI)

`tests/smoke_test.sh` builds a tiny model with PyTorch, runs `info`/`optimize`/`quantize` through the real compiled binary, and checks the expected output files appear:

```bash
pip install torch onnx
./tests/smoke_test.sh build/onnx_optimizer
```

### 3. Manual inspection

You can visually inspect a quantized model using [Netron](https://netron.app). Look for `QuantizeLinear`/`DequantizeLinear` nodes and check that weight initializers have `INT8` data type.

To compare original vs. quantized weight values directly:
```bash
python tests/compare_weights.py models/mobilenetv2-12.onnx models/mobilenetv2_quantized.onnx
```

See below for a full step-by-step walkthrough of the whole process, from a clean clone to a passing smoke test.

---

## Project Structure

```
onnx-quantizer-cli/
├── src/
│   ├── main.cpp
│   ├── model_loader.cpp/hpp
│   ├── model_info.cpp/hpp
│   ├── optimizer.cpp/hpp
│   ├── path_utils.hpp          # cross-platform path handling for ORT
│   ├── quant_math.cpp/hpp      # pure quantization math (unit-tested)
│   └── quantizer.cpp/hpp       # graph-level QDQ insertion
├── tests/
│   ├── test_quant_math.cpp     # GoogleTest unit tests
│   ├── compare_weights.py      # manual weight comparison
│   └── smoke_test.sh           # end-to-end CLI smoke test
├── models/                     # sample .onnx models for manual testing
├── CMakeLists.txt
└── README.md
```

---

## Roadmap (Phase 2)

- [x] Major code cleanup and refactoring (remove duplication, improve error handling)
- [x] Support for more operators (Gemm, MatMul, in addition to Conv)
- [x] Linux and macOS support
- [x] Automated tests (unit tests for the quant core + CLI smoke test)
- [x] Basic timing metrics on quantization
- [ ] Per-channel quantization
- [ ] Dataset-based calibration (min-max, entropy)
- [ ] Full quantization error reporting (MSE/SQNR per tensor)

---

## License

MIT License
