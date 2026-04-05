# ONNX Quantizer CLI

A command-line tool written in **C++20** for inspecting, optimizing, and quantizing ONNX models.
Built on top of [ONNX Runtime](https://onnxruntime.ai/) and the [ONNX protobuf API](https://onnx.ai/), it operates directly at the graph level — allowing weight manipulation that a standard inference session cannot expose.

> **Early-stage project (Phase 1)**
> First functional version. Phase 2 will focus on full refactoring for code quality, portability, and maintainability. Breaking changes are expected.

---

## Features (Phase 1)

- **`info`** — Inspect model inputs and outputs (names, shapes, types)
- **`optimize`** — Apply ONNX Runtime graph-level optimizations (operator fusion, constant folding, dead node elimination, etc.)
- **`quantize`** — Quantize Conv layer weights from FLOAT32 → INT8 using per-tensor static quantization with QDQ pattern (QuantizeLinear + DequantizeLinear nodes)

---

## Requirements

| Dependency | Version / Note |
|---|---|
| Operating System | Windows x64 (Linux/macOS support planned) |
| Compiler | MSVC — Visual Studio 2022 (v143) |
| CMake | ≥ 3.15 |
| vcpkg | Latest |
| onnxruntime | via vcpkg (`x64-windows-static`) |
| onnx | via vcpkg (`x64-windows-static`) |

> **Note:** The project is currently Windows-only due to path handling and `wstring` conversions. This will be addressed in Phase 2.

---

## Setup

### 1. Install vcpkg (if not already installed)

```bash
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```

### 2. Install dependencies

```bash
C:\vcpkg\vcpkg install onnxruntime:x64-windows-static onnx:x64-windows-static
```

### 3. Clone this repository

```bash
git clone https://github.com/PedroFrasao/onnx-quantizer-cli.git
cd onnx-quantizer-cli
```

### 4. Configure and build with CMake

The `CMakeLists.txt` points to `C:/vcpkg` by default. If your vcpkg is installed elsewhere, update the `CMAKE_TOOLCHAIN_FILE` accordingly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The executable will be available at `build\Release\main.exe`.

---

## Usage

```
main <subcommand> [options]
```

**Inspect a model**
```bash
main info -i "models\mobilenetv2-12.onnx"
```

**Apply graph optimizations**
```bash
main optimize -i "models\mobilenetv2-12.onnx" -o "models\mobilenetv2_optimized.onnx"
```

**Quantize model weights**
```bash
main quantize -i "models\mobilenetv2-12.onnx" -o "models\mobilenetv2_quantized.onnx"
```

---

## Verifying Quantization

You can visually inspect the model using [Netron](https://netron.app). Look for `QuantizeLinear` and `DequantizeLinear` nodes, and check that weight initializers have `INT8` data type.

Alternatively, use Python to compare original and quantized weights:

```python
import onnx
import numpy as np

original  = onnx.load("models/mobilenetv2-12.onnx")
quantized = onnx.load("models/mobilenetv2-12_quantized.onnx")

orig_init  = {t.name: t for t in original.graph.initializer}
quant_init = {t.name: t for t in quantized.graph.initializer}

for name, tensor in list(quant_init.items())[:5]:
    if name not in orig_init:
        continue
    orig  = np.array(onnx.numpy_helper.to_array(orig_init[name]))
    quant = np.array(onnx.numpy_helper.to_array(tensor))

    print(f"\n{name}")
    print(f"  dtype original:   {orig.dtype}")
    print(f"  dtype quantized:  {quant.dtype}")
    print(f"  sample original:  {orig.flat[:6]}")
    print(f"  sample quantized: {quant.flat[:6]}")
```

---

## Project Structure

```
onnx-quantizer-cli/
├── src/
│   ├── main.cpp
│   ├── model_loader.cpp/hpp
│   ├── model_info.cpp/hpp
│   ├── optimizer.cpp/hpp
│   └── quantizer.cpp/hpp      # Core quantization logic
├── models/                    # Place your .onnx test models here
├── CMakeLists.txt
└── README.md
```

---

## Roadmap (Phase 2)

- Major code cleanup and refactoring (remove duplication, improve portability, better error handling, modern C++ practices)
- Support for more operators (Linear, MatMul, Gemm, etc.)
- Per-channel quantization
- Dataset-based calibration (min-max, entropy)
- Linux and macOS support
- Size metrics and quantization error reporting
- Automated tests

---

## License

MIT License *(will be formally applied during Phase 2)*