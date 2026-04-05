# ONNX Edge Optimizer

A command-line tool written in **C++20** for inspecting, optimizing, and quantizing ONNX models. Built on top of [ONNX Runtime](https://onnxruntime.ai/) and the [ONNX protobuf API](https://onnx.ai/), it operates directly at the graph level — allowing weight manipulation that a standard inference session cannot expose.

> ⚠️ **Early-stage project.** This is the first phase of a larger planned toolchain. Expect breaking changes as new features are added.

---

## Features

- **`info`** — Inspect model inputs and outputs (names, shapes, types)
- **`optimize`** — Apply OnnxRuntime graph-level optimizations (operator fusion, dead node elimination, etc.)
- **`quantize`** — Quantize Conv layer weights from FLOAT32 → INT8 using per-tensor static quantization with QDQ (QuantizeLinear / DequantizeLinear) nodes inserted into the graph

---

## Requirements

| Dependency | Version |
|---|---|
| Windows | x64 |
| MSVC | Visual Studio 2022 (v143) |
| CMake | ≥ 3.15 |
| vcpkg | latest |
| onnxruntime | via vcpkg (`x64-windows-static`) |
| onnx (protobuf) | via vcpkg (`x64-windows-static`) |

> Linux/macOS support is not available yet. The path handling and `wstring` conversions are currently Windows-specific.

---

## Setup

### 1. Install vcpkg

If you don't have vcpkg installed:

```bash
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
```

### 2. Install dependencies

```bash
C:\vcpkg\vcpkg install onnxruntime:x64-windows-static
C:\vcpkg\vcpkg install onnx:x64-windows-static
```

This may take several minutes on the first run.

### 3. Clone the repository

```bash
git clone https://github.com/your-username/ONNX_OPTIMIZER.git
cd ONNX_OPTIMIZER
```

### 4. Configure with CMake

The `CMakeLists.txt` already points to `C:\vcpkg` by default. If your vcpkg is installed elsewhere, edit this line before configuring:

```cmake
# CMakeLists.txt
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
```

Then run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 5. Build

```bash
cmake --build build --config Release
```

The binary will be available at:

```
build\Release\main.exe
```

---

## Usage

All commands follow the same pattern:

```
main <subcommand> [options]
```

### `info` — Inspect a model

Prints the input and output tensor names and shapes.

```bash
main info -i "path\to\model.onnx"
```

**Example:**

```
main info -i "models\mobilenetv2-12.onnx"

Input 0 (input) shape: 1 3 224 224
Output 0 (output) shape: 1 1000
```

---

### `optimize` — Apply graph optimizations

Applies OnnxRuntime's extended graph optimizations (operator fusion, constant folding, etc.) and saves the result to a new file.

```bash
main optimize -i "path\to\model.onnx" -o "path\to\model_optimized.onnx"
```

**Example:**

```
main optimize -i "models\mobilenetv2-12.onnx" -o "models\mobilenetv2-12_optimized.onnx"

[optimize] Applying graph optimizations...
[optimize] Done. Model saved to: models\mobilenetv2-12_optimized.onnx
```

---

### `quantize` — Quantize model weights

Quantizes the weights of **Conv** layers from FLOAT32 to INT8. For each Conv weight, the tool:

1. Computes the per-tensor scale and zero-point
2. Quantizes the weights to INT8
3. Inserts a `DequantizeLinear` node before the Conv operator (QDQ pattern)
4. Saves the modified model

```bash
main quantize -i "path\to\model.onnx" -o "path\to\model_quantized.onnx"
```

**Example:**

```
main quantize -i "models\mobilenetv2-12.onnx" -o "models\mobilenetv2-12_quantized.onnx"

[quantize] Loading model...
[quantize] Applying INT8 quantization...
[quantize] 52 tensor(s) quantized.
[quantize] Model saved to: models\mobilenetv2-12_quantized.onnx
```

> **Note:** The output shape reported by `info` will remain unchanged after quantization — only the weight data types and graph structure change. To verify the quantization, open the model in [Netron](https://netron.app) and inspect the initializer data types, or use the Python snippet below.

---

## Verifying Quantization

You can use the following Python script to confirm that weights were actually quantized and measure the quantization error:

```python
import onnx
import numpy as np

original  = onnx.load("models/mobilenetv2-12.onnx")
quantized = onnx.load("models/mobilenetv2-12_quantized.onnx")

orig_init  = {t.name: t for t in original.graph.initializer}
quant_init = {t.name: t for t in quantized.graph.initializer}

for name, tensor in list(quant_init.items())[:3]:
    if name not in orig_init:
        continue
    orig  = np.array(onnx.numpy_helper.to_array(orig_init[name]))
    quant = np.array(onnx.numpy_helper.to_array(tensor))

    print(f"\n{name}")
    print(f"  dtype original:   {orig.dtype}")
    print(f"  dtype quantized:  {quant.dtype}")
    print(f"  orig  sample: {orig.flat[:5]}")
    print(f"  quant sample: {quant.flat[:5]}")
```

Alternatively, inspect the model visually with [Netron](https://netron.app) — quantized weights will show `INT8` as their data type.

---

## Project Structure

```
ONNX_OPTIMIZER/
├── include/
│   └── CLI11.hpp              # CLI parsing (header-only)
├── src/
│   ├── main.cpp               # Entry point, subcommand dispatch
│   ├── model_loader.cpp/hpp   # Loads ONNX model into an ORT session
│   ├── model_info.cpp/hpp     # Prints model input/output metadata
│   ├── optimizer.cpp/hpp      # Wraps ORT graph optimization
│   └── quantizer.cpp/hpp      # Quantization logic (QDQ pattern)
├── models/                    # Place your .onnx files here
├── tests/                     # Unit tests (WIP)
├── CMakeLists.txt
└── README.md
```

---

## Roadmap

- [ ] Per-channel quantization
- [ ] Activation quantization (dynamic and static)
- [ ] Linux / macOS support
- [ ] Extended operator coverage beyond Conv
- [ ] Automated accuracy benchmarking after quantization
- [ ] Unit test coverage

---

## License

This project currently has no license defined. All rights reserved until further notice.