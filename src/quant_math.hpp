#pragma once
#include <cstdint>
#include <vector>

// Pure INT8 quantization math. Deliberately has zero dependency on ONNX,
// protobuf, or ONNX Runtime, so it can be unit-tested in isolation
// (see tests/test_quant_math.cpp) without needing a model file, a graph,
// or any I/O at all.
namespace quant_math {

struct QuantResult {
    std::vector<int8_t> data;
    float scale = 1.0f;
    int8_t zero_point = 0;
};

// scale = (max - min) / 255, the width of one quantization step.
float compute_scale(float min_val, float max_val);

// Maps min_val to quantized code -128 (the bottom of the int8 range).
// NOTE: this is the signed-int8 formula. It is intentionally different
// from the classic (unsigned) uint8 formula `-min_val / scale`, which
// maps min_val to code 0 instead -- using that formula here was the
// project's original bug, silently discarding roughly half of the
// available int8 range.
int compute_zero_point(float min_val, float scale);

// Quantizes a single float to int8, clamped to [-128, 127].
int8_t quantize_value(float value, float scale, int zero_point);

// Reconstructs the approximate float value from a quantized code.
float dequantize_value(int8_t q_val, float scale, int zero_point);

// Quantizes a full tensor using per-tensor min/max statistics.
// Handles the empty-input and constant-tensor (min == max) edge cases
// explicitly instead of falling through to a scale-of-zero division.
QuantResult quantize(const std::vector<float>& weights);

// Quantizes then immediately dequantizes -- useful for measuring
// quantization error (e.g. MSE) without touching any ONNX types.
std::vector<float> quantize_dequantize(const std::vector<float>& weights);

} // namespace quant_math
