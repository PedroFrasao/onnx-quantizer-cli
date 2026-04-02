#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ranges>
#include <onnx/onnx_pb.h>

class Quantizer {
private:
    float scale(float max_val, float min_val) const;
    int zero_point(float min_val, float scale) const;
    int8_t quantized(float float_val, float scale, int zero_point) const;

    template <typename T>
    static T clamp(T value, T min_val, T max_val) {
        if (value > max_val) return max_val;
        if (value < min_val) return min_val;
        return value;
    }

public:
    std::vector<int8_t> quantize(const std::vector<float>& weights) const;

    void Quantizer::apply(onnx::GraphProto* graph);
};