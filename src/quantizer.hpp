#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ranges>
#include <unordered_set>
#include <onnx/onnx_pb.h>

struct QuantResult {
    std::vector<int8_t> data;
    float scale;
    int8_t zero_point;
};

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

    
    onnx::TensorProto* find_initializer(onnx::GraphProto* graph, const std::string& name);
    std::vector<float> get_weights(onnx::TensorProto* tensor);
    QuantResult quantize(const std::vector<float>& weights) const; 
    std::vector<float> quantize_dequantize(const std::vector<float>& weights) const;

    static std::unordered_set<std::string> build_protected_set(const onnx::GraphProto& graph);

    float dequantized(int8_t q_val, float scale, int zero_point) const;
    void remove_initializer(onnx::GraphProto* graph, const std::string& name);

public:
  //  int apply(onnx::ModelProto& model);
    int apply_qdq(onnx::ModelProto& model);
};