#include "quantizer.hpp"
#include <onnx/onnx_pb.h>
#include <iostream>

float Quantizer::scale(float max_val, float min_val) const {
    return (max_val - min_val) / 255.0f;
}

int Quantizer::zero_point(float min_val, float scale) const {
    return static_cast<int>(std::round(-min_val / scale));
}

int8_t Quantizer::quantized(float float_val, float scale, int zero_point) const {
    if (scale == 0.0f) {
        std::cout << "Error: Scale cannot be 0" << std::endl;
        return 0;
    }
    float value = std::round((float_val / scale) + zero_point);
    value = clamp(value, -128.0f, 127.0f);
    return static_cast<int8_t>(value);
}

std::vector<int8_t> Quantizer::quantize(const std::vector<float>& weights) const {
    if (weights.empty()) return {};

    auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
    float min = *min_it;
    float max = *max_it;

    float s = scale(max, min);
    int z = zero_point(min, s);

    auto view = weights
        | std::views::transform([=](float w) {
            return quantized(w, s, z);
        });

    return std::vector<int8_t>(view.begin(), view.end());
}

int Quantizer::apply(onnx::ModelProto& model){
        std::ifstream input("model.onnx", std::ios::in | std::ios::binary);
        if(!input){
            std::cerr<< "Error to open file " <<  std::endl;
        }
        if (!model.ParseFromIstream(&input)) {
        std::cerr << "Erro ao fazer parse do modelo ONNX.\n";
        return 1;
    }
    std::cout << "Sucess!" << std::endl;

    

    }