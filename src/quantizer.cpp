#include "quantizer.hpp"
#include <onnx/onnx_pb.h>
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <vector>
#include <ranges>

onnx::TensorProto* Quantizer::find_initializer(onnx::GraphProto* graph, const std::string& name) {
    for (auto& init : *graph->mutable_initializer()) {
        if (init.name() == name) return &init;
    }
    return nullptr;
}

void Quantizer::remove_initializer(onnx::GraphProto* graph, const std::string& name) {
    auto* inits = graph->mutable_initializer();

    for (int i = 0; i < inits->size(); ++i) {
        if ((*inits)[i].name() == name) {
            inits->DeleteSubrange(i, 1);
            return;
        }
    }
}

std::vector<float> Quantizer::get_weights(onnx::TensorProto* tensor) {
    std::vector<float> weights;
    if (!tensor) return weights;

    if (tensor->has_raw_data()) {
        const std::string& raw = tensor->raw_data();
        size_t count = raw.size() / sizeof(float);
        const float* ptr = reinterpret_cast<const float*>(raw.data());
        weights.assign(ptr, ptr + count);
    } else {
        weights.assign(tensor->float_data().begin(), tensor->float_data().end());
    }

    return weights;
}


float Quantizer::scale(float max_val, float min_val) const {
    return (max_val - min_val) / 255.0f;
}

int Quantizer::zero_point(float min_val, float scale) const {
    if (scale == 0.0f) {
        std::cerr << "Error: Scale cannot be 0\n";
        return 0;
    }
    return static_cast<int>(std::round(-min_val / scale));
}

int8_t Quantizer::quantized(float float_val, float scale, int zero_point) const {
    if (scale == 0.0f) {
        std::cerr << "Error: Scale cannot be 0\n";
        return 0;
    }
    float value = std::round((float_val / scale) + zero_point);
    value = std::clamp(value, -128.0f, 127.0f);
    return static_cast<int8_t>(value);
}

float Quantizer::dequantized(int8_t q_val, float scale, int zero_point) const {
    return scale * (static_cast<int>(q_val) - zero_point);
}

QuantResult Quantizer::quantize(const std::vector<float>& weights) const {
    QuantResult result;

    if (weights.empty()) return result;

    auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
    float min_val = *min_it;
    float max_val = *max_it;

    result.scale = scale(max_val, min_val);
    result.zero_point = static_cast<int8_t>(zero_point(min_val, result.scale));

    result.data.reserve(weights.size());
    for (float w : weights) {
        result.data.push_back(quantized(w, result.scale, result.zero_point));
    }

    return result;
}

std::vector<float> Quantizer::quantize_dequantize(const std::vector<float>& weights) const {
    if (weights.empty()) return {};

    auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
    float min = *min_it;
    float max = *max_it;

    if (min == max) return weights;

    float s = scale(max, min);
    int z = zero_point(min, s);

    std::vector<float> result;
    result.reserve(weights.size());

    for (float w : weights) {
        int8_t q = quantized(w, s, z);
        float dq = dequantized(q, s, z);
        result.push_back(dq);
    }

    return result;
}

std::unordered_set<std::string> Quantizer::build_protected_set(const onnx::GraphProto& graph) {
    std::unordered_set<std::string> protected_names;

    for (const auto& node : graph.node()) {
        const google::protobuf::RepeatedPtrField<std::string>& inputs = node.input();
        int n = inputs.size();

        if (node.op_type() == "QuantizeLinear" || node.op_type() == "DequantizeLinear") {
            if (n > 1 && !inputs.Get(1).empty())
                protected_names.insert(inputs.Get(1));
            if (n > 2 && !inputs.Get(2).empty())
                protected_names.insert(inputs.Get(2));
        }
    }

    return protected_names;
}

// int Quantizer::apply(onnx::ModelProto& model) {
//     enum class QuantMode { INT8, FAKE_INT8 };
//     QuantMode mode = QuantMode::FAKE_INT8;

//     onnx::GraphProto* graph = model.mutable_graph();
//     const auto protected_names = build_protected_set(*graph);
//     int cont = 0;

//     for (auto& initializer : *graph->mutable_initializer()) {
//         if (initializer.data_type() != onnx::TensorProto_DataType_FLOAT) continue;
//         if (protected_names.count(initializer.name())) {
//             std::cout << "[quantize] skipping (protected): " << initializer.name() << "\n";
//             continue;
//         }

//         std::vector<float> weights;
//         if (initializer.has_raw_data()) {
//             const std::string& raw = initializer.raw_data();
//             size_t count = raw.size() / sizeof(float);
//             const float* ptr = reinterpret_cast<const float*>(raw.data());
//             weights.assign(ptr, ptr + count);
//         } else {
//             weights.assign(initializer.float_data().begin(), initializer.float_data().end());
//         }

//         // Limpa dados existentes
//         initializer.clear_float_data();
//         initializer.clear_raw_data();

//         if (mode == QuantMode::INT8) {
//             auto q_weights = quantize(weights);
//             initializer.set_data_type(onnx::TensorProto_DataType_INT8);
//             initializer.set_raw_data(q_weights.data(), q_weights.size() * sizeof(int8_t));
//         } else {
//             auto dq_weights = quantize_dequantize(weights);
//             initializer.set_data_type(onnx::TensorProto_DataType_FLOAT);
//             initializer.set_raw_data(reinterpret_cast<const char*>(dq_weights.data()), dq_weights.size() * sizeof(float));
//         }

//         std::cout << "[quantize] processed: " << initializer.name() << " (" << weights.size() << " values)\n";
//         cont++;
//     }

//     return cont;
// }

int Quantizer::apply_qdq(onnx::ModelProto& model) {
    onnx::GraphProto* graph = model.mutable_graph();
    const auto protected_names = build_protected_set(*graph);

    int tensors_quantized = 0;

    for (auto& node : *graph->mutable_node()) {
        if (node.op_type() != "Conv") continue;

        std::string weight_name = node.input(1);

        onnx::TensorProto* weight_tensor = find_initializer(graph, weight_name);
        if (!weight_tensor) continue;

        std::string q_weight = weight_name + "_quantized";
        std::string scale_name = weight_name + "_scale";
        std::string zp_name = weight_name + "_zero_point";
        std::string dq_output = weight_name + "_dequantized";

        // extrair pesos
        std::vector<float> weights = get_weights(weight_tensor);
        if (weights.empty()) continue;

        // quantizar e obter scale / zero_point
        QuantResult qres = quantize(weights);

        //  tensor quantizado 
        auto* q_tensor = graph->add_initializer();
        q_tensor->set_name(q_weight);
        q_tensor->set_data_type(onnx::TensorProto_DataType_INT8);
        for (int i = 0; i < weight_tensor->dims_size(); i++)
            q_tensor->add_dims(weight_tensor->dims(i));

        q_tensor->set_raw_data(
            reinterpret_cast<const char*>(qres.data.data()),
            qres.data.size() * sizeof(int8_t)
        );
        

        //  tensor scale 
        auto* scale_tensor = graph->add_initializer();
        scale_tensor->set_name(scale_name);
        scale_tensor->set_data_type(onnx::TensorProto_DataType_FLOAT);
        scale_tensor->add_dims(1);
        scale_tensor->set_raw_data(
            reinterpret_cast<const char*>(&qres.scale),
            sizeof(float)
        );

        //tensor zero point 
        auto* zp_tensor = graph->add_initializer();
        zp_tensor->set_name(zp_name);
        zp_tensor->set_data_type(onnx::TensorProto_DataType_INT8);
        zp_tensor->add_dims(1);
        zp_tensor->set_raw_data(
            reinterpret_cast<const char*>(&qres.zero_point),
            sizeof(int8_t)

        );

        // criar node DequantizeLinear
        auto* dq_node = graph->add_node();
        dq_node->set_op_type("DequantizeLinear");
        dq_node->add_input(q_weight);
        dq_node->add_input(scale_name);
        dq_node->add_input(zp_name);
        dq_node->add_output(dq_output);
        dq_node->set_name(dq_output + "_node");

        //  substituir input do Conv 
        node.set_input(1, dq_output);

        remove_initializer(graph, weight_name);

        tensors_quantized++;
    }

    return tensors_quantized;
}