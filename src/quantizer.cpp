#include "quantizer.hpp"
#include <iostream>
#include <cstring>

onnx::TensorProto* Quantizer::find_initializer(onnx::GraphProto& graph, const std::string& name) {
    auto* inits = graph.mutable_initializer();
    for (int i = 0; i < inits->size(); ++i) {
        if ((*inits)[i].name() == name) return &(*inits)[i];
    }
    return nullptr;
}

void Quantizer::remove_initializer(onnx::GraphProto& graph, const std::string& name) {
    auto* inits = graph.mutable_initializer();
    for (int i = 0; i < inits->size(); ++i) {
        if ((*inits)[i].name() == name) {
            inits->DeleteSubrange(i, 1);
            return;
        }
    }
}

std::optional<std::vector<float>> Quantizer::get_weights(const onnx::TensorProto* tensor) const {
    if (!tensor) return std::nullopt;

    if (tensor->data_type() != onnx::TensorProto_DataType_FLOAT) {
        std::cerr << "Warning: tensor '" << tensor->name()
                  << "' is not FLOAT (data_type=" << tensor->data_type()
                  << "), skipping quantization\n";
        return std::nullopt;
    }

    std::vector<float> weights;

    if (tensor->has_raw_data()) {
        const std::string& raw = tensor->raw_data();
        if (raw.size() % sizeof(float) != 0) {
            std::cerr << "Error: raw_data size for '" << tensor->name()
                      << "' is not a multiple of sizeof(float)\n";
            return std::nullopt;
        }
        size_t count = raw.size() / sizeof(float);
        weights.resize(count);
        // memcpy instead of reinterpret_cast'ing raw.data() directly:
        // a std::string's buffer has no guaranteed float alignment, so
        // reading through a reinterpret_cast<const float*> is technically
        // UB and not portable across platforms/optimization levels.
        std::memcpy(weights.data(), raw.data(), raw.size());
    } else {
        weights.assign(tensor->float_data().begin(), tensor->float_data().end());
    }

    if (weights.empty()) {
        std::cerr << "Warning: tensor '" << tensor->name() << "' has no data\n";
        return std::nullopt;
    }

    return weights;
}

std::unordered_set<std::string> Quantizer::build_protected_set(const onnx::GraphProto& graph) {
    std::unordered_set<std::string> protected_names;

    for (const auto& node : graph.node()) {
        if (node.op_type() != "QuantizeLinear" && node.op_type() != "DequantizeLinear")
            continue;

        // input(0) is the data tensor actually being (de)quantized.
        // Protecting it (plus this node's outputs) stops apply_qdq from
        // re-quantizing something that is already part of a QDQ pair.
        if (node.input_size() > 0 && !node.input(0).empty())
            protected_names.insert(node.input(0));

        for (const auto& out : node.output())
            protected_names.insert(out);
    }

    return protected_names;
}

const std::unordered_map<std::string, Quantizer::OpQuantSpec>& Quantizer::quantizable_ops() {
    static const std::unordered_map<std::string, OpQuantSpec> ops = {
        {"Conv",   OpQuantSpec{{1}}},
        {"Gemm",   OpQuantSpec{{1}}},
        // MatMul has no fixed convention for which side holds the
        // constant weight, so both candidate indices are tried.
        {"MatMul", OpQuantSpec{{0, 1}}},
    };
    return ops;
}

int Quantizer::apply_qdq(onnx::GraphProto& graph) {
    const auto protected_names = build_protected_set(graph);
    const auto& specs = quantizable_ops();

    int tensors_quantized = 0;

    // Snapshot the node count: new DequantizeLinear nodes are appended
    // during this loop and must not be reprocessed.
    const int original_node_count = graph.node_size();

    for (int i = 0; i < original_node_count; ++i) {
        auto* node = graph.mutable_node(i);

        auto spec_it = specs.find(node->op_type());
        if (spec_it == specs.end()) continue;

        int weight_input_idx = -1;
        onnx::TensorProto* weight_tensor = nullptr;
        std::string weight_name;

        for (int idx : spec_it->second.candidate_input_indices) {
            if (idx >= node->input_size()) continue;
            const std::string& candidate = node->input(idx);
            if (candidate.empty() || protected_names.count(candidate)) continue;

            if (auto* t = find_initializer(graph, candidate)) {
                weight_input_idx = idx;
                weight_tensor = t;
                weight_name = candidate;
                break;
            }
        }

        if (!weight_tensor) continue;

        auto weights_opt = get_weights(weight_tensor);
        if (!weights_opt) continue;
        const std::vector<float>& weights = *weights_opt;

        std::string q_weight   = weight_name + "_quantized";
        std::string scale_name = weight_name + "_scale";
        std::string zp_name    = weight_name + "_zero_point";
        std::string dq_output  = weight_name + "_dequantized";

        quant_math::QuantResult qres = quant_math::quantize(weights);

        std::vector<int64_t> dims(weight_tensor->dims().begin(), weight_tensor->dims().end());

        auto* q_tensor = graph.add_initializer();
        *q_tensor = make_tensor(q_weight, onnx::TensorProto_DataType_INT8,
                                 dims, qres.data.data(), qres.data.size());

        auto* scale_tensor = graph.add_initializer();
        *scale_tensor = make_tensor(scale_name, onnx::TensorProto_DataType_FLOAT,
                                      std::vector<int64_t>{1}, &qres.scale, 1);

        auto* zp_tensor = graph.add_initializer();
        *zp_tensor = make_tensor(zp_name, onnx::TensorProto_DataType_INT8,
                                   std::vector<int64_t>{1}, &qres.zero_point, 1);

        auto* dq_node = graph.add_node();
        dq_node->set_op_type("DequantizeLinear");
        dq_node->set_name(dq_output + "_node");
        dq_node->add_input(q_weight);
        dq_node->add_input(scale_name);
        dq_node->add_input(zp_name);
        dq_node->add_output(dq_output);

        node->set_input(weight_input_idx, dq_output);

        remove_initializer(graph, weight_name);

        tensors_quantized++;
    }

    return tensors_quantized;
}
