#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <onnx/onnx_pb.h>
#include "quant_math.hpp"

// Applies static per-tensor INT8 QDQ (QuantizeLinear/DequantizeLinear)
// quantization to the weights of supported operators in an ONNX graph.
//
// All the actual quantization math lives in quant_math.hpp/.cpp; this
// class is only responsible for finding candidate weight tensors in the
// graph, reading/writing ONNX TensorProtos, and rewriting the graph to
// insert the QDQ nodes.
class Quantizer {
public:
    // Returns how many tensors were quantized.
    int apply_qdq(onnx::GraphProto& graph);

private:
    onnx::TensorProto* find_initializer(onnx::GraphProto& graph, const std::string& name);
    void remove_initializer(onnx::GraphProto& graph, const std::string& name);

    // Returns nullopt (instead of an empty vector) when the tensor can't
    // be safely read, so callers can tell "no data" apart from "error".
    std::optional<std::vector<float>> get_weights(const onnx::TensorProto* tensor) const;

    // Names of tensors that are already part of an existing QuantizeLinear/
    // DequantizeLinear pair (their data input and their outputs). Used so
    // apply_qdq doesn't try to re-quantize something already quantized,
    // making repeated runs over the same graph idempotent.
    static std::unordered_set<std::string> build_protected_set(const onnx::GraphProto& graph);

    // Builds a TensorProto from a contiguous buffer. Replaces what used
    // to be three near-identical CreateZeroPointTensor/CreateScaleTensor/
    // CreateQuantizeTensor functions.
    template <typename T>
    onnx::TensorProto make_tensor(
        const std::string& name,
        onnx::TensorProto_DataType dtype,
        const std::vector<int64_t>& dims,
        const T* data,
        size_t count) const
    {
        onnx::TensorProto tensor;
        tensor.set_name(name);
        tensor.set_data_type(dtype);
        for (int64_t d : dims) tensor.add_dims(d);
        tensor.set_raw_data(reinterpret_cast<const char*>(data), count * sizeof(T));
        return tensor;
    }

    // Describes, for a supported op_type, which input indices might hold
    // the weight (constant) tensor. For MatMul there's no fixed
    // convention -- the constant operand can be on either side -- so we
    // list both and use whichever one resolves to an initializer.
    struct OpQuantSpec {
        std::vector<int> candidate_input_indices;
    };
    static const std::unordered_map<std::string, OpQuantSpec>& quantizable_ops();
};
