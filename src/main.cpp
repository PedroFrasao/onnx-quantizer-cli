#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnx/onnx_pb.h>
#include <CLI11.hpp>

#include "model_info.hpp"
#include "model_loader.hpp"
#include "optimizer.hpp"
#include "quantizer.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Loads a ModelProto directly from disk (required for quantization).
// Ort::Session does not expose the raw proto, so we need a separate
// function that uses the protobuf API.
static onnx::ModelProto load_proto(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Failed to open file: " + path);

    onnx::ModelProto proto;
    if (!proto.ParseFromIstream(&ifs))
        throw std::runtime_error("Failed to parse ONNX model: " + path);

    return proto;
}

// Saves a ModelProto to disk.
static void save_proto(const onnx::ModelProto& proto, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Failed to create output file: " + path);

    if (!proto.SerializeToOstream(&ofs))
        throw std::runtime_error("Failed to serialize ONNX model: " + path);
}

// ---------------------------------------------------------------------------
// Subcommand: info
//   Only inspects and prints the model inputs/outputs.
// ---------------------------------------------------------------------------
static int cmd_info(const std::string& input) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_info");
    Ort::SessionOptions opts;

    Ort::Session session = ModelLoader(input, env, opts);
    ModelInfo(session);
    return 0;
}

// ---------------------------------------------------------------------------
// Subcommand: optimize
//   Applies OnnxRuntime graph optimizations and saves the resulting model.
//   Does not change weight types — only fuses operators, removes redundancy, etc.
// ---------------------------------------------------------------------------
static int cmd_optimize(const std::string& input, const std::string& output) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_optimizer");
    Ort::SessionOptions opts;

    std::cout << "[optimize] Applying graph optimizations...\n";
    Ort::Session session = Optimizer(input, output, env, opts);

    std::cout << "[optimize] Done. Model saved to: " << output << "\n\n";
    std::cout << "[optimize] Optimized model info:\n";
    ModelInfo(session);
    return 0;
}

// ---------------------------------------------------------------------------
// Subcommand: quantize
//   Reads the ONNX proto, converts FLOAT32 weights → INT8 (static per-tensor
//   quantization, with scale + zero-point) and saves.
//
//   Why use the proto directly?
//   Ort::Session is an inference session — it does not expose or allow
//   modification of internal weights. To rewrite the model initializers,
//   we must operate at the protobuf level (onnx::ModelProto).
// ---------------------------------------------------------------------------
static int cmd_quantize(const std::string& input, const std::string& output) {
    std::cout << "[quantize] Loading model...\n";
    onnx::ModelProto model = load_proto(input);

    Quantizer q;
    std::cout << "[quantize] Applying INT8 quantization...\n";
    int tensors_quantized = q.apply_qdq(model);

    std::cout << "[quantize] " << tensors_quantized
              << " tensor(s) quantized.\n";

    save_proto(model, output);
    std::cout << "[quantize] Model saved to: " << output << "\n";

    // Opens the saved model with ORT just to print final info.
    // This also validates that the generated file is a valid ONNX model.
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_validate");
        Ort::SessionOptions opts;
        Ort::Session session = ModelLoader(output, env, opts);
        std::cout << "\n[quantize] Quantized model info:\n";
        ModelInfo(session);
    } catch (const std::exception& e) {
        // Warning: ORT may reject INT8 models without full quantization metadata
        // (scale/zero-point as separate inputs in the graph).
        // Manual initializer quantization is valid for weight compression,
        // but may require adjustments for inference with ORT.
        std::cerr << "[warning] ORT failed to open model for validation: "
                  << e.what() << "\n";
        std::cerr << "[warning] File was saved, but may require additional "
                     "metadata for inference.\n";
    }
    return 0;
}




int main(int argc, char** argv) {
    CLI::App app{"ONNX Edge Optimizer — quantize and optimize ONNX models via terminal"};
    app.require_subcommand(1); 

    
    auto* info_cmd = app.add_subcommand("info", "Displays model inputs and outputs");
    std::string info_input;
    info_cmd->add_option("-i,--input", info_input, "Path to ONNX model")->required();

    
    auto* opt_cmd = app.add_subcommand("optimize", "Applies graph optimizations (OnnxRuntime)");
    std::string opt_input, opt_output;
    opt_cmd->add_option("-i,--input",  opt_input,  "Input model")->required();
    opt_cmd->add_option("-o,--output", opt_output, "Output model")->required();

    
    auto* quant_cmd = app.add_subcommand("quantize", "Quantizes weights FLOAT32 → INT8");
    std::string quant_input, quant_output;
    quant_cmd->add_option("-i,--input",  quant_input,  "Input model")->required();
    quant_cmd->add_option("-o,--output", quant_output, "Output model")->required();

    CLI11_PARSE(app, argc, argv);

    try {
        if (info_cmd->parsed())
            return cmd_info(info_input);

        if (opt_cmd->parsed())
            return cmd_optimize(opt_input, opt_output);

        if (quant_cmd->parsed())
            return cmd_quantize(quant_input, quant_output);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}