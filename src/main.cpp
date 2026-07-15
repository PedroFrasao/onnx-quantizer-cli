#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>

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
    if (!fs::exists(path))
        throw std::runtime_error("Input file not found: " + path);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Failed to open file: " + path);

    onnx::ModelProto proto;
    if (!proto.ParseFromIstream(&ifs))
        throw std::runtime_error("Failed to parse ONNX model (invalid or corrupted file): " + path);

    return proto;
}

// Saves a ModelProto to disk. Creates the output directory if it doesn't
// exist yet, since users commonly pass "-o out/model.onnx" without
// having created "out/" beforehand.
static void save_proto(const onnx::ModelProto& proto, const std::string& path) {
    fs::path out_path(path);
    if (out_path.has_parent_path() && !out_path.parent_path().empty())
        fs::create_directories(out_path.parent_path());

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Failed to create output file: " + path);

    if (!proto.SerializeToOstream(&ofs))
        throw std::runtime_error("Failed to serialize ONNX model: " + path);
}

// Warns (without blocking) when input and output resolve to the same
// file, since quantize/optimize would otherwise silently overwrite the
// source model.
static void warn_if_same_file(const std::string& input, const std::string& output) {
    std::error_code ec;
    if (fs::exists(input, ec) && fs::exists(output, ec) &&
        fs::equivalent(input, output, ec)) {
        std::cerr << "[warning] Input and output point to the same file. "
                     "The original model will be overwritten.\n";
    }
}

// Registers the "-i/--input" and "-o/--output" options shared by the
// optimize and quantize subcommands.
static void add_input_output_options(CLI::App* cmd, std::string& input, std::string& output) {
    cmd->add_option("-i,--input",  input,  "Input model")->required();
    cmd->add_option("-o,--output", output, "Output model")->required();
}

// ---------------------------------------------------------------------------
// Subcommand: info
// ---------------------------------------------------------------------------
static int cmd_info(const std::string& input) {
    if (!fs::exists(input))
        throw std::runtime_error("Input file not found: " + input);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_info");
    Ort::SessionOptions opts;

    Ort::Session session = ModelLoader(input, env, opts);
    ModelInfo(session);
    return 0;
}

// ---------------------------------------------------------------------------
// Subcommand: optimize
//   Applies OnnxRuntime graph optimizations and saves the resulting model.
//   Does not change weight types -- only fuses operators, removes
//   redundancy, etc.
// ---------------------------------------------------------------------------
static int cmd_optimize(const std::string& input, const std::string& output) {
    if (!fs::exists(input))
        throw std::runtime_error("Input file not found: " + input);

    warn_if_same_file(input, output);

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
//   Reads the ONNX proto, converts FLOAT32 weights of supported ops
//   (Conv, Gemm, MatMul) to INT8 via static per-tensor QDQ quantization,
//   and saves.
//
//   Why use the proto directly?
//   Ort::Session is an inference session -- it does not expose or allow
//   modification of internal weights. To rewrite the model initializers
//   we must operate at the protobuf level (onnx::ModelProto).
// ---------------------------------------------------------------------------
static int cmd_quantize(const std::string& input, const std::string& output) {
    warn_if_same_file(input, output);

    std::cout << "[quantize] Loading model...\n";
    onnx::ModelProto model = load_proto(input);
    onnx::GraphProto& graph = *model.mutable_graph();

    Quantizer q;
    std::cout << "[quantize] Applying INT8 quantization...\n";

    const auto t_start = std::chrono::steady_clock::now();
    int tensors_quantized = q.apply_qdq(graph);
    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "[quantize] " << tensors_quantized
              << " tensor(s) quantized in " << elapsed_ms << " ms.\n";

    if (tensors_quantized == 0) {
        std::cerr << "[warning] No tensors were quantized. This usually means "
                     "the model has no supported operators (Conv/Gemm/MatMul) "
                     "with float initializers, or it was already quantized.\n";
    }

    save_proto(model, output);
    std::cout << "[quantize] Model saved to: " << output << "\n";

    // Optional round-trip validation with ORT.
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_validate");
        Ort::SessionOptions opts;
        Ort::Session session = ModelLoader(output, env, opts);
        std::cout << "\n[quantize] Quantized model info:\n";
        ModelInfo(session);
    } catch (const std::exception& e) {
        std::cerr << "[warning] ORT failed to open model for validation: "
                  << e.what() << "\n";
    }

    return 0;
}

int main(int argc, char** argv) {
    CLI::App app{"ONNX Edge Optimizer -- quantize and optimize ONNX models via terminal"};
    app.require_subcommand(1);

    auto* info_cmd = app.add_subcommand("info", "Displays model inputs and outputs");
    std::string info_input;
    info_cmd->add_option("-i,--input", info_input, "Path to ONNX model")->required();

    auto* opt_cmd = app.add_subcommand("optimize", "Applies graph optimizations (OnnxRuntime)");
    std::string opt_input, opt_output;
    add_input_output_options(opt_cmd, opt_input, opt_output);

    auto* quant_cmd = app.add_subcommand("quantize", "Quantizes weights FLOAT32 -> INT8");
    std::string quant_input, quant_output;
    add_input_output_options(quant_cmd, quant_input, quant_output);

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
