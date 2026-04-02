#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "model_info.hpp"
#include "model_loader.hpp"
#include "optimizer.hpp"
#include "quantizer.hpp"
#include <CLI11.hpp>



int main(int argc, char** argv) {
    CLI::App app{"ONNX Edge Optimizer"};

    std::string input_model;
    std::string output_model;

    app.add_option("-i,--input", input_model, "Modelo de entrada")->required();
    app.add_option("-o,--output", output_model, "Modelo de saida")->required();

    CLI11_PARSE(app, argc, argv);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_optimizer");
    Ort::SessionOptions session_options;

    try {
        // 1. Otimizar e salvar
        Ort::Session session = Optimizer(input_model, output_model, env, session_options);

        // 2. Inspecionar o modelo otimizado
        ModelInfo(session);

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}