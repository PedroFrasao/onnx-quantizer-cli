#include <iostream>
#include <string>
#include <filesystem>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "model_info.hpp"
#include "model_loader.hpp"
#include "optimizer.hpp"
#include "quantizer.hpp"



int main() {
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_workflow");

    
    Ort::SessionOptions session_options;

    
    std::string original_model = "C:/Users/pedro/Desktop/ONNX_OPTIMIZER/models/mobilenetv2-12-qdq.onnx";
    std::string optimized_model = "C:/Users/pedro/Desktop/ONNX_OPTIMIZER/models/model_optimized.onnx";

    try {
        
        Ort::Session session = Optimizer(original_model, optimized_model, env, session_options); 

        std::cout << "Sessao pronta para inferencia com o modelo otimizado!" << std::endl;

        std::cout << "informacoes" << std::endl;

        ModelInfo(session);

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}