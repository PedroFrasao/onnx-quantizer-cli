#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

Ort::Session ModelLoader(const std::string& model_path_str, Ort::Env& env, const Ort::SessionOptions& session_options) {

    if(!fs::exists(model_path_str)){
        throw std::runtime_error("Arquivo do modelo não encontrado: " + model_path_str);
    }

    std::wstring w_model_path(model_path_str.begin(), model_path_str.end());
    std::cout << "Modelo carregado com sucesso: " << model_path_str << std::endl;

    return Ort::Session(env, w_model_path.c_str(), session_options);
}