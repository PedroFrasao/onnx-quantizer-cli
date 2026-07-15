#include "model_loader.hpp"
#include "path_utils.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

Ort::Session ModelLoader(const std::string& model_path_str, Ort::Env& env,
                          const Ort::SessionOptions& session_options) {
    if (!fs::exists(model_path_str)) {
        throw std::runtime_error("File not found: " + model_path_str);
    }

    auto native_path = to_native_path(model_path_str);
    std::cout << "Loaded model: " << model_path_str << std::endl;

    return Ort::Session(env, native_path.c_str(), session_options);
}
