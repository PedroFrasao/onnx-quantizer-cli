#include "optimizer.hpp"
#include "path_utils.hpp"
#include <iostream>

Ort::Session Optimizer(const std::string& model_path_str,
                        const std::string& optimized_path,
                        Ort::Env& env,
                        Ort::SessionOptions& session_options) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    auto native_optimized_path = to_native_path(optimized_path);
    session_options.SetOptimizedModelFilePath(native_optimized_path.c_str());

    auto native_model_path = to_native_path(model_path_str);
    Ort::Session session(env, native_model_path.c_str(), session_options);

    std::cout << "Optimized model saved in: " << optimized_path << std::endl;
    return session;
}
