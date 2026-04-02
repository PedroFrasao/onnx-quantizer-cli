#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>

Ort::Session Optimizer(const std::string& model_path_str, 
                       const std::string& optimized_path, 
                       Ort::Env& env,  
                       Ort::SessionOptions& session_options) 
{
    //Ort::SessionOptions session_options;

    // 1. Nível de otimização
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 2. Caminho do modelo otimizado (Windows precisa de wstring)
    std::wstring w_optimized_path(optimized_path.begin(), optimized_path.end());
    session_options.SetOptimizedModelFilePath(w_optimized_path.c_str());

    // 3. Criar a sessão (aplica a otimização)
    std::wstring w_model_path(model_path_str.begin(), model_path_str.end());
    Ort::Session session(env, w_model_path.c_str(), session_options);

    std::cout << "Modelo otimizado salvo em: " << optimized_path << std::endl;
    return std::move(session);
}


