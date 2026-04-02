#pragma once

#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>



Ort::Session Optimizer(const std::string& model_path_str, 
                       const std::string& optimized_path, 
                       Ort::Env& env,  
                       Ort::SessionOptions& session_options);