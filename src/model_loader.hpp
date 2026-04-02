#pragma once

#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>


Ort::Session ModelLoader(const std::string& model_path_str,
     Ort::Env& env,
    const Ort::SessionOptions& session_options);