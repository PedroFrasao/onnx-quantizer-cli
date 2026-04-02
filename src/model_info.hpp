#pragma once

#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>

// Função que imprime informações de input/output de um modelo ONNX
void ModelInfo(Ort::Session& session);