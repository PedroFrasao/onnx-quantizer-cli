#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>

void ModelInfo(Ort::Session& session) {
    
    // conversão para Windows
    //std::wstring w_model_path(model_path_str.begin(), model_path_str.end());

    

    auto input_model = session.GetInputCount();
    auto output_model = session.GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    
    for (size_t i = 0; i < input_model; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);

        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Input " << i << " (" << name.get() << ") shape: ";
        for (auto dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    
    for (size_t i = 0; i < output_model; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);

        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Output " << i << " (" << name.get() << ") shape: ";
        for (auto dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
}