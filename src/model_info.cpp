#include "model_info.hpp"
#include <iostream>

namespace {

// The input loop and output loop used to be two copy-pasted blocks that
// differed only in which Ort::Session methods they called. This template
// takes those differences as small lambdas so the printing logic (shape
// formatting, labels) exists exactly once.
template <typename NameFn, typename TypeFn>
void print_io_list(size_t count, const char* label, NameFn get_name, TypeFn get_type_info) {
    for (size_t i = 0; i < count; ++i) {
        auto name = get_name(i);
        auto type_info = get_type_info(i);
        auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();

        std::cout << label << " " << i << " (" << name.get() << ") shape: ";
        for (auto dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
}

} // namespace

void ModelInfo(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;

    print_io_list(
        session.GetInputCount(), "Input",
        [&](size_t i) { return session.GetInputNameAllocated(i, allocator); },
        [&](size_t i) { return session.GetInputTypeInfo(i); });

    print_io_list(
        session.GetOutputCount(), "Output",
        [&](size_t i) { return session.GetOutputNameAllocated(i, allocator); },
        [&](size_t i) { return session.GetOutputTypeInfo(i); });
}
