// int Quantizer::apply(onnx::ModelProto& model){
//     const onnx::GraphProto& graph = model.graph();

//     //percorrer grafo:

//     for(const auto& initializer : graph.initializer()){
//         std::cout << "name: " << initializer.name() << "\n";
//         std::cout << "data type: " << initializer.data_type() << "\n";
//         std::cout << "dimensions" << std::endl;

//         for(auto dim : initializer.dims()){
//             std::cout<< dim << " ";
//         }
//         std::cout << "\n";
//         std::cout << "------------------------\n";
//     }