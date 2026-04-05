Changelog
src/main.cpp — reescrito completo
Antes: um único fluxo fixo que só chamava Optimizer() + ModelInfo(), sem CLI funcional para escolher operações.
Depois:

load_proto() — novo. Carrega um onnx::ModelProto direto do disco via protobuf. Necessário porque Ort::Session é só inferência e não expõe os pesos internos para modificação.
save_proto() — novo. Serializa o ModelProto modificado de volta para disco.
cmd_info() — novo. Implementação do subcomando info: carrega o modelo e chama ModelInfo().
cmd_optimize() — novo. Implementação do subcomando optimize: chama Optimizer() e imprime info do modelo resultante.
cmd_quantize() — novo. Implementação do subcomando quantize: usa load_proto → Quantizer::apply → save_proto, com validação via ORT no final.
main() — alterado. Adicionado app.require_subcommand(1) e os três subcomandos (info, optimize, quantize) com seus respectivos -i/-o via CLI11.


src/quantizer.cpp — método apply alterado, método build_protected_set criado
build_protected_set() — novo
Percorre todos os nós do grafo antes de qualquer modificação e retorna um unordered_set com os nomes de todos os initializers que não devem ser tocados. A lógica final ficou:

Nós QuantizeLinear/DequantizeLinear → protege seletivamente input[1] (scale, deve ser FLOAT32) e input[2] (zero_point)
Qualquer outro nó (Gemm, Add, Conv, etc.) → protege todos os inputs, pois nenhum deles aceita pesos INT8

Correção importante na implementação: uso de .Get(i) em vez de [i] e tipo explícito const google::protobuf::RepeatedPtrField<std::string>& em vez de const auto&, evitando dangling reference com RepeatedPtrField.
apply() — alterado

Passou a chamar build_protected_set() antes do loop e pular initializers cujo nome esteja no conjunto protegido
Adicionado initializer.clear_raw_data() antes de setar o novo dado (o código original esquecia de limpar esse campo quando o tensor usava float_data)