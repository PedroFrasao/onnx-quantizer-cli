#!/usr/bin/env bash
# End-to-end smoke test for the CLI. Builds a tiny ONNX model with
# Python/PyTorch, runs info/optimize/quantize through the real binary,
# and checks that nothing crashed and the expected files were produced.
#
# Requirements: python3 with `torch` and `onnx` installed.
#
# Usage (from the project root, after building):
#   ./tests/smoke_test.sh [path-to-binary]
set -euo pipefail

BINARY="${1:-build/onnx_optimizer}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

if [ ! -x "$BINARY" ]; then
    echo "Error: binary not found or not executable at '$BINARY'"
    echo "Build the project first, or pass the binary path as an argument."
    exit 1
fi

if ! python3 -c "import torch, onnx" >/dev/null 2>&1; then
    echo "Error: this script needs 'torch' and 'onnx' installed for python3."
    echo "Install with: pip install torch onnx"
    exit 1
fi

echo "== Generating a tiny test model =="
python3 - "$WORKDIR/model.onnx" <<'PYEOF'
import sys
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.flatten(1))

model = TinyNet().eval()
dummy = torch.randn(1, 3, 8, 8)
torch.onnx.export(model, dummy, sys.argv[1],
                   input_names=["input"], output_names=["output"],
                   opset_version=13)
print(f"Model written to {sys.argv[1]}")
PYEOF

echo
echo "== info =="
"$BINARY" info -i "$WORKDIR/model.onnx"

echo
echo "== optimize =="
"$BINARY" optimize -i "$WORKDIR/model.onnx" -o "$WORKDIR/model_opt.onnx"
[ -f "$WORKDIR/model_opt.onnx" ] || { echo "FAIL: optimized model not created"; exit 1; }

echo
echo "== quantize =="
"$BINARY" quantize -i "$WORKDIR/model.onnx" -o "$WORKDIR/model_quant.onnx"
[ -f "$WORKDIR/model_quant.onnx" ] || { echo "FAIL: quantized model not created"; exit 1; }

echo
echo "== comparing weights =="
python3 "$(dirname "$0")/compare_weights.py" "$WORKDIR/model.onnx" "$WORKDIR/model_quant.onnx"

echo
echo "OK: smoke test passed"
