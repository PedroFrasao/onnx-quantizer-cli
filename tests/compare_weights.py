#!/usr/bin/env python3
"""
Compares initializer weights between an original and a quantized ONNX
model. Generalized from a one-off script that had hardcoded filenames
that didn't match any file actually produced by the CLI.

Usage:
    python compare_weights.py original.onnx quantized.onnx [--max N]
"""
import argparse
import onnx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", help="Path to the original (float32) ONNX model")
    parser.add_argument("quantized", help="Path to the quantized (int8) ONNX model")
    parser.add_argument("--max", type=int, default=5,
                         help="Max number of tensors to print (default: 5)")
    args = parser.parse_args()

    original = onnx.load(args.original)
    quantized = onnx.load(args.quantized)

    orig_init = {t.name: t for t in original.graph.initializer}
    quant_init = {t.name: t for t in quantized.graph.initializer}

    # Quantized weight tensors are named "<original_name>_quantized" by
    # the CLI's apply_qdq(); strip that suffix to find the matching
    # original tensor.
    matched = 0
    for name, tensor in quant_init.items():
        if not name.endswith("_quantized"):
            continue

        base_name = name[: -len("_quantized")]
        if base_name not in orig_init:
            continue

        matched += 1
        if matched > args.max:
            continue

        orig = onnx.numpy_helper.to_array(orig_init[base_name])
        quant = onnx.numpy_helper.to_array(tensor)

        print(f"\n{base_name}")
        print(f"  dtype original:   {orig.dtype}")
        print(f"  dtype quantized:  {quant.dtype}")
        print(f"  sample original:  {orig.flat[:6]}")
        print(f"  sample quantized: {quant.flat[:6]}")

    if matched == 0:
        print("No matching quantized tensors found. Did quantization run "
              "on this model?")
    else:
        print(f"\n{matched} quantized tensor(s) found "
              f"(showing up to {args.max}).")


if __name__ == "__main__":
    main()
