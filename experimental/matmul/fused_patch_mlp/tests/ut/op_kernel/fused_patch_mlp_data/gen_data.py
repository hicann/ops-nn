#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
from pathlib import Path

import numpy as np


def parse_shape(value):
    fields = value.strip().strip("()").split(",")
    shape = tuple(int(field.strip()) for field in fields if field.strip())
    if len(shape) < 2 or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid input shape: {value}")
    return shape


def pack_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def round_bf16(values):
    packed = pack_bf16(values)
    return (packed.astype(np.uint32) << 16).view(np.float32)


def round_to_dtype(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":
        return round_bf16(values)
    return values


def gelu_reference(values):
    """Compute the documented standard tanh GELU independently of the kernel."""
    values = np.asarray(values, dtype=np.float32)
    inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)
    return 0.5 * values * (1.0 + np.tanh(inner))


def write_tensor(path, values, dtype, force_float32=False):
    if force_float32 or dtype == "float32":
        np.asarray(values, dtype=np.float32).tofile(path)
    elif dtype == "float16":
        np.asarray(values, dtype=np.float16).tofile(path)
    else:
        pack_bf16(values).tofile(path)


def generate(input_shape_text, hidden_text, layers_text, dtype):
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype: {dtype}")

    input_shape = parse_shape(input_shape_text)
    hidden = int(hidden_text)
    num_layers = int(layers_text)
    if hidden <= 0 or num_layers <= 0:
        raise ValueError("hidden and num_layers must be positive")

    patch = input_shape[-1]
    rows = int(np.prod(input_shape[:-1]))
    rng = np.random.default_rng(42)
    x = round_to_dtype(rng.normal(0.0, 0.5, input_shape).astype(np.float32), dtype)

    input_dims = [patch] + [hidden] * (num_layers - 1)
    weights = [
        round_to_dtype(
            rng.normal(0.0, 0.02, (hidden, in_dim)).astype(np.float32), dtype
        )
        for in_dim in input_dims
    ]
    biases = [np.zeros(hidden, dtype=np.float32) for _ in range(num_layers)]
    if dtype == "float16":
        biases = [round_to_dtype(bias, dtype) for bias in biases]

    current = x.reshape(rows, patch)
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        output = current @ weight.T + bias
        output = round_to_dtype(output, dtype)
        if layer + 1 < num_layers:
            current = round_to_dtype(gelu_reference(output), dtype)

    flat_weights = np.concatenate([weight.T.reshape(-1) for weight in weights])
    flat_biases = np.concatenate(biases)

    write_tensor(f"{dtype}_x_fused_patch_mlp.bin", x, dtype)
    write_tensor(f"{dtype}_weights_fused_patch_mlp.bin", flat_weights, dtype)
    write_tensor(
        f"{dtype}_biases_fused_patch_mlp.bin",
        flat_biases,
        dtype,
        force_float32=dtype == "bfloat16",
    )
    write_tensor(f"{dtype}_golden_fused_patch_mlp.bin", output, dtype)


def main():
    if len(sys.argv) != 5:
        raise SystemExit(f"usage: {sys.argv[0]} SHAPE HIDDEN NUM_LAYERS DTYPE")
    for old_file in Path(".").glob("*.bin"):
        old_file.unlink()
    generate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
