# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def golden(grad_output, input_data, target, var, eps=1e-6, reduction="none"):
    """Compute GaussianNllLossGrad golden outputs in float32."""
    target_broadcast = np.broadcast_to(target, input_data.shape).astype(np.float32)
    var_broadcast = np.broadcast_to(var, input_data.shape).astype(np.float32)
    safe_var = np.maximum(var_broadcast, np.float32(eps))
    scale = (
        np.float32(1.0 / input_data.size)
        if reduction == "mean" and input_data.size
        else np.float32(1.0)
    )
    upstream = (
        grad_output
        if reduction == "none"
        else np.broadcast_to(grad_output, input_data.shape)
    )
    upstream = upstream.astype(np.float32) * scale
    difference = input_data.astype(np.float32) - target_broadcast
    grad_input = upstream * difference / safe_var
    grad_var_full = (
        upstream
        * np.float32(0.5)
        * (np.float32(1.0) / safe_var - difference * difference / (safe_var * safe_var))
    )
    grad_var = grad_var_full.sum(axis=-1, keepdims=True, dtype=np.float32)
    return grad_input.astype(np.float32), grad_var.astype(np.float32)


def main():
    input_data = np.array([[0.2, -0.1, 1.0], [1.5, 2.0, 2.5]], dtype=np.float32)
    target = np.array([[0.0], [2.0]], dtype=np.float32)
    var = np.array([[0.5], [2.0]], dtype=np.float32)
    grad_output = np.array([[1.0, 0.5, -1.0], [2.0, -0.5, 1.5]], dtype=np.float32)
    grad_input, grad_var = golden(grad_output, input_data, target, var)
    grad_output.tofile("gradOutput.bin")
    input_data.tofile("input.bin")
    target.tofile("target.bin")
    var.tofile("var.bin")
    grad_input.tofile("gradInput_golden.bin")
    grad_var.tofile("gradVar_golden.bin")


if __name__ == "__main__":
    main()
