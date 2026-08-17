# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import cann_ops_nn

DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}


def compare_results(result_npu, result_cpu, dtype, name=""):
    rtol, atol = DTYPE_TOLERANCE.get(dtype, (1e-5, 1e-5))
    result_on_cpu = result_npu.cpu() if result_npu.device.type != "cpu" else result_npu
    max_diff = (result_on_cpu - result_cpu).abs().max().item()
    print(f"{name} - max error: {max_diff:.6f}")
    assert torch.allclose(
        result_on_cpu.float(), result_cpu.float(), rtol=rtol, atol=atol
    ), f"Precision check failed: {name}"


def situ_glu_cpu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    d = x.shape[dim] // 2
    left, right = x.split(d, dim=dim)
    if activate_left:
        gate, up = left, right
    else:
        gate, up = right, left

    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None and linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def situ_glu_grad_cpu(grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    d = x.shape[dim] // 2
    left, right = x.split(d, dim=dim)
    if activate_left:
        gate, up = left, right
    else:
        gate, up = right, left

    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    grad_y = grad_y.to(torch.float32)

    t = torch.tanh(gate / beta)
    s = torch.sigmoid(gate)
    situ_a = beta * t * s
    if linear_beta > 0:
        up_prime = linear_beta * torch.tanh(up / linear_beta)
    else:
        up_prime = up

    grad_situ_a = grad_y * up_prime
    grad_up_prime = grad_y * situ_a

    grad_gate = grad_situ_a * s * ((1 - t * t) + beta * t * (1 - s))
    if linear_beta > 0:
        t_u = torch.tanh(up / linear_beta)
        grad_up = grad_up_prime * (1 - t_u * t_u)
    else:
        grad_up = grad_up_prime

    grad_gate = grad_gate.to(x.dtype)
    grad_up = grad_up.to(x.dtype)
    if activate_left:
        return torch.cat([grad_gate, grad_up], dim=dim)
    else:
        return torch.cat([grad_up, grad_gate], dim=dim)


NETWORK_SHAPES = [
    (2, 32),
    (1, 4624, 6144),
    (15908, 3072),
]

DTYPES = [torch.float32, torch.float16]

NET_DIM = -1
NET_BETA = 4.0
NET_LINEAR_BETA = 25.0
NET_ACTIVATE_LEFT = True


def _make_x(shape, dtype):
    if dtype == torch.float32:
        return torch.randn(*shape, dtype=torch.float32).npu()
    return torch.randn(*shape, dtype=torch.float32).to(dtype).npu()


def _make_grad_y_for(x, dim=-1):
    d = x.shape[dim] // 2
    y_shape = list(x.shape)
    y_shape[dim if dim >= 0 else dim + len(x.shape)] = d
    return torch.randn(*y_shape, dtype=torch.float32).to(x.dtype).npu()


@pytest.mark.parametrize("shape", NETWORK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_situ_glu_grad_network(shape, dtype):
    x = _make_x(shape, dtype)
    grad_y = _make_grad_y_for(x, dim=NET_DIM)
    grad_x = cann_ops_nn.situ_glu_grad(
        grad_y,
        x,
        dim=NET_DIM,
        beta=NET_BETA,
        linear_beta=NET_LINEAR_BETA,
        activate_left=NET_ACTIVATE_LEFT,
    )
    grad_x_cpu = situ_glu_grad_cpu(
        grad_y.cpu(),
        x.cpu(),
        dim=NET_DIM,
        beta=NET_BETA,
        linear_beta=NET_LINEAR_BETA,
        activate_left=NET_ACTIVATE_LEFT,
    )
    assert grad_x.dtype == x.dtype
    compare_results(
        grad_x, grad_x_cpu, dtype, name=f"situ_glu_grad_network_{shape}_{dtype}"
    )
