# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
    torch.int32: (0, 0),
    torch.int8: (0, 0),
}


def get_tolerance(dtype):
    return DTYPE_TOLERANCE.get(dtype, (1e-5, 1e-5))


def generate_test_data(shape, dtype, device="npu"):
    if dtype in (torch.int32, torch.int8):
        return torch.randint(-100, 100, shape, dtype=dtype, device=device)
    if dtype == torch.bfloat16:
        return torch.randn(*shape, dtype=torch.float32, device=device).to(dtype)
    if dtype == torch.float16:
        return torch.randn(*shape, dtype=dtype, device=device) * 0.1
    return torch.randn(*shape, dtype=dtype, device=device)


def compare_results(result_npu, result_cpu, dtype, name=""):
    rtol, atol = get_tolerance(dtype)
    result_on_cpu = result_npu.cpu() if result_npu.device.type != "cpu" else result_npu
    max_diff = (result_on_cpu - result_cpu).abs().max().item()
    mean_diff = (result_on_cpu - result_cpu).abs().mean().item()
    print(f"{name} - max error: {max_diff:.6f}, mean error: {mean_diff:.6f}")
    if dtype in (torch.int32, torch.int8):
        assert torch.equal(result_on_cpu, result_cpu), (
            f"Exact match check failed: {name}"
        )
    else:
        assert torch.allclose(
            result_on_cpu.float(), result_cpu.float(), rtol=rtol, atol=atol
        ), f"Precision check failed: {name}"
