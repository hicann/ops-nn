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

# 参数设置
T = 8
H = 128
twoH = 2 * H

# 创建输入 tensor
grad_output = torch.randn(T, H, dtype=torch.float16).npu()
x = torch.randn(T, twoH, dtype=torch.float16).npu()
weight = torch.randn(T, 1, dtype=torch.float32).npu()
y_origin = torch.randn(T, H, dtype=torch.float16).npu()
group_index = torch.tensor([8], dtype=torch.int64).npu()

# 调用算子
grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
    grad_output,
    x,
    weight=weight,
    y_origin=y_origin,
    group_index=group_index,
    clamp_limit=7.0,
)

print(f"grad_x shape: {grad_x.shape}")
print(f"grad_x dtype: {grad_x.dtype}")
print(f"grad_x[0, :10]: {grad_x[0, :10]}")
print(f"grad_weight shape: {grad_weight.shape}")
print(f"grad_weight dtype: {grad_weight.dtype}")
print(f"grad_weight[0, :10]: {grad_weight[0, :10]}")
