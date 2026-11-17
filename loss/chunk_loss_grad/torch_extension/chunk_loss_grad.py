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
from cann_ops_nn.op_builder import OpBuilder, get_as_library
from torch.library import impl


class ChunkLossGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("chunk_loss_grad")

    def sources(self):
        return [self.resolve_source("chunk_loss_grad.cpp")]

    def schema(self):
        return (
            "chunk_loss_grad(Tensor hidden, Tensor weight, Tensor labels, "
            "float alpha, Tensor softmax_max, Tensor softmax_sum, "
            "Tensor grad_output, int ignore_index=-100) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def chunk_loss_grad_meta(
            hidden,
            weight,
            labels,
            alpha,
            softmax_max,
            softmax_sum,
            grad_output,
            ignore_index=-100,
        ):
            return weight.new_empty(weight.shape)


builder = ChunkLossGradOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _chunk_loss_grad_npu(
    hidden,
    weight,
    labels,
    alpha,
    softmax_max,
    softmax_sum,
    grad_output,
    ignore_index=-100,
):
    op_module = builder.load()
    return op_module.chunk_loss_grad(
        hidden,
        weight,
        labels,
        alpha,
        softmax_max,
        softmax_sum,
        grad_output,
        ignore_index,
    )


def chunk_loss_grad(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    grad_output: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """计算 ChunkLoss 对投影权重的梯度。

    Args:
        hidden: 前向隐藏状态，shape 为 ``[N, H]``，dtype 为 float16 或 bfloat16。
        weight: 投影权重，shape 为 ``[V, H]``，dtype 与 ``hidden`` 一致。
        labels: 标签，shape 为 ``[N]``，dtype 为 int64。
        alpha: 梯度归一化系数。
        softmax_max: 前向 Softmax 的逐行最大值，shape 为 ``[N]``，dtype 为 float32。
        softmax_sum: 前向 Softmax 的逐行指数和，shape 为 ``[N]``，dtype 为 float32。
        grad_output: 上游梯度，标量 Tensor，dtype 为 float32。
        ignore_index: 忽略的标签值，默认值为 -100。

    Returns:
        shape 和 dtype 与 ``weight`` 相同的权重梯度。
    """
    return torch.ops.cann_ops_nn.chunk_loss_grad(
        hidden,
        weight,
        labels,
        alpha,
        softmax_max,
        softmax_sum,
        grad_output,
        ignore_index,
    )


chunk_loss_grad._cann_ops_nn_public_entry_ = True
