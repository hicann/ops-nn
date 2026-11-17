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
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class ChunkLossOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("chunk_loss")

    def sources(self):
        return [self.resolve_source("chunk_loss.cpp")]

    def schema(self):
        return (
            "chunk_loss(Tensor hidden, Tensor weight, Tensor labels, float alpha, "
            "int ignore_index=-100) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def chunk_loss_meta(hidden, weight, labels, alpha, ignore_index=-100):
            m = hidden.shape[0]
            loss = hidden.new_empty((), dtype=torch.float32)
            max_out = hidden.new_empty((m,), dtype=torch.float32)
            sum_out = hidden.new_empty((m,), dtype=torch.float32)
            d_h = hidden.new_empty(hidden.shape)
            return loss, max_out, sum_out, d_h


builder = ChunkLossOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def chunk_loss(hidden, weight, labels, alpha, ignore_index=-100):
    op_module = builder.load()
    return op_module.chunk_loss(hidden, weight, labels, alpha, ignore_index)
