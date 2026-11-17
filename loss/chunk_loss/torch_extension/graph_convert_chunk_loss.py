# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode (ChunkLoss)

try:
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.chunk_loss.default)
    def convert_chunk_loss(
        hidden: Tensor,
        weight: Tensor,
        labels: Tensor,
        alpha: float,
        ignore_index: int = -100,
        meta_outputs: TensorSpec = None,
    ):
        return ge_op(
            op_type="ChunkLoss",
            inputs={
                "hidden": hidden,
                "weight": weight,
                "labels": labels,
            },
            attrs={
                "alpha": attr.Float(alpha),
                "ignore_index": attr.Int(ignore_index),
            },
            outputs=["loss", "max", "sum", "dH"],
            ir=IrDef("ChunkLoss")
            .input("hidden", "DT_FLOAT16, DT_BF16")
            .input("weight", "DT_FLOAT16, DT_BF16")
            .input("labels", "DT_INT64")
            .required_attr("alpha", attr.Float)
            .required_attr("ignore_index", attr.Int)
            .output("loss", "DT_FLOAT")
            .output("max", "DT_FLOAT")
            .output("sum", "DT_FLOAT")
            .output("dH", "DT_FLOAT16, DT_BF16"),
        )

else:

    def convert_chunk_loss(*args, **kwargs):
        raise RuntimeError("ChunkLoss graph converter: torchair is not available.")
