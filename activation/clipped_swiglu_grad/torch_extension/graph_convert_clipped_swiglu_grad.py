# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

try:
    from typing import Optional

    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        Tensor,
        TensorSpec,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.clipped_swiglu_grad.default)
    def convert_clipped_swiglu_grad(
        grad_y: Tensor,
        x: Tensor,
        *,
        group_index: Optional[Tensor] = None,
        dim: int = -1,
        alpha: float = 1.702,
        limit: float = 7.0,
        bias: float = 1.0,
        interleaved: bool = True,
        clamp_mode: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"grad_y": grad_y, "x": x}
        if group_index is not None:
            inputs["group_index"] = group_index

        grad_x = ge_op(
            op_type="ClippedSwigluGrad",
            inputs=inputs,
            attrs={
                "dim": attr.Int(dim),
                "alpha": attr.Float(alpha),
                "limit": attr.Float(limit),
                "bias": attr.Float(bias),
                "interleaved": attr.Bool(interleaved),
            },
            outputs=["grad_x"],
            ir=IrDef("ClippedSwigluGrad")
            .input("grad_y", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .input("x", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("group_index", "DT_INT64")
            .attr("dim", attr.Int(-1))
            .attr("alpha", attr.Float(1.702))
            .attr("limit", attr.Float(7.0))
            .attr("bias", attr.Float(1.0))
            .attr("interleaved", attr.Bool(True))
            .output("grad_x", "DT_FLOAT16, DT_BF16, DT_FLOAT"),
        )

        grad_x.desc.dtype = x.desc.dtype
        return grad_x
else:

    def convert_clipped_swiglu_grad(*args, **kwargs):
        raise RuntimeError(
            "ClippedSwigluGrad graph converter: torchair is not available."
        )
