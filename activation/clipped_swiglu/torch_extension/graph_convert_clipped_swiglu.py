# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    from typing import Optional

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

    def ClippedSwiglu(
        x: Tensor,
        *,
        group_index: Optional[Tensor],
        dim: int = -1,
        alpha: float = 1.702,
        limit: float = 7.0,
        bias: float = 1.0,
        interleaved: bool = True,
        clamp_mode: int = 0,
    ):
        inputs = {"x": x}
        if group_index is not None:
            inputs["group_index"] = group_index

        return ge_op(
            op_type="ClippedSwiglu",
            inputs=inputs,
            attrs={
                "dim": attr.Int(dim),
                "alpha": attr.Float(alpha),
                "limit": attr.Float(limit),
                "bias": attr.Float(bias),
                "interleaved": attr.Bool(interleaved),
                "clamp_mode": attr.Int(clamp_mode),
            },
            outputs=["y"],
            ir=IrDef("ClippedSwiglu")
            .input("x", "DT_BF16, DT_FLOAT16, DT_FLOAT")
            .optional_input("group_index", "DT_INT64")
            .attr("dim", attr.Int(-1))
            .attr("alpha", attr.Float(1.702))
            .attr("limit", attr.Float(7.0))
            .attr("bias", attr.Float(1.0))
            .attr("interleaved", attr.Bool(True))
            .attr("clamp_mode", attr.Int(0))
            .output("y", "DT_BF16, DT_FLOAT16, DT_FLOAT"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.clipped_swiglu.default)
    def convert_clipped_swiglu(
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
        return ClippedSwiglu(
            x,
            group_index=group_index,
            dim=dim,
            alpha=alpha,
            limit=limit,
            bias=bias,
            interleaved=interleaved,
            clamp_mode=clamp_mode,
        )
else:

    def convert_clipped_swiglu(*args, **kwargs):
        raise RuntimeError("ClippedSwiglu graph converter: torchair is not available.")
