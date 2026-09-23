# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GE converter for the forward-only SwigluGroupQuantWithDualAxis operator."""

try:
    from typing import Optional
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.swiglu_group_quant_with_dual_axis.default
    )
    def convert_swiglu_group_quant_with_dual_axis(
        x: Tensor,
        weight: Optional[Tensor] = None,
        group_index: Optional[Tensor] = None,
        *,
        dst_type: int = 292,
        quant_mode: int = 1,
        clamp_limit: float = -1.0,
        output_origin: bool = False,
        alpha: float = 1.0,
        bias: float = 0.0,
        meta_outputs: TensorSpec = None,
    ):
        if dst_type not in (35, 36, 291, 292) or quant_mode != 1:
            raise RuntimeError("invalid SwigluGroupQuantWithDualAxis MX attributes")
        if x.rank != 2:
            raise RuntimeError("x must be a 2D tensor")
        if weight is not None and group_index is None:
            raise RuntimeError("weight is only supported when group_index is provided")
        output_dtype = (
            DataType.DT_FLOAT8_E5M2
            if dst_type in (35, 291)
            else DataType.DT_FLOAT8_E4M3FN
        )

        inputs = {"x": x}
        if weight is not None:
            inputs["weight"] = weight
        if group_index is not None:
            inputs["group_index"] = group_index

        outputs = ge_op(
            op_type="SwigluGroupQuantWithDualAxis",
            inputs=inputs,
            attrs={
                "dst_type": attr.Int(output_dtype),
                "quant_mode": attr.Int(quant_mode),
                "clamp_limit": attr.Float(clamp_limit),
                "output_origin": attr.Bool(output_origin),
                "alpha": attr.Float(alpha),
                "bias": attr.Float(bias),
            },
            outputs=["y1", "mxscale1", "y2", "mxscale2", "y_origin"],
            ir=IrDef("SwigluGroupQuantWithDualAxis")
            .input("x", "DT_FLOAT16, DT_BF16")
            .optional_input("weight", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("group_index", "DT_INT64")
            .attr("dst_type", attr.Int(DataType.DT_FLOAT8_E4M3FN))
            .attr("quant_mode", attr.Int(1))
            .attr("clamp_limit", attr.Float(-1.0))
            .attr("output_origin", attr.Bool(False))
            .attr("alpha", attr.Float(1.0))
            .attr("bias", attr.Float(0.0))
            .output("y1", "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("mxscale1", "DT_FLOAT8_E8M0")
            .output("y2", "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("mxscale2", "DT_FLOAT8_E8M0")
            .output("y_origin", "DT_FLOAT16, DT_BF16"),
        )
        y1, mxscale1, y2, mxscale2, y_origin = outputs
        y1.desc.dtype = _ge_dtype_to_ge_proto_dtype(output_dtype)
        mxscale1.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT8_E8M0)
        y2.desc.dtype = _ge_dtype_to_ge_proto_dtype(output_dtype)
        mxscale2.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT8_E8M0)
        y_origin.desc.dtype = x.desc.dtype
        return y1, mxscale1, y2, mxscale2, y_origin
else:

    def convert_swiglu_group_quant_with_dual_axis(*args, **kwargs):
        raise RuntimeError(
            "SwigluGroupQuantWithDualAxis graph converter: torchair is not available"
        )
