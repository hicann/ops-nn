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
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
        torch_dtype_value_to_ge_type,
        torch_dtype_value_to_ge_proto_type,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

BLOCK_FP8_QUANT_MODE = 0
MX_QUANT_MODE = 1
STATIC_HIFP8_QUANT_MODE = 2
DYNAMIC_HIFP8_QUANT_MODE = 3
FLOAT8_E5M2 = 291
GE_DTYPE_FLOAT8_E4M3FN = 36


def _resolve_dst_type(dst_type, quant_mode):
    if quant_mode < BLOCK_FP8_QUANT_MODE or quant_mode > DYNAMIC_HIFP8_QUANT_MODE:
        raise RuntimeError(f"quant_mode should be 0, 1, 2 or 3, but got {quant_mode}")
    if quant_mode in (STATIC_HIFP8_QUANT_MODE, DYNAMIC_HIFP8_QUANT_MODE):
        return DataType.DT_HIFLOAT8

    try:
        resolved = torch_dtype_value_to_ge_type(dst_type)
    except RuntimeError as exc:
        raise RuntimeError(f"unsupported dst_type: {dst_type}") from exc

    supported = (
        DataType.DT_FLOAT8_E5M2,
        DataType.DT_FLOAT8_E4M3FN,
        DataType.DT_FLOAT4_E2M1,
        DataType.DT_FLOAT4_E1M2,
    )
    if resolved not in supported:
        raise RuntimeError(f"unsupported dst_type: {dst_type}")
    if quant_mode != MX_QUANT_MODE and resolved in (
        DataType.DT_FLOAT4_E2M1,
        DataType.DT_FLOAT4_E1M2,
    ):
        raise RuntimeError("dst_type FLOAT4_E2M1/FLOAT4_E1M2 requires quant_mode=1")
    return resolved


def _resolve_output_proto_dtype(dst_type, y_dtype):
    if y_dtype == DataType.DT_HIFLOAT8:
        return _ge_dtype_to_ge_proto_dtype(y_dtype)
    return torch_dtype_value_to_ge_proto_type(dst_type)


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.swiglu_group_quant.default)
    def convert_swiglu_group_quant(
        x: Tensor,
        *,
        weight: Optional[Tensor] = None,
        group_index: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        dst_type: int = FLOAT8_E5M2,
        quant_mode: int = BLOCK_FP8_QUANT_MODE,
        block_size: int = 0,
        round_scale: bool = False,
        clamp_limit: float = -1.0,
        dst_type_max: float = 15.0,
        output_origin: bool = False,
        meta_outputs: TensorSpec = None,
    ):
        y_dtype = _resolve_dst_type(dst_type, quant_mode)

        inputs = {"x": x}
        if weight is not None:
            inputs["weight"] = weight
        if group_index is not None:
            inputs["group_index"] = group_index
        if scale is not None:
            inputs["scale"] = scale

        y, y_scale, y_origin = ge_op(
            op_type="SwigluGroupQuant",
            inputs=inputs,
            attrs={
                "dst_type": attr.Int(y_dtype),
                "quant_mode": attr.Int(quant_mode),
                "block_size": attr.Int(block_size),
                "round_scale": attr.Bool(round_scale),
                "clamp_limit": attr.Float(clamp_limit),
                "dst_type_max": attr.Float(dst_type_max),
                "output_origin": attr.Bool(output_origin),
            },
            outputs=["y", "y_scale", "y_origin"],
            ir=IrDef("SwigluGroupQuant")
            .input("x", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("weight", "DT_FLOAT")
            .optional_input("group_index", "DT_INT64")
            .optional_input("scale", "DT_FLOAT")
            .attr("dst_type", attr.Int(GE_DTYPE_FLOAT8_E4M3FN))
            .attr("quant_mode", attr.Int(0))
            .attr("block_size", attr.Int(0))
            .attr("round_scale", attr.Bool(False))
            .attr("clamp_limit", attr.Float(-1.0))
            .attr("dst_type_max", attr.Float(15.0))
            .attr("output_origin", attr.Bool(False))
            .output(
                "y",
                "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_HIFLOAT8",
            )
            .output("y_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .output("y_origin", "DT_FLOAT16, DT_BF16, DT_FLOAT"),
        )

        y.desc.dtype = _resolve_output_proto_dtype(dst_type, y_dtype)
        y_scale_dtype = (
            DataType.DT_FLOAT8_E8M0
            if quant_mode == MX_QUANT_MODE
            else DataType.DT_FLOAT
        )
        y_scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(y_scale_dtype)
        if output_origin:
            y_origin.desc.dtype = x.desc.dtype
        else:
            y_origin = ge.Fill([0], ge.Cast(0.0, dst_type=x.dtype))
        return y, y_scale, y_origin
else:

    def convert_swiglu_group_quant(*args, **kwargs):
        raise RuntimeError(
            "SwigluGroupQuant graph converter: torchair is not available."
        )
