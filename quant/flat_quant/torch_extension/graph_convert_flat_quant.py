# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for FlatQuant

try:
    from typing import Optional

    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph import ge_apis as ge

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.flat_quant.default)
    def convert_flat_quant(
        x: Tensor,
        kronecker_p1: Tensor,
        kronecker_p2: Tensor,
        clip_ratio: float = 1.0,
        dst_dtype: int = 16,
        dst_type_max: float = 0.0,
        group_list: Optional[Tensor] = None,
        group_list_type: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"x": x, "kronecker_p1": kronecker_p1, "kronecker_p2": kronecker_p2}
        if group_list is not None:
            inputs["group_list"] = group_list

        ge_dst_dtype = 29 if dst_dtype == 16 else 40
        y, quant_scale = ge_op(
            op_type="FlatQuant",
            inputs=inputs,
            attrs={
                "clip_ratio": attr.Float(clip_ratio),
                "dst_dtype": attr.Int(ge_dst_dtype),
                "dst_type_max": attr.Float(dst_type_max),
                "group_list_type": attr.Int(group_list_type),
            },
            outputs=["out", "quant_scale"],
            ir=IrDef("FlatQuant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .input("kronecker_p1", "DT_FLOAT16, DT_BF16")
            .input("kronecker_p2", "DT_FLOAT16, DT_BF16")
            .optional_input("group_list", "DT_INT64")
            .attr("clip_ratio", attr.Float(1.0))
            .attr("dst_dtype", attr.Int(29))
            .attr("dst_type_max", attr.Float(0.0))
            .attr("group_list_type", attr.Int(0))
            .output("out", "DT_INT4, DT_FLOAT4_E2M1")
            .output("quant_scale", "DT_FLOAT, DT_FLOAT8_E8M0"),
        )
        if dst_dtype == 16:
            y_shape_int32 = ge.Div(
                ge.Shape(y), ge.Const([1, 1, 8], dtype=DataType.DT_INT32)
            )
            y_shape_int4_bitcast = ge.ConcatV2(
                [y_shape_int32, ge.Const([8], dtype=DataType.DT_INT32)],
                concat_dim=0,
                N=2,
            )
            y = ge.Bitcast(ge.Reshape(y, y_shape_int4_bitcast), type=DataType.DT_INT32)
            return ge.Reshape(y, y_shape_int32), quant_scale
        else:
            y_shape_uint8 = ge.Div(
                ge.Shape(y), ge.Const([1, 2], dtype=DataType.DT_INT32)
            )
            y_shape_int4_bitcast = ge.ConcatV2(
                [y_shape_uint8, ge.Const([2], dtype=DataType.DT_INT32)],
                concat_dim=0,
                N=2,
            )
            y = ge.Bitcast(ge.Reshape(y, y_shape_int4_bitcast), type=DataType.DT_UINT8)
            return ge.Reshape(y, y_shape_uint8), ge.Bitcast(
                quant_scale, type=DataType.DT_UINT8
            )

else:

    def convert_flat_quant(*args, **kwargs):
        raise RuntimeError("FlatQuant graph converter: torchair is not available.")
