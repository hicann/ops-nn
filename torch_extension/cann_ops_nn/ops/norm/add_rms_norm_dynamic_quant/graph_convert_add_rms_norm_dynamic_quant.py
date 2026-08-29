#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


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
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

_DST_TYPE_MAP = {
    35: DataType.DT_FLOAT8_E5M2,
    36: DataType.DT_FLOAT8_E4M3FN,
    40: DataType.DT_FLOAT4_E2M1,
    41: DataType.DT_FLOAT4_E1M2,
}


def _is_fp4_dtype(ge_dtype):
    return ge_dtype in (DataType.DT_FLOAT4_E2M1, DataType.DT_FLOAT4_E1M2)


def _pack_fp4_output_to_uint8(y, rank):
    """Pack FP4 GE output (shape=..., R, dtype=DT_FLOAT4) into uint8 (shape=..., R//2).

    GE op's infershape sets y shape = x1 shape (unpacked, R elements).
    Torch op contract expects uint8 with two FP4 values packed per byte (R//2 elements).
    Chains GE graph ops: Reshape -> Bitcast -> Reshape.
    """
    bit_shape = [1] * (rank - 1)
    bit_shape.append(2)
    div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    y_shape_fp4 = ge.Shape(y)
    y_shape_uint8 = ge.Div(y_shape_fp4, div_x2)
    y_shape_fp4_2bit = ge.ConcatV2(
        [y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
        concat_dim=0,
        N=2,
    )
    y = ge.Bitcast(ge.Reshape(y, y_shape_fp4_2bit), type=DataType.DT_UINT8)
    y = ge.Reshape(y, y_shape_uint8)
    y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)
    return y


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.add_rms_norm_dynamic_quant.default
    )
    def convert_add_rms_norm_dynamic_quant(
        x1: Tensor,
        x2: Tensor,
        gamma: Tensor,
        beta: Optional[Tensor] = None,
        x3: Optional[Tensor] = None,
        epsilon: float = 1e-6,
        scale_alg: int = 0,
        round_mode: str = "rint",
        dst_type: int = 40,
        output_rstd: bool = False,
        meta_outputs: TensorSpec = None,
    ):
        y_ge_dtype = _DST_TYPE_MAP.get(dst_type)
        if y_ge_dtype is None:
            raise RuntimeError(f"unsupported dst_type: {dst_type}")

        inputs = {"x1": x1, "x2": x2, "gamma": gamma}
        if beta is not None:
            inputs["beta"] = beta
        if x3 is not None:
            inputs["x3"] = x3

        y, x_out, mxscale, rstd = ge_op(
            op_type="AddRmsNormDynamicMxQuant",
            inputs=inputs,
            attrs={
                "epsilon": attr.Float(epsilon),
                "scale_alg": attr.Int(scale_alg),
                "round_mode": attr.Str(round_mode),
                "dst_type": attr.Int(y_ge_dtype),
                # Force True even when user requested False: FX (csrc/meta) always
                # returns rstd with shape [N,1], so GE must produce matching [N,1].
                "output_rstd": attr.Bool(True),
            },
            outputs=["y", "x", "mxscale", "rstd"],
            ir=IrDef("AddRmsNormDynamicMxQuant")
            .input("x1", "DT_FLOAT16, DT_BF16")
            .input("x2", "DT_FLOAT16, DT_BF16")
            .input("gamma", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("beta", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("x3", "DT_FLOAT16, DT_BF16")
            .attr("epsilon", attr.Float(1e-6))
            .attr("scale_alg", attr.Int(0))
            .attr("round_mode", attr.Str("rint"))
            .attr("dst_type", attr.Int(40))
            .attr("output_rstd", attr.Bool(False))
            .output(
                "y", "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2"
            )
            .output("x", "DT_FLOAT16, DT_BF16")
            .output("mxscale", "DT_FLOAT8_E8M0")
            .output("rstd", "DT_FLOAT"),
        )

        if _is_fp4_dtype(y_ge_dtype):
            y = _pack_fp4_output_to_uint8(y, x1.rank)
        else:
            y.desc.dtype = _ge_dtype_to_ge_proto_dtype(y_ge_dtype)
        mxscale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)

        return y, x_out, mxscale, rstd

else:

    def convert_add_rms_norm_dynamic_quant(*args, **kwargs):
        raise RuntimeError(
            "AddRmsNormDynamicMxQuant graph converter: torchair is not available."
        )
