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
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    # GE DataType枚举值（与GE types.h/aclDataType保持一致）：42=HIFLOAT4，43=HIFLOAT4_SCALE。
    # 旧版本torchair未定义该枚举时以数值兜底，此时算子无法在图模式实际执行（依赖目标CANN版本）。
    _GE_DT_HIFLOAT4 = getattr(DataType, "DT_HIFLOAT4", 42)
    _GE_DT_HIFLOAT4_SCALE = getattr(DataType, "DT_HIFLOAT4_SCALE", 43)

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    def SwigluHiF4Quant(
        x: Tensor,
        group_index: Optional[Tensor],
        *,
        activate_dim: int,
        activate_left: bool,
        swiglu_mode: int,
        clamp_limit: float,
        glu_alpha: float,
        glu_bias: float,
        axis: int,
        scale_alg: int,
    ):
        """
        SwigluHiF4Quant 的 GE op 封装，REG_OP 的 IR 定义如下：

        REG_OP(SwigluHiF4Quant)
            .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
            .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
            .OUTPUT(y, TensorType({DT_HIFLOAT4}))
            .OUTPUT(scale, TensorType({DT_HIFLOAT4_SCALE}))
            .ATTR(activate_dim, Int, -1)
            .ATTR(activate_left, Bool, false)
            .ATTR(swiglu_mode, Int, 0)
            .ATTR(clamp_limit, Float, 7.0f)
            .ATTR(glu_alpha, Float, 1.702f)
            .ATTR(glu_bias, Float, 1.0f)
            .ATTR(axis, Int, -1)
            .ATTR(scale_alg, Int, 0)
        """
        inputs = {"x": x}
        if group_index is not None:
            inputs["group_index"] = group_index

        return ge_op(
            op_type="SwigluHiF4Quant",
            inputs=inputs,
            attrs={
                "activate_dim": attr.Int(activate_dim),
                "activate_left": attr.Bool(activate_left),
                "swiglu_mode": attr.Int(swiglu_mode),
                "clamp_limit": attr.Float(clamp_limit),
                "glu_alpha": attr.Float(glu_alpha),
                "glu_bias": attr.Float(glu_bias),
                "axis": attr.Int(axis),
                "scale_alg": attr.Int(scale_alg),
            },
            outputs=["y", "scale"],
            ir=IrDef("SwigluHiF4Quant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .optional_input("group_index", "DT_INT32, DT_INT64")
            .attr("activate_dim", attr.Int(-1))
            .attr("activate_left", attr.Bool(False))
            .attr("swiglu_mode", attr.Int(0))
            .attr("clamp_limit", attr.Float(7.0))
            .attr("glu_alpha", attr.Float(1.702))
            .attr("glu_bias", attr.Float(1.0))
            .attr("axis", attr.Int(-1))
            .attr("scale_alg", attr.Int(0))
            .output("y", "DT_HIFLOAT4")
            .output("scale", "DT_HIFLOAT4_SCALE"),
        )

    def _pack_hif4_output_to_uint8(y, rank):
        """HIFLOAT4按fp4x2打包为uint8（每字节2个fp4元素），将最后一维除以2"""
        bit_shape = [1] * (rank - 1)
        bit_shape.append(2)
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        y_shape_hif4 = ge.Shape(y)
        y_shape_uint8 = ge.Div(y_shape_hif4, div_x2)
        y_shape_hif4_2bit = ge.ConcatV2(
            [y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
            concat_dim=0,
            N=2,
        )
        y = ge.Bitcast(ge.Reshape(y, y_shape_hif4_2bit), type=DataType.DT_UINT8)
        y = ge.Reshape(y, y_shape_uint8)
        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)
        return y

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.swiglu_hi_f4_quant.default)
    def convert_swiglu_hi_f4_quant(
        x: Tensor,
        *,
        group_index: Optional[Tensor] = None,
        activate_dim: int = -1,
        activate_left: bool = False,
        swiglu_mode: int = 0,
        clamp_limit: float = 7.0,
        glu_alpha: float = 1.702,
        glu_bias: float = 1.0,
        axis: int = -1,
        scale_alg: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        """swiglu_hi_f4_quant 图模式 Converter，参数顺序与 schema 一致。"""
        y, scale = SwigluHiF4Quant(
            x,
            group_index,
            activate_dim=activate_dim,
            activate_left=activate_left,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=glu_alpha,
            glu_bias=glu_bias,
            axis=axis,
            scale_alg=scale_alg,
        )

        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(_GE_DT_HIFLOAT4)
        scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(_GE_DT_HIFLOAT4_SCALE)
        y = _pack_hif4_output_to_uint8(y, x.rank)
        return y, scale

else:

    def convert_swiglu_hi_f4_quant(*args, **kwargs):
        raise RuntimeError(
            "SwigluHiF4Quant graph converter: torchair is not available."
        )
