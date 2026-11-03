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

DEFAULT_SCALE_ALG = 0
SCALE_ALG_CHOICES = (0, 1)


def _check_axis(x, axis):
    if axis < -x.rank or axis >= x.rank:
        raise RuntimeError(
            f"axis is out of input dimension range [{-x.rank}, {x.rank}), but got {axis}"
        )


def _validate_scale_alg(scale_alg):
    """scale_alg: 0-标准阈值(4/2), 1-放宽阈值(3.75/1.875)"""
    if scale_alg not in SCALE_ALG_CHOICES:
        raise RuntimeError(
            f"scale_alg must be one of {SCALE_ALG_CHOICES}, but got {scale_alg}"
        )


def _pack_hif4_to_uint8(x, y):
    bit_shape = [1] * x.rank
    bit_shape[-1] = 2
    div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    y_shape_hif4 = ge.Shape(y)
    y_shape_uint8 = ge.Div(y_shape_hif4, div_x2)
    y_shape_hif4_2bit = ge.ConcatV2(
        [y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
        concat_dim=0,
        N=2,
    )
    y_uint8 = ge.Bitcast(ge.Reshape(y, y_shape_hif4_2bit), type=DataType.DT_UINT8)
    return ge.Reshape(y_uint8, y_shape_uint8)


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.dynamic_hi_f4_quant.default)
    def convert_dynamic_hi_f4_quant(
        x: Tensor,
        *,
        axis: int = -1,
        scale_alg: int = DEFAULT_SCALE_ALG,
        meta_outputs: TensorSpec = None,
    ):
        _check_axis(x, axis)
        _validate_scale_alg(scale_alg)
        y, scale = ge_op(
            op_type="DynamicHiF4Quant",
            inputs={"x": x},
            attrs={
                "axis": attr.Int(axis),
                "scale_alg": attr.Int(scale_alg),
            },
            outputs=["y", "scale"],
            ir=IrDef("DynamicHiF4Quant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .attr("axis", attr.Int(-1))
            .attr("scale_alg", attr.Int(DEFAULT_SCALE_ALG))
            .output("y", "DT_HIFLOAT4")
            .output("scale", "DT_HIFLOAT4_SCALE"),
        )

        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4)
        y = _pack_hif4_to_uint8(x, y)
        scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4_SCALE)
        # Preserve the packed 32-bit scale payload in the torch-visible float32 carrier.
        scale = ge.Bitcast(scale, type=DataType.DT_FLOAT)
        return y, scale

else:

    def convert_dynamic_hi_f4_quant(*args, **kwargs):
        raise RuntimeError(
            "DynamicHiF4Quant graph converter: torchair is not available."
        )
