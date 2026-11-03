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

HIF4_BLOCK_SIZE = 64
SUPPORTED_ROUND_MODES = ("rint", "trunc")
DEFAULT_SCALE_ALG = 0
SCALE_ALG_CHOICES = (0, 1)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _check_round_mode(round_mode):
    if round_mode not in SUPPORTED_ROUND_MODES:
        raise RuntimeError(
            f"round_mode should be one of {SUPPORTED_ROUND_MODES}, but got {round_mode}"
        )


def _validate_scale_alg(scale_alg):
    """scale_alg: 0-标准阈值(4/2), 1-放宽阈值(3.75/1.875)"""
    if scale_alg not in SCALE_ALG_CHOICES:
        raise RuntimeError(
            f"scale_alg must be one of {SCALE_ALG_CHOICES}, but got {scale_alg}"
        )


def _scale1_shape(x):
    shape = list(x.shape)
    shape[-1] = _ceil_div(shape[-1], HIF4_BLOCK_SIZE)
    shape.append(1)
    return shape


def _scale2_shape(x):
    shape = list(x.shape)
    shape[-2] = _ceil_div(shape[-2], HIF4_BLOCK_SIZE)
    shape.append(1)
    return shape


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

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.dynamic_block_hi_f4_quant.default
    )
    def convert_dynamic_block_hi_f4_quant(
        x: Tensor,
        *,
        round_mode: str = "rint",
        scale_alg: int = DEFAULT_SCALE_ALG,
        meta_outputs: TensorSpec = None,
    ):
        _check_round_mode(round_mode)
        _validate_scale_alg(scale_alg)
        y, y_scale1, y_scale2 = ge_op(
            op_type="DynamicBlockHiF4Quant",
            inputs={"x": x},
            attrs={
                "round_mode": attr.Str(round_mode),
                "scale_alg": attr.Int(scale_alg),
            },
            outputs=["y", "scale1", "scale2"],
            ir=IrDef("DynamicBlockHiF4Quant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .attr("round_mode", attr.Str("rint"))
            .attr("scale_alg", attr.Int(DEFAULT_SCALE_ALG))
            .output("y", "DT_HIFLOAT4")
            .output("scale1", "DT_HIFLOAT4_SCALE")
            .output("scale2", "DT_HIFLOAT4_SCALE"),
        )

        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4)
        y = _pack_hif4_to_uint8(x, y)
        y_scale1.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4_SCALE)
        y_scale2.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4_SCALE)
        # Preserve both packed 32-bit scale payloads in torch-visible float32 carriers.
        y_scale1 = ge.Bitcast(y_scale1, type=DataType.DT_FLOAT)
        y_scale2 = ge.Bitcast(y_scale2, type=DataType.DT_FLOAT)
        return y, y_scale1, y_scale2

else:

    def convert_dynamic_block_hi_f4_quant(*args, **kwargs):
        raise RuntimeError(
            "DynamicBlockHiF4Quant graph converter: torchair is not available."
        )
