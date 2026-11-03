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

DST_TYPE_FLOAT16 = 1
DST_TYPE_BFLOAT16 = 27
X_DIM_NUM_MIN = 2
X_DIM_NUM_MAX = 7
SCALE_DIM_NUM_MIN = 3
SCALE_DIM_NUM_MAX = 8


def _check_dst_type(dst_type):
    if dst_type not in (DST_TYPE_FLOAT16, DST_TYPE_BFLOAT16):
        raise RuntimeError(
            f"dst_type should be 1 (float16) or 27 (bfloat16), but got {dst_type}"
        )


def _dst_ge_dtype(dst_type):
    if dst_type == DST_TYPE_FLOAT16:
        return DataType.DT_FLOAT16
    return DataType.DT_BF16


def _check_rank(x, scale):
    if x.rank < X_DIM_NUM_MIN or x.rank > X_DIM_NUM_MAX:
        raise RuntimeError(
            f"x rank should be in [{X_DIM_NUM_MIN}, {X_DIM_NUM_MAX}], but got {x.rank}"
        )
    if scale.rank < SCALE_DIM_NUM_MIN or scale.rank > SCALE_DIM_NUM_MAX:
        raise RuntimeError(
            f"scale rank should be in [{SCALE_DIM_NUM_MIN}, {SCALE_DIM_NUM_MAX}], but got {scale.rank}"
        )


def _unpack_uint8_to_hif4(x):
    bit_shape = [1] * x.rank
    bit_shape[-1] = 2
    mul_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    x_shape_uint8 = ge.Shape(x)
    x_shape_hif4 = ge.Mul(x_shape_uint8, mul_x2)
    x_hif4 = ge.Bitcast(x, type=DataType.DT_HIFLOAT4)
    return ge.Reshape(x_hif4, x_shape_hif4)


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.anti_hi_f4_quant.default)
    def convert_anti_hi_f4_quant(
        x: Tensor,
        scale: Tensor,
        *,
        dst_type: int = 27,
        meta_outputs: TensorSpec = None,
    ):
        _check_dst_type(dst_type)
        _check_rank(x, scale)
        if x.dtype != DataType.DT_UINT8:
            raise RuntimeError("x dtype should be uint8 (packed HiFloat4)")
        if scale.dtype != DataType.DT_FLOAT:
            raise RuntimeError(
                "scale dtype should be float32 (packed HiFloat4_SCALE storage)"
            )
        x_hif4 = _unpack_uint8_to_hif4(x)
        # Both storage types occupy 32 bits, so this preserves shape and payload.
        scale_hif4 = ge.Bitcast(scale, type=DataType.DT_HIFLOAT4_SCALE)
        y = ge_op(
            op_type="AntiHiF4Quant",
            inputs={"x": x_hif4, "scale": scale_hif4},
            attrs={"dst_type": attr.Int(dst_type)},
            outputs=["y"],
            ir=IrDef("AntiHiF4Quant")
            .input("x", "DT_HIFLOAT4")
            .input("scale", "DT_HIFLOAT4_SCALE")
            .attr("dst_type", attr.Int(DST_TYPE_BFLOAT16))
            .output("y", "DT_FLOAT16, DT_BF16"),
        )

        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(_dst_ge_dtype(dst_type))
        return y

else:

    def convert_anti_hi_f4_quant(*args, **kwargs):
        raise RuntimeError("AntiHiF4Quant graph converter: torchair is not available.")
