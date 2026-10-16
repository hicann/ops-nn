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

FP4_IN_BYTE = 2
SUPPORTED_SCALE_ALGS = (0, 1)


def _check_scale_alg(scale_alg):
    if scale_alg not in SUPPORTED_SCALE_ALGS:
        raise RuntimeError(
            f"scale_alg should be one of {SUPPORTED_SCALE_ALGS} "
            f"(0=mse_first, 1=unbiased), but got {scale_alg}"
        )


if _TORCHAIR_AVAILABLE:

    def RmsNormHiF4Quant(x, gamma, beta, epsilon, output_rstd, scale_alg):
        """RmsNormHiF4Quant

        REG_OP
        RmsNormHiF4Quant
            INPUT x : tensor
            INPUT gamma : tensor
            OPTIONAL_INPUT beta : tensor
            OUTPUT y : tensor
            OUTPUT scale : tensor
            OPTIONAL_OUTPUT rstd : tensor
            ATTR epsilon : float = 1e-6
            ATTR output_rstd : bool = false
            ATTR scale_alg : int = 0
        """
        y, scale, rstd = ge_op(
            op_type="RmsNormHiF4Quant",
            inputs={"x": x, "gamma": gamma, "beta": beta},
            attrs={
                "epsilon": attr.Float(epsilon),
                "output_rstd": attr.Bool(output_rstd),
                "scale_alg": attr.Int(scale_alg),
            },
            outputs=["y", "scale", "rstd"],
            ir=IrDef("RmsNormHiF4Quant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .input("gamma", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("beta", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .attr("epsilon", attr.Float(1e-6))
            .attr("output_rstd", attr.Bool(False))
            .attr("scale_alg", attr.Int(0))
            .output("y", "DT_HIFLOAT4")
            .output("scale", "DT_HIFLOAT4_SCALE")
            .optional_output("rstd", "DT_FLOAT"),
        )
        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4)
        y = _pack_hif4_to_uint8(x, y)
        scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT4_SCALE)
        rstd.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT)
        return y, scale, rstd

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

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.rms_norm_hi_f4_quant.default)
    def convert_rms_norm_hi_f4_quant(
        x: Tensor,
        gamma: Tensor,
        *,
        beta_optional: Tensor = None,
        epsilon: float = 1e-6,
        output_rstd: bool = False,
        scale_alg: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        _check_scale_alg(scale_alg)
        return RmsNormHiF4Quant(
            x, gamma, beta_optional, epsilon, output_rstd, scale_alg
        )


else:

    def convert_rms_norm_hi_f4_quant(*args, **kwargs):
        raise RuntimeError(
            "RmsNormHiF4Quant graph converter: torchair is not available."
        )
