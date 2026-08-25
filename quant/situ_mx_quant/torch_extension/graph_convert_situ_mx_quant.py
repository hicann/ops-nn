# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


try:
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

# 量化轴固定为最后一维（GE算子axis属性当前仅支持-1）
AXIS_LAST_DIM = -1
# GE算子dst_type属性注册默认值：40=FP4_E2M1（torch扩展层默认36=FP8_E4M3FN）
GE_DST_TYPE_DEFAULT = 40

if _TORCHAIR_AVAILABLE:

    def SituMxQuant(
        x: Tensor,
        beta: float,
        linear_beta: float,
        activate_left: bool,
        axis: int,
        dst_type: int,
        round_mode: str,
    ):
        """
        SituMxQuant 的 GE op 封装，REG_OP 的 IR 定义如下：

        REG_OP(SituMxQuant)
            .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
            .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
            .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
            .ATTR(beta, Float, 1.0f)
            .ATTR(linear_beta, Float, 0.0f)
            .ATTR(activate_left, Bool, false)
            .ATTR(axis, Int, -1)
            .ATTR(dst_type, Int, 40)  # 40=FP4_E2M1
            .ATTR(round_mode, String, "rint")
            .OP_END_FACTORY_REG(SituMxQuant)

        说明：torch扩展层当前仅暴露 dst_type 为 36(FLOAT8_E4M3FN)/35(FLOAT8_E5M2)
        的 FP8 量化路径，因此 y 的 dtype 仅声明 FP8 两种。
        """
        return ge_op(
            op_type="SituMxQuant",
            inputs={"x": x},
            attrs={
                "beta": attr.Float(beta),
                "linear_beta": attr.Float(linear_beta),
                "activate_left": attr.Bool(activate_left),
                "axis": attr.Int(axis),
                "dst_type": attr.Int(dst_type),
                "round_mode": attr.String(round_mode),
            },
            outputs=["y", "y_scale"],
            ir=IrDef("SituMxQuant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .attr("beta", attr.Float(1.0))
            .attr("linear_beta", attr.Float(0.0))
            .attr("activate_left", attr.Bool(False))
            .attr("axis", attr.Int(-1))
            .attr("dst_type", attr.Int(GE_DST_TYPE_DEFAULT))
            .attr("round_mode", attr.String("rint"))
            .output("y", "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("y_scale", "DT_FLOAT8_E8M0"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.situ_mx_quant.default)
    def convert_situ_mx_quant(
        x: Tensor,
        *,
        beta: float = 1.0,
        linear_beta: float = 0.0,
        activate_left: bool = False,
        dst_type: int = 36,
        round_mode: str = "rint",
        meta_outputs: TensorSpec = None,
    ):
        """situ_mx_quant 图模式 Converter，参数顺序与 schema 一致。"""
        return SituMxQuant(
            x,
            beta=beta,
            linear_beta=linear_beta,
            activate_left=activate_left,
            axis=AXIS_LAST_DIM,
            dst_type=dst_type,
            round_mode=round_mode,
        )

else:

    def convert_situ_mx_quant(*args, **kwargs):
        raise RuntimeError("SituMxQuant graph converter: torchair is not available.")
