# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for SituMxQuant

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

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.situ_mx_quant.default)
    def convert_situ_mx_quant(
        x: Tensor,
        beta: float = 1.0,
        linear_beta: float = 0.0,
        activate_left: bool = False,
        dst_type: int = 36,
        round_mode: str = "rint",
        meta_outputs: TensorSpec = None,
    ):
        return ge_op(
            op_type="SituMxQuant",
            inputs={"x": x},
            attrs={
                "beta": attr.Float(beta),
                "linear_beta": attr.Float(linear_beta),
                "activate_left": attr.Bool(activate_left),
                "axis": attr.Int(-1),
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
            .attr("dst_type", attr.Int(36))
            .attr("round_mode", attr.String("rint"))
            .output("y", "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("y_scale", "DT_FLOAT8_E8M0"),
        )

else:

    def convert_situ_mx_quant(*args, **kwargs):
        raise RuntimeError("SituMxQuant graph converter: torchair is not available.")
