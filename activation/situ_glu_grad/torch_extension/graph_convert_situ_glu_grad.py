# coding=utf-8
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

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.situ_glu_grad.default)
    def convert_situ_glu_grad(
        grad_y: Tensor,
        x: Tensor,
        *,
        dim: int = -1,
        beta: float = 1.0,
        linear_beta: float = 0.0,
        activate_left: bool = True,
        meta_outputs: TensorSpec = None,
    ):
        return ge_op(
            op_type="SituGluGrad",
            inputs={"grad_y": grad_y, "x": x},
            attrs={
                "dim": attr.Int(dim),
                "beta": attr.Float(beta),
                "linear_beta": attr.Float(linear_beta),
                "activate_left": attr.Bool(activate_left),
            },
            outputs=["grad_x"],
            ir=IrDef("SituGluGrad")
            .input("grad_y", "DT_FLOAT")
            .input("x", "DT_FLOAT")
            .attr("dim", attr.Int(-1))
            .attr("beta", attr.Float(1.0))
            .attr("linear_beta", attr.Float(0.0))
            .attr("activate_left", attr.Bool(True))
            .output("grad_x", "DT_FLOAT"),
        )
else:

    def convert_situ_glu_grad(*args, **kwargs):
        raise RuntimeError("SituGluGrad graph converter: torchair is not available.")
