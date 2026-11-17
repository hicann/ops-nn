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

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.situ_glu_group.default)
    def convert_situ_glu_group(
        x: Tensor,
        *,
        weight: Tensor = None,
        group_index: Tensor = None,
        dim: int = -1,
        beta: float = 1.0,
        linear_beta: float = 0.0,
        activate_left: bool = True,
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"x": x}
        if weight is not None:
            inputs["weight"] = weight
        if group_index is not None:
            inputs["group_index"] = group_index
        return ge_op(
            op_type="SituGluGroup",
            inputs=inputs,
            attrs={
                "dim": attr.Int(dim),
                "beta": attr.Float(beta),
                "linear_beta": attr.Float(linear_beta),
                "activate_left": attr.Bool(activate_left),
            },
            outputs=["y"],
            ir=IrDef("SituGluGroup")
            .input("x", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("weight", "DT_FLOAT")
            .optional_input("group_index", "DT_INT64")
            .attr("dim", attr.Int(-1))
            .attr("beta", attr.Float(1.0))
            .attr("linear_beta", attr.Float(0.0))
            .attr("activate_left", attr.Bool(True))
            .output("y", "DT_FLOAT16, DT_BF16, DT_FLOAT"),
        )
else:

    def convert_situ_glu_group(*args, **kwargs):
        raise RuntimeError("SituGluGroup graph converter: torchair is not available.")
