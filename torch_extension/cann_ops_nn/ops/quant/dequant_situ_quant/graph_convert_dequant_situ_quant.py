# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for DequantSituQuant

try:
    from typing import Optional

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

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.dequant_situ_quant.default)
    def convert_dequant_situ_quant(
        x: Tensor,
        *,
        weight_scale: Optional[Tensor] = None,
        activation_scale: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        quant_scale: Optional[Tensor] = None,
        quant_offset: Optional[Tensor] = None,
        group_index: Optional[Tensor] = None,
        beta: float = 4.0,
        linear_beta: float = 25.0,
        activate_left: bool = True,
        quant_type: str = "dynamic",
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"x": x}
        if weight_scale is not None:
            inputs["weight_scale"] = weight_scale
        if activation_scale is not None:
            inputs["activation_scale"] = activation_scale
        if bias is not None:
            inputs["bias"] = bias
        if quant_scale is not None:
            inputs["quant_scale"] = quant_scale
        if quant_offset is not None:
            inputs["quant_offset"] = quant_offset
        if group_index is not None:
            inputs["group_index"] = group_index

        return ge_op(
            op_type="DequantSituQuant",
            inputs=inputs,
            attrs={
                "beta": attr.Float(beta),
                "linear_beta": attr.Float(linear_beta),
                "activate_left": attr.Bool(activate_left),
                "quant_type": attr.String(quant_type),
            },
            outputs=["y", "y_scale"],
            ir=IrDef("DequantSituQuant")
            .input("x", "DT_INT8", "DT_INT32", "DT_BF16", "DT_FLOAT16")
            .optional_input(
                "weight_scale", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT"
            )
            .optional_input(
                "activation_scale", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT"
            )
            .optional_input("bias", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT")
            .optional_input(
                "quant_scale", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT"
            )
            .optional_input(
                "quant_offset", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT"
            )
            .optional_input(
                "group_index", "DT_INT64", "DT_INT64", "DT_INT64", "DT_INT64"
            )
            .attr("beta", attr.Float(4.0))
            .attr("linear_beta", attr.Float(25.0))
            .attr("activate_left", attr.Bool(True))
            .attr("quant_type", attr.String("dynamic"))
            .output("y", "DT_INT8", "DT_INT8", "DT_INT8", "DT_INT8")
            .output("y_scale", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT", "DT_FLOAT"),
        )

else:

    def convert_dequant_situ_quant(*args, **kwargs):
        raise RuntimeError(
            "DequantSituQuant graph converter: torchair is not available."
        )
