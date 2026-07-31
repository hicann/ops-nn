# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for RmsNormDynamicQuant

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

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.rms_norm_dynamic_quant.default)
    def convert_rms_norm_dynamic_quant(
        x: Tensor,
        gamma: Tensor,
        smooth_scales: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
        epsilon: float = 1e-6,
        dst_type: int = 2,
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"x": x, "gamma": gamma}
        if smooth_scales is not None:
            inputs["smooth_scales"] = smooth_scales
        if beta is not None:
            inputs["beta"] = beta

        return ge_op(
            op_type="RmsNormDynamicQuant",
            inputs=inputs,
            attrs={
                "epsilon": attr.Float(epsilon),
                "dst_type": attr.Int(dst_type),
            },
            outputs=["y", "scale"],
            ir=IrDef("RmsNormDynamicQuant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .input("gamma", "DT_FLOAT16, DT_BF16")
            .optional_input("smooth_scales", "DT_FLOAT16, DT_BF16")
            .optional_input("beta", "DT_FLOAT16, DT_BF16")
            .attr("epsilon", attr.Float(1e-6))
            .attr("dst_type", attr.Int(2))
            .output("y", "DT_INT8")
            .output("scale", "DT_FLOAT"),
        )

else:

    def convert_rms_norm_dynamic_quant(*args, **kwargs):
        raise RuntimeError(
            "RmsNormDynamicQuant graph converter: torchair is not available."
        )
