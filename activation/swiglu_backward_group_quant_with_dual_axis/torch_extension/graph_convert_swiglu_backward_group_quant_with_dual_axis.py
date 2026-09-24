# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# GE converter for graph mode.

try:
    from typing import Optional

    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec, _ge_dtype_to_ge_proto_dtype
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.swiglu_backward_group_quant_with_dual_axis.default
    )
    def convert_swiglu_backward_group_quant_with_dual_axis(
        grad_y: Tensor,
        x: Tensor,
        *,
        weight: Optional[Tensor] = None,
        y_origin: Optional[Tensor] = None,
        group_index: Optional[Tensor] = None,
        clamp_limit: float = -1.0,
        alpha: float = 1.0,
        bias: float = 0.0,
        quant_mode: int = 1,
        dst_type: int = 36,
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"grad_y": grad_y, "x": x}
        if weight is not None:
            inputs["weight"] = weight
            inputs["y_origin"] = y_origin
        if group_index is not None:
            inputs["group_index"] = group_index

        output_names = ["y1", "scale1", "y2", "scale2", "grad_weight"]
        ge_outputs = list(
            ge_op(
                op_type="SwigluBackwardGroupQuantWithDualAxis",
                inputs=inputs,
                attrs={
                    "clamp_limit": attr.Float(clamp_limit),
                    "alpha": attr.Float(alpha),
                    "bias": attr.Float(bias),
                    "quant_mode": attr.Int(quant_mode),
                    "dst_type": attr.Int(dst_type),
                },
                outputs=output_names,
                ir=IrDef("SwigluBackwardGroupQuantWithDualAxis")
                .input("grad_y", "DT_FLOAT16, DT_BF16")
                .input("x", "DT_FLOAT16, DT_BF16")
                .optional_input("weight", "DT_FLOAT16, DT_BF16, DT_FLOAT")
                .optional_input("y_origin", "DT_FLOAT16, DT_BF16")
                .optional_input("group_index", "DT_INT64")
                .attr("clamp_limit", attr.Float(-1.0))
                .attr("alpha", attr.Float(1.0))
                .attr("bias", attr.Float(0.0))
                .attr("quant_mode", attr.Int(1))
                .attr("dst_type", attr.Int(36))
                .output("y1", "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
                .output("scale1", "DT_FLOAT8_E8M0")
                .output("y2", "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
                .output("scale2", "DT_FLOAT8_E8M0")
                .output("grad_weight", "DT_FLOAT16, DT_BF16, DT_FLOAT"),
            )
        )
        specs = (
            meta_outputs if isinstance(meta_outputs, (list, tuple)) else [meta_outputs]
        )
        for output, spec in zip(ge_outputs, specs):
            if spec is not None:
                output.desc.dtype = _ge_dtype_to_ge_proto_dtype(spec.dtype)
        # Keep GE's five-output ABI stable; grad_weight is optional at the
        # PyTorch boundary and is hidden when no weight was supplied.
        return ge_outputs if weight is not None else ge_outputs[:4]

else:

    def convert_swiglu_backward_group_quant_with_dual_axis(*args, **kwargs):
        raise RuntimeError(
            "swiglu_backward_group_quant_with_dual_axis graph converter: "
            "torchair is not available."
        )
