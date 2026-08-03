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

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.swiglu_group.default)
    def convert_swiglu_group(
        x: Tensor,
        *,
        weight: Optional[Tensor] = None,
        group_index: Optional[Tensor] = None,
        clamp_limit: float = -1.0,
        meta_outputs: TensorSpec = None,
    ):
        inputs = {"x": x}
        if weight is not None:
            inputs["weight"] = weight
        if group_index is not None:
            inputs["group_index"] = group_index

        return ge_op(
            op_type="SwigluGroup",
            inputs=inputs,
            attrs={"clamp_limit": attr.Float(clamp_limit)},
            outputs=["y"],
            ir=IrDef("SwigluGroup")
            .input("x", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("weight", "DT_FLOAT")
            .optional_input("group_index", "DT_INT64")
            .attr("clamp_limit", attr.Float(-1.0))
            .output("y", "DT_FLOAT16, DT_BF16, DT_FLOAT"),
        )
else:

    def convert_swiglu_group(*args, **kwargs):
        raise RuntimeError("SwigluGroup graph converter: torchair is not available.")
