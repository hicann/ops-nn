# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""QuantMaxV2 的 GE 图接入，与算子 Torch 扩展一起发布。"""

try:
    import torch
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import DataType, _ge_dtype_to_ge_proto_dtype
except ImportError:
    # 未安装图后端时，保留普通 ACLNN 调用能力。
    _GE_AVAILABLE = False
else:
    _GE_AVAILABLE = True


if _GE_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.quant_max.default)
    def convert_quant_max_v2(
        x,
        scale=None,
        group_list=None,
        *,
        round_mode,
        dst_type,
        dst_type_max=15.0,
        meta_outputs=None,
    ):
        dst_types = {
            34: DataType.DT_HIFLOAT8,
            35: DataType.DT_FLOAT8_E5M2,
            36: DataType.DT_FLOAT8_E4M3FN,
        }
        if dst_type not in dst_types:
            raise ValueError("QuantMaxV2 dst_type must be 34, 35 or 36")
        inputs = {"x": x}
        if scale is not None:
            inputs["scale"] = scale
        if group_list is not None:
            inputs["group_list"] = group_list
        y, amax, y_scale = ge_op(
            op_type="QuantMaxV2",
            inputs=inputs,
            attrs={
                "round_mode": attr.Str(round_mode),
                "dst_type": attr.Int(dst_type),
                "dst_type_max": attr.Float(dst_type_max),
            },
            outputs=["y", "amax", "y_scale"],
            ir=IrDef("QuantMaxV2")
            .input("x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("scale", "DT_FLOAT")
            .optional_input("group_list", "DT_INT64")
            .required_attr("round_mode", attr.Str)
            .required_attr("dst_type", attr.Int)
            .attr("dst_type_max", attr.Float(15.0))
            .output("y", "DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN")
            .output("amax", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .output("y_scale", "DT_FLOAT"),
        )
        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(dst_types[dst_type])
        amax.desc.dtype = x.desc.dtype
        y_scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT)
        if dst_type == 34:
            # Torch 用 uint8 保存 HIFLOAT8 原始字节，只重解释位模式，不进行数值转换。
            y = ge.Bitcast(y, type=DataType.DT_UINT8)
        # GE 原型顺序为 y/amax/y_scale，Torch 对外顺序为 y/y_scale/amax。
        return y, y_scale, amax
