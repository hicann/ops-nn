# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for TransposeBatchMatMul (TorchAir graph mode)
#
# x1/x2 are FLOAT16/BFLOAT16/FLOAT32 3D tensors. perm_x1/perm_x2 select the batch axis and the
# matmul axes; perm_y is fixed to [1, 0, 2]. The optional scale turns on FLOAT16 -> INT8
# quantization (output shape becomes (M, 1, B * N)); without scale the output dtype follows x1.

from ._perm_defaults import _DEFAULT_PERM_X1, _DEFAULT_PERM_X2, _DEFAULT_PERM_Y

try:
    from typing import List, Optional

    import torch
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:
    _TBMM_IR = (
        IrDef("TransposeBatchMatMul")
        .input("x1", "DT_FLOAT, DT_FLOAT16, DT_BF16")
        .input("x2", "DT_FLOAT, DT_FLOAT16, DT_BF16")
        .optional_input("bias", "DT_FLOAT, DT_FLOAT16, DT_BF16")
        .optional_input("scale", "DT_INT64, DT_UINT64")
        .attr("perm_x1", attr.ListInt([]))
        .attr("perm_x2", attr.ListInt([]))
        .attr("perm_y", attr.ListInt([]))
        .attr("enable_hf32", attr.Bool(False))
        .attr("batch_split_factor", attr.Int(1))
        .output("y", "DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8")
    )

    def _resolve_perm(perm, default):
        return [int(value) for value in perm] if perm is not None else list(default)

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.transpose_batch_mat_mul.default
    )
    def convert_transpose_batch_mat_mul(
        x1: Tensor,
        x2: Tensor,
        *,
        bias: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        perm_x1: Optional[List[int]] = None,
        perm_x2: Optional[List[int]] = None,
        perm_y: Optional[List[int]] = None,
        enable_hf32: bool = False,
        batch_split_factor: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        perm_x1_real = _resolve_perm(perm_x1, _DEFAULT_PERM_X1)
        perm_x2_real = _resolve_perm(perm_x2, _DEFAULT_PERM_X2)
        perm_y_real = _resolve_perm(perm_y, _DEFAULT_PERM_Y)
        batch_split = 1 if batch_split_factor is None else int(batch_split_factor)

        if perm_x1_real not in (_DEFAULT_PERM_X1, [0, 1, 2]):
            raise RuntimeError(
                "perm_x1 only supports [0, 1, 2] and [1, 0, 2], but got %s"
                % perm_x1_real
            )
        if perm_x2_real not in (_DEFAULT_PERM_X2, [0, 2, 1]):
            raise RuntimeError(
                "perm_x2 only supports [0, 1, 2] and [0, 2, 1], but got %s"
                % perm_x2_real
            )
        if perm_y_real != _DEFAULT_PERM_Y:
            raise RuntimeError(
                "perm_y only supports [1, 0, 2], but got %s" % perm_y_real
            )
        if batch_split <= 0:
            raise RuntimeError(
                "batch_split_factor must be positive, but got %s" % batch_split
            )
        if scale is not None and batch_split != 1:
            raise RuntimeError("batch_split_factor must be 1 when scale is provided")

        inputs = {"x1": x1, "x2": x2}
        if bias is not None:
            inputs["bias"] = bias
        if scale is not None:
            inputs["scale"] = scale

        y = ge_op(
            op_type="TransposeBatchMatMul",
            inputs=inputs,
            attrs={
                "perm_x1": attr.ListInt(perm_x1_real),
                "perm_x2": attr.ListInt(perm_x2_real),
                "perm_y": attr.ListInt(perm_y_real),
                "enable_hf32": attr.Bool(bool(enable_hf32)),
                "batch_split_factor": attr.Int(batch_split),
            },
            outputs=["y"],
            ir=_TBMM_IR,
        )

        if scale is not None:
            y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_INT8)
        else:
            y.desc.dtype = x1.desc.dtype
        return y

else:

    def convert_transpose_batch_mat_mul(*args, **kwargs):
        raise RuntimeError(
            "TransposeBatchMatMul graph converter: torchair is not available."
        )
