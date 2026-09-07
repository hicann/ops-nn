#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
from copy import deepcopy

__golden__ = {
    "aclnn": {
        "aclnnInplaceQuantScatterV2": "aclnn_inplace_quant_scatter_v2_golden",
        "aclnnInplaceQuantScatter": "aclnn_inplace_quant_scatter_golden",
    },
    "kernel": {"quant_update_scatter": "quant_update_scatter_golden"},
}


def quant_update_scatter_golden(
    var,
    indices,
    updates,
    quant_scales,
    quant_zero_points=None,
    *,
    reduce,
    axis=-2,
    quant_axis=-1,
    reciprocal_scale=False,
    round_mode="rint",
    **kwargs,
):
    """
    Golden function for quant_update_scatter.
    All the parameters (names and order) follow @quant_update_scatter_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    update_value = updates.astype(np.float32)
    dtype = var.dtype

    all_shape = len(var.shape)
    if axis < 0:
        axis = all_shape + axis

    # quantize
    quant_scales_value = quant_scales.reshape(-1).astype(np.float32)
    quant_zero_points_value = quant_zero_points
    if quant_zero_points_value is not None:
        quant_zero_points_value = quant_zero_points_value.reshape(-1).astype(np.float32)

    if reciprocal_scale:
        update_value = update_value * quant_scales_value
    else:
        update_value = update_value / quant_scales_value
    if quant_zero_points_value is not None:
        update_value = update_value + quant_zero_points_value

    if dtype == "int8":
        update_value = np.round(update_value, 0)
        update_value = np.clip(update_value, -128, 127)
        update_value = update_value.astype(dtype)
    elif dtype == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        update_value = update_value.astype(float8_e5m2, copy=False)
    elif dtype == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        update_value = update_value.astype(float8_e4m3fn, copy=False)
    elif dtype == "hifloat8":
        from en_dtypes import hifloat8

        update_value = update_value.astype(hifloat8, copy=False)

    # merge axis:[dim0, reduce(lambda x, y: x*y, [dim1:dim[axis]), dim[axis], dim[-1])]
    if axis != 0 and axis != all_shape - 1:
        trans_shape_0 = var.shape[0]
        update_shape_0 = update_value.shape[0]

        seceond_dim = 1
        update_second_dim = 1

        for i in range(1, axis):
            seceond_dim *= var.shape[i]
            update_second_dim *= update_value.shape[i]
        trans_shape_1 = seceond_dim
        update_shape_1 = update_second_dim

        trans_shape_2 = var.shape[axis]
        update_shape_2 = update_value.shape[axis]

        fourth_dim = 1
        update_fourth_dim = 1
        for i in range(axis + 1, all_shape):
            fourth_dim *= var.shape[i]
            update_fourth_dim *= update_value.shape[i]
        trans_shape_3 = fourth_dim
        update_shape_3 = update_fourth_dim

        var = var.reshape(trans_shape_0, trans_shape_1, trans_shape_2, trans_shape_3)
        update_value = update_value.reshape(
            update_shape_0, update_shape_1, update_shape_2, update_shape_3
        )

    if 0 < axis < all_shape - 1:
        axis = 2
    else:
        axis = axis

    shape_0 = update_value.shape[0]
    shape_1 = update_value.shape[1]
    shape_2 = update_value.shape[2]
    shape_3 = update_value.shape[3]

    output = deepcopy(var)

    indices_value = indices.astype(np.int64)

    # scatter
    if len(indices.shape) == 2:
        if axis == -2 or axis == 2:
            for i in range(indices.shape[0]):
                for j in range(shape_1):
                    for k in range(shape_2):
                        for m in range(shape_3):
                            output[indices_value[i][0]][j][indices_value[i][1] + k][
                                m
                            ] = update_value[i][j][k][m]

        elif axis == -1 or axis == 3:
            for i in range(indices.shape[0]):
                for j in range(shape_1):
                    for k in range(shape_2):
                        for m in range(shape_3):
                            output[indices_value[i][0]][j][k][
                                indices_value[i][1] + m
                            ] = update_value[i][j][k][m]
    else:
        if axis == -2 or axis == 2:
            for i in range(shape_0):
                indices_key = indices_value[i]
                for j in range(shape_1):
                    for k in range(shape_2):
                        for m in range(shape_3):
                            output[i][j][indices_key + k][m] = update_value[i][j][k][m]
        elif axis == -1 or axis == 3:
            for i in range(shape_0):
                indices_key = indices_value[i]
                for j in range(shape_1):
                    for k in range(shape_2):
                        for m in range(shape_3):
                            output[i][j][k][indices_key + m] = update_value[i][j][k][m]

    return [output]


def aclnn_inplace_quant_scatter_golden(
    selfRef,
    indices,
    updates,
    quantScales,
    quantZeroPoints,
    axis,
    quantAxis,
    reduction,
    **kwargs,
):
    """
    Aclnn golden for aclnnInplaceQuantScatter.
    Parameters follow @aclnnInplaceQuantScatterGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import numpy as np
    from copy import deepcopy

    if hasattr(axis, "item"):
        axis = axis.item()
    if hasattr(quantAxis, "item"):
        quantAxis = quantAxis.item()
    if hasattr(reduction, "item"):
        reduction = reduction.item()

    def _to_np(t):
        if t is None:
            return None
        if isinstance(t, np.ndarray):
            return t
        dt = t.dtype
        dt_str = str(dt)
        if dt == torch.bfloat16:
            return t.to(torch.float32).numpy()
        if "float8_e5m2" in dt_str:
            from ml_dtypes import float8_e5m2 as np_f8_e5m2

            return t.view(torch.uint8).numpy().view(np_f8_e5m2)
        if "float8_e4m3fn" in dt_str:
            from ml_dtypes import float8_e4m3fn as np_f8_e4m3

            return t.view(torch.uint8).numpy().view(np_f8_e4m3)
        if "hifloat8" in dt_str:
            from en_dtypes import hifloat8 as np_hf8

            return t.view(torch.uint8).numpy().view(np_hf8)
        return t.numpy()

    var_np = _to_np(selfRef)
    indices_np = _to_np(indices)
    updates_np = _to_np(updates)
    scales_np = _to_np(quantScales)
    zp_np = _to_np(quantZeroPoints)

    result = quant_update_scatter_golden(
        deepcopy(var_np),
        indices_np,
        updates_np,
        scales_np,
        zp_np,
        reduce=reduction,
        axis=axis,
        quant_axis=quantAxis,
    )

    if isinstance(result, list):
        result = result[0]
    dt_str = str(selfRef.dtype)
    if "float8" in dt_str or "hifloat8" in dt_str:
        if isinstance(selfRef, np.ndarray):
            return [result]
        return [
            torch.from_numpy(result.copy().view(np.uint8).copy()).view(selfRef.dtype)
        ]
    if isinstance(selfRef, np.ndarray):
        return [result]
    return [torch.from_numpy(np.ascontiguousarray(result.copy())).to(selfRef.dtype)]


def aclnn_inplace_quant_scatter_v2_golden(
    selfRef,
    indices,
    updates,
    quantScales,
    quantZeroPoints,
    axis,
    quantAxis,
    reduction,
    roundMode,
    **kwargs,
):
    """
    Aclnn golden for aclnnInplaceQuantScatterV2.
    Parameters follow @aclnnInplaceQuantScatterV2GetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import numpy as np
    from copy import deepcopy

    if hasattr(axis, "item"):
        axis = axis.item()
    if hasattr(quantAxis, "item"):
        quantAxis = quantAxis.item()
    if hasattr(reduction, "item"):
        reduction = reduction.item()
    if isinstance(roundMode, bytes):
        roundMode = roundMode.decode()
    elif hasattr(roundMode, "item"):
        roundMode = roundMode.item()

    def _to_np(t):
        if t is None:
            return None
        if isinstance(t, np.ndarray):
            return t
        dt = t.dtype
        dt_str = str(dt)
        if dt == torch.bfloat16:
            return t.to(torch.float32).numpy()
        if "float8_e5m2" in dt_str:
            from ml_dtypes import float8_e5m2 as np_f8_e5m2

            return t.view(torch.uint8).numpy().view(np_f8_e5m2)
        if "float8_e4m3fn" in dt_str:
            from ml_dtypes import float8_e4m3fn as np_f8_e4m3

            return t.view(torch.uint8).numpy().view(np_f8_e4m3)
        if "hifloat8" in dt_str:
            from en_dtypes import hifloat8 as np_hf8

            return t.view(torch.uint8).numpy().view(np_hf8)
        return t.numpy()

    var_np = _to_np(selfRef)
    indices_np = _to_np(indices)
    updates_np = _to_np(updates)
    scales_np = _to_np(quantScales)
    zp_np = _to_np(quantZeroPoints)

    result = quant_update_scatter_golden(
        deepcopy(var_np),
        indices_np,
        updates_np,
        scales_np,
        zp_np,
        reduce=reduction,
        axis=axis,
        quant_axis=quantAxis,
        round_mode=roundMode if isinstance(roundMode, str) else "rint",
    )

    if isinstance(result, list):
        result = result[0]
    dt_str = str(selfRef.dtype)
    if "float8" in dt_str or "hifloat8" in dt_str:
        if isinstance(selfRef, np.ndarray):
            return [result]
        return [
            torch.from_numpy(result.copy().view(np.uint8).copy()).view(selfRef.dtype)
        ]
    if isinstance(selfRef, np.ndarray):
        return [result]
    return [torch.from_numpy(np.ascontiguousarray(result.copy())).to(selfRef.dtype)]
