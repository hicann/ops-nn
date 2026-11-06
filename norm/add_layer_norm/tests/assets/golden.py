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


__golden__ = {"kernel": {"add_layer_norm": "add_layer_norm_golden"}}

__spec__ = {"add_layer_norm": "AddLayerNormTestSpec"}


def add_layer_norm_golden(
    x1,
    x2,
    gamma,
    beta,
    bias=None,  # inputs
    epsilon: float = 1e-5,
    additional_output: bool = False,  # attributes
    **kwargs,
):
    """
    Golden function for add_layer_norm.
    All the parameters (names and order) follow @add_layer_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    # FP32/FP64 输入用 FP64 累积，兼容框架升精度输入。
    wide = x1.dtype.itemsize >= 4 or x2.dtype.itemsize >= 4
    calc_dtype = np.float64 if wide else np.float32
    out_dtype = (
        np.float64
        if x1.dtype == np.float64 or x2.dtype == np.float64
        else np.float32
        if wide
        else x1.dtype
    )
    normalized_shape = tuple(gamma.shape)
    stat_shape = x1.shape[: x1.ndim - gamma.ndim] + (1,) * gamma.ndim

    x = x1.astype(calc_dtype) + x2.astype(calc_dtype)
    if bias is not None:
        x = x + bias.astype(calc_dtype)

    if 0 in normalized_shape:
        # 950 空归一化维度：mean=0，rstd=NaN。
        y_out = np.empty(x1.shape, dtype=out_dtype)
        mean_out = np.zeros(stat_shape, dtype=np.float32)
        rstd_out = np.full(stat_shape, np.nan, dtype=np.float32)
    else:
        y, mean, rstd = torch.ops.aten.native_layer_norm(
            torch.from_numpy(x),
            list(normalized_shape),
            weight=torch.from_numpy(gamma.astype(calc_dtype)),
            bias=torch.from_numpy(beta.astype(calc_dtype)),
            eps=float(calc_dtype(epsilon)),
        )
        y_out = y.numpy().astype(out_dtype)
        mean_out = mean.numpy().reshape(stat_shape).astype(np.float32)
        rstd_out = rstd.numpy().reshape(stat_shape).astype(np.float32)

    # 未启用的 x 输出用 None 跳过比较。
    x_out = x.astype(out_dtype) if additional_output else None
    return y_out, mean_out, rstd_out, x_out


class AddLayerNormTorchThirdParty:
    """Torch 三方对照，低精度输入用 FP32 累积。"""

    def __call__(
        self,
        x1,
        x2,
        gamma,
        beta,
        bias=None,
        epsilon=1e-5,
        additional_output=False,
        **kwargs,
    ):
        import torch

        out_dtype = torch.promote_types(x1.dtype, x2.dtype)
        calc_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
        x = x1.to(calc_dtype) + x2.to(calc_dtype)
        if bias is not None:
            x = x + bias.to(calc_dtype)
        y, mean, rstd = torch.ops.aten.native_layer_norm(
            x,
            list(gamma.shape),
            weight=gamma.to(calc_dtype),
            bias=beta.to(calc_dtype),
            eps=float(epsilon),
        )
        # 保留输出槽位，关闭 x 输出时跳过比较。
        return (
            y.to(out_dtype),
            mean.to(torch.float32),
            rstd.to(torch.float32),
            x.to(out_dtype) if additional_output else None,
        )


class AddLayerNormTestSpec:
    golden = staticmethod(add_layer_norm_golden)
    third_party = {"torch": AddLayerNormTorchThirdParty}
