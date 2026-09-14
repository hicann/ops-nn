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

__golden__ = {"kernel": {"inplace_add_layer_norm": "inplace_add_layer_norm_golden"}}

__spec__ = {"inplace_add_layer_norm": "InplaceAddLayerNormTestSpec"}


def inplace_add_layer_norm_golden(
    x1,
    x2,
    gamma,
    beta,
    bias=None,
    epsilon: float = 1e-5,
    additional_output: bool = False,
    **kwargs,
):
    import torch

    # 计算精度：fp32/fp64 输入用 float64，fp16/bf16 用 float32。
    wide = x1.dtype.itemsize >= 4
    calc_dtype = np.float64 if wide else np.float32
    calc_dtype_torch = torch.float64 if wide else torch.float32

    yx_dtype = x1.dtype
    normalized_shape = tuple(gamma.shape)
    norm_rank = len(normalized_shape)
    stat_shape = tuple(x1.shape[: x1.ndim - norm_rank]) + (1,) * norm_rank

    if bias is not None:
        x = x2.astype(calc_dtype) + bias.astype(calc_dtype) + x1.astype(calc_dtype)
    else:
        x = x1.astype(calc_dtype) + x2.astype(calc_dtype)

    if 0 in normalized_shape:
        # 950平台支持空tensor，mean/rstd 填 NaN、y/x 为空不写；
        y_out = np.zeros(x1.shape, dtype=yx_dtype)
        mean_out = np.full(stat_shape, np.nan).astype("float32")
        rstd_out = np.full(stat_shape, np.nan).astype("float32")
    else:
        eps = np.float64(epsilon) if wide else np.float32(epsilon)
        y, mean, rstd = torch.ops.aten.native_layer_norm(
            torch.from_numpy(x).to(calc_dtype_torch),
            list(normalized_shape),
            weight=torch.from_numpy(gamma.astype(calc_dtype)),
            bias=torch.from_numpy(beta.astype(calc_dtype)),
            eps=eps,
        )
        y_out = y.numpy().astype(yx_dtype)
        mean_out = mean.numpy().reshape(stat_shape).astype("float32")
        rstd_out = rstd.numpy().reshape(stat_shape).astype("float32")

    if not additional_output:
        return y_out, mean_out, rstd_out
    return y_out, mean_out, rstd_out, x.astype(yx_dtype)


class TorchGpuThirdParty:
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

        out_dtype = x1.dtype
        calc = torch.float32
        if bias is not None:
            x = x2.to(calc) + bias.to(calc) + x1.to(calc)
        else:
            x = x1.to(calc) + x2.to(calc)
        normalized_shape = list(gamma.shape)
        y, mean, rstd = torch.ops.aten.native_layer_norm(
            x,
            normalized_shape,
            weight=gamma.to(calc),
            bias=beta.to(calc),
            eps=float(epsilon),
        )

        stat_shape = list(x1.shape[: x1.dim() - len(normalized_shape)]) + [1] * len(
            normalized_shape
        )
        x_out = x.to(out_dtype) if additional_output else x2.clone().to(out_dtype)
        return (
            y.to(out_dtype),
            mean.reshape(stat_shape).to(torch.float32),
            rstd.reshape(stat_shape).to(torch.float32),
            x_out,
        )


class InplaceAddLayerNormTestSpec:
    golden = staticmethod(inplace_add_layer_norm_golden)
    third_party = {"torch": TorchGpuThirdParty}
