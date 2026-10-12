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
"""TTK TestSpec for global_lp_pool (kernel / GEIR 通路, arch35/Ascend950).

y = (sum_spatial |x|^p)^(1/p), keepdims=True.
输入 x: 4D(NCHW) / 5D(NCD0D1D2), dtype ∈ {float32, float16, bfloat16};
属性 p: float, 支持任意非零实数（p>0 为主契约，p<0 按同一公式自然延拓）; 输出 dtype 与 x 相同。

注册说明：
    __spec__  —— TTK TestSpec（优先加载）。GEIR 通路复用 kernel 通路同一注册键
                 （op_name = global_lp_pool），不单独注册。
    __golden__ —— 兼容旧插件的 kernel 入口，与 __spec__ 共用同一实现。

数值口径：kernel 内部把 fp16/bf16 抬到 fp32 计算后再舍回原 dtype，golden 采用
相同口径（fp32 计算 + 舍回 x.dtype），不额外抬 fp64。
"""

import numpy as np

__spec__ = {"global_lp_pool": "GlobalLpPoolKernelSpec"}


def global_lp_pool_golden(x, **kwargs):
    """y = (sum(|x|^p) over spatial dims)^(1/p), keepdims=True.

    Parameters follow global_lp_pool_def.cpp without outputs (x, attr p).
    Returns [y] (one ndarray per output).
    """
    p = float(kwargs.get("p", 2.0))
    if p == 0.0:
        # p=0 无定义：1/p 发散，且左右极限分别为 0 / +inf（极点）。
        # 注：-0.0 == 0.0，同样落在此分支（与 kernel 的 `p != 0.0f` 判断一致）。
        raise ValueError("global_lp_pool golden requires p != 0, got %r" % (p,))

    # 空归约（R = prod(spatial) == 0）：按算子约定输出恒等元 0（kernel EMPTY_R 路径
    # 直接写 0.0f，见 global_lp_pool_kernel.h isEmptyTensor 分支）。
    # 注：对 p<0，公式的自然延拓为 0^(1/p) = +inf，与 kernel 的 0 不一致；
    #     此处按算子既有约定编码，该语义缺口单独记录（不在此 golden 内裁定）。
    spatial_size = 1
    for d in x.shape[2:]:
        spatial_size *= d
    if spatial_size == 0:
        out_shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        return [np.zeros(out_shape, dtype=x.dtype)]

    # fp16/bf16 抬 fp32 计算（与 kernel 内部计算域一致）
    x_f32 = x.astype(np.float32)

    # Spatial axes: dims[2:]
    spatial_axes = tuple(range(2, x_f32.ndim))

    # p<0 时含 0 输入会 |x|^p -> inf、Σ -> inf、(inf)^(1/p) -> 0（自然延拓），
    # 与 kernel 的 exp/ln 链一致；抑制 numpy 的 divide/over 警告。
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # |x|^p
        powered = np.power(np.abs(x_f32), p)

        # Sum along spatial dims
        summed = np.sum(powered, axis=spatial_axes, keepdims=True)

        # (sum)^(1/p)
        result = np.power(summed, 1.0 / p)

    # 输出与输入同 dtype
    return [result.astype(x.dtype, copy=False)]


class GlobalLpPoolKernelSpec:
    """kernel / GEIR 共用 spec: golden 收 numpy.ndarray, 返 list[np.ndarray]."""

    def golden(x, **kwargs):
        return global_lp_pool_golden(x, **kwargs)


__golden__ = {"kernel": {"global_lp_pool": "global_lp_pool_golden"}}
