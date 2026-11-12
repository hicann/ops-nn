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

"""InplaceAddRmsNorm golden：Add + RmsNorm 融合算子参考实现。

kernel / GEIR = ``inplace_add_rms_norm``（numpy 入/出）：残差和 x = x1 + x2、
rstd = rsqrt(mean(x^2, reduce) + eps)、y = x * rstd * gamma，输出
[y, rstd, x]（x1/y 与 x2/x 为原位输出，与 add_rms_norm 语义一致）。

aclnn = ``aclnnInplaceAddRmsNorm``（torch 入/出）：输出 [y, x, rstd]，
x1Ref / x2Ref 为原位输入输出，rstdOut 为输出占位参数。

e2e = ``torch_npu.npu_add_rms_norm_v2``（torch 入/出）：返回值 rstd，
x1 / x2 原位改写；参考输出顺序 [rstd, y, x]。

芯片差异：A2/A3（Ascend910B / Ascend910_93）的残差和会先按输入 dtype
保存中间结果，950 系列则全程 FP32 归约，分别走 v1 / v2 两条参考路径
（与 add_rms_norm 一致）。
"""

import numpy as np
import torch

__spec__ = {
    "inplace_add_rms_norm": "InplaceAddRmsNormSpec",  # Kernel / GEIR
    "aclnnInplaceAddRmsNorm": "AclnnInplaceAddRmsNormSpec",  # ACLNN
    "torch_npu.npu_add_rms_norm_v2": "TorchInplaceAddRmsNormSpec",  # E2E
}


def _np_to_torch(array):
    """numpy → torch CPU 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.float16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy(tensor, np_dtype):
    """torch CPU 张量 → numpy，按输出 dtype 转换（bf16 经 uint16 位视图）。"""
    np_dtype = np.dtype(np_dtype)
    if np_dtype.name == "bfloat16":
        return tensor.to(torch.bfloat16).view(torch.float16).numpy().view(np_dtype)
    return np.ascontiguousarray(tensor.numpy().astype(np_dtype, copy=False))


def _is_910b_or_93(short_soc_version):
    """A2/A3（Ascend910B / Ascend910_93）判定：归一化后按前缀匹配。

    框架传入的 short_soc_version 大小写、下划线与前后缀并不统一，统一归一化为
    小写并去掉下划线后按前缀比对，避免同名 kwarg 在不同入口解析规则不一致。
    """
    soc = str(short_soc_version or "").lower().replace("_", "")
    return soc.startswith(("ascend910b", "ascend91093"))


def _rstd_of(x, gamma, eps):
    """按 gamma 覆盖的末尾维规约，求 rstd = rsqrt(mean(x^2) + eps)。"""
    reduce_ndim = len(gamma.shape)
    prefix = x.shape[: len(x.shape) - reduce_ndim]
    if 0 in prefix:
        x_flat = x.reshape((*prefix, 0))
    else:
        x_flat = x.reshape((*prefix, -1))
    return torch.rsqrt(
        x_flat.pow(2).mean(-1, keepdim=True).reshape((*prefix, *([1] * reduce_ndim)))
        + eps
    )


def add_rms_v1(x1, x2, gamma, eps):
    """A2/A3（910B/910_93）参考路径：残差和先按输入 dtype 保存，再抬 FP32 归约。

    返回三元组 (y, rstd, x)：y 为归一化输出，rstd 为 FP32，x 为残差和。
    """
    if x1.dtype == torch.bfloat16:
        x = (x1.type(torch.float32) + x2.type(torch.float32)).type(x1.dtype)
    else:
        x = x1 + x2

    x_fp32 = x.type(torch.float32)
    rstd = _rstd_of(x_fp32, gamma, eps)

    tmp_x = x_fp32 * rstd
    if x1.dtype == torch.bfloat16:
        tmp_x = tmp_x.type(torch.bfloat16).type(torch.float32)
        y = (tmp_x * gamma.type(torch.float32)).type(x1.dtype)
    elif x.dtype == torch.float16:
        tmp_x = tmp_x.type(torch.float16)
        y = tmp_x * gamma
    else:
        y = tmp_x * gamma
    return y, rstd, x


def add_rms_v2(x1, x2, gamma, eps):
    """950 系列参考路径：残差和不提前转回低精度，全程 FP32（或 FP64）归约。

    返回三元组 (y, rstd, x)：y 为归一化输出，rstd 为 FP32，x 为残差和。
    """
    calc_dtype = (
        torch.float64
        if torch.float64 in (x1.dtype, x2.dtype, gamma.dtype)
        else torch.float32
    )
    x = x1.type(calc_dtype) + x2.type(calc_dtype)
    rstd = _rstd_of(x, gamma, eps)

    y_fp32 = x * rstd * gamma.type(calc_dtype)
    y = y_fp32.type(x1.dtype)
    x = x.type(x1.dtype)
    return y, rstd, x


def _dispatch(x1, x2, gamma, eps, short_soc_version):
    """按 SoC 选择 v1 / v2 参考路径。"""
    if _is_910b_or_93(short_soc_version):
        return add_rms_v1(x1, x2, gamma, eps)
    return add_rms_v2(x1, x2, gamma, eps)


def inplace_add_rms_norm_golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
    """Kernel/GEIR golden：numpy 入/出，返回 [y, rstd, x]。"""
    x1_t = _np_to_torch(x1)
    x2_t = _np_to_torch(x2)
    gamma_t = _np_to_torch(gamma)

    y, rstd, x = _dispatch(
        x1_t, x2_t, gamma_t, epsilon, kwargs.get("short_soc_version")
    )

    return [
        _torch_to_numpy(y, x1.dtype),
        rstd.numpy(),
        _torch_to_numpy(x, x1.dtype),
    ]


def aclnn_inplace_add_rms_norm_golden(
    x1Ref, x2Ref, gamma, epsilon=1e-6, rstdOut=None, **kwargs
):
    """ACLNN golden：torch 入/出，返回 [y, x, rstd]。

    参数位置跟随 aclnnInplaceAddRmsNormGetWorkspaceSize（不含 workspaceSize 与
    executor）；rstdOut 为输出占位参数，不参与计算。输出顺序与 aclnn 用例
    output_tensor_indexes=(x1Ref, x2Ref, rstdOut) 一致。
    """
    y, rstd, x = _dispatch(
        x1Ref, x2Ref, gamma, epsilon, kwargs.get("short_soc_version")
    )
    return [y, x, rstd]


def torch_inplace_add_rms_norm_golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
    """E2E golden：torch 入/出，返回 [rstd, y, x]。

    torch_npu.npu_add_rms_norm_v2 返回 rstd，x1 / x2 原位改写；参考输出顺序
    与 npu_add_rms_norm_v2_functional 一致（rstd, x1, x2）。
    """
    y, rstd, x = _dispatch(x1, x2, gamma, epsilon, kwargs.get("short_soc_version"))
    return [rstd, y, x]


class InplaceAddRmsNormXpuRef:
    """torch 三方标杆（GPU 参考，用于 cross_check）。

    输入为设备侧 torch.Tensor：按 950 语义全程 FP32 残差求和；子类按各自
    输出顺序重排 [y, rstd, x] 三元组。
    """

    def _compute(self, x1, x2, gamma, epsilon):
        calc_dtype = (
            torch.float64
            if torch.float64 in (x1.dtype, x2.dtype, gamma.dtype)
            else torch.float32
        )
        x = torch.add(x1.to(calc_dtype), x2.to(calc_dtype))
        rstd = _rstd_of(x, gamma, epsilon)
        y = (x * rstd * gamma.to(calc_dtype)).to(x1.dtype)
        return y, rstd, x.to(x1.dtype)

    def __call__(self, x1, x2, gamma, epsilon=1e-6, **kwargs):
        y, rstd, x = self._compute(x1, x2, gamma, epsilon)
        return [y, rstd, x]


class AclnnInplaceAddRmsNormXpuRef(InplaceAddRmsNormXpuRef):
    """torch 三方标杆（aclnn 形态，输出顺序 [y, x, rstd]）。"""

    def __call__(self, x1, x2, gamma, epsilon=1e-6, **kwargs):
        y, rstd, x = self._compute(x1, x2, gamma, epsilon)
        return [y, x, rstd]


class TorchInplaceAddRmsNormXpuRef(InplaceAddRmsNormXpuRef):
    """torch 三方标杆（e2e 形态，输出顺序 [rstd, y, x]）。"""

    def __call__(self, x1, x2, gamma, epsilon=1e-6, **kwargs):
        y, rstd, x = self._compute(x1, x2, gamma, epsilon)
        return [rstd, y, x]


class InplaceAddRmsNormSpec:
    """kernel/GEIR 流程：numpy 入/出，3 输出 [y, rstd, x]。"""

    golden = staticmethod(inplace_add_rms_norm_golden)
    third_party = {"torch": InplaceAddRmsNormXpuRef}


class AclnnInplaceAddRmsNormSpec(InplaceAddRmsNormSpec):
    """aclnn 流程：torch 入/出，3 输出 [y, x, rstd]。"""

    golden = staticmethod(aclnn_inplace_add_rms_norm_golden)
    third_party = {"torch": AclnnInplaceAddRmsNormXpuRef}


class TorchInplaceAddRmsNormSpec(InplaceAddRmsNormSpec):
    """e2e 流程：torch 入/出，3 输出 [rstd, y, x]。"""

    golden = staticmethod(torch_inplace_add_rms_norm_golden)
    third_party = {"torch": TorchInplaceAddRmsNormXpuRef}
