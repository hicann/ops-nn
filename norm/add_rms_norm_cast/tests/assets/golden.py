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

"""AddRmsNormCast golden：AddRmsNorm 与 Cast 的融合算子参考实现。

kernel / GEIR = ``add_rms_norm_cast``（numpy 入/出）：真值在 FP32 中完成
x = float(x1) + float(x2)、rstd = rsqrt(mean(x^2)+eps)、y1 = x*rstd*gamma，
之后 y2 = cast(y1)、x = cast(x)，返回 [y1, y2, rstd, x]；
aclnn = ``aclnnAddRmsNormCast``（torch 入/出）；
e2e = ``torch_npu.npu_add_rms_norm_cast``（torch 入/出）。

A2/A3（Ascend910B / Ascend910_93）与 950 系列在残差和的中间舍入上存在差异，
分别走 ``add_rms_cast_v1`` / ``add_rms_cast_v2`` 两条参考路径。
"""

import numpy as np
import torch

__spec__ = {
    "add_rms_norm_cast": "AddRmsNormCastSpec",  # Kernel / GEIR
    "aclnnAddRmsNormCast": "AclnnAddRmsNormCastSpec",
    "torch_npu.npu_add_rms_norm_cast": "TorchAddRmsNormCastSpec",
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
        return tensor.view(torch.float16).numpy().view(np_dtype)
    return np.ascontiguousarray(tensor.numpy().astype(np_dtype, copy=False))


def _is_910b_or_93(short_soc_version):
    """A2/A3（Ascend910B / Ascend910_93）判定：归一化后按前缀匹配。

    框架传入的 short_soc_version 大小写、下划线与前后缀并不统一（如
    Ascend910B、ascend910b、Ascend910_93、Ascend910B4），统一归一化为
    小写并去掉下划线后按前缀比对，避免同名 kwarg 在不同入口解析规则不一致。
    """
    soc = str(short_soc_version or "").lower().replace("_", "")
    return soc.startswith(("ascend910b", "ascend91093"))


def add_rms_cast_v1(x1, x2, gamma, eps):
    """A2/A3（910B/910_93）参考路径：残差和先按输入 dtype 保存，再抬 FP32 归约。

    返回四元组 (y1, y2, rstd, x)：y1 为 FP32 归一化输出，y2 为其 cast 结果。
    """
    if x1.dtype == torch.bfloat16:
        x = (x1.type(torch.float32) + x2.type(torch.float32)).type(x1.dtype)
    else:
        x = x1 + x2

    xFp32 = x.type(torch.float32)
    if 0 in xFp32.shape[: len(xFp32.shape) - len(gamma.shape)]:
        xFp32_1 = xFp32.reshape(
            (*xFp32.shape[: len(xFp32.shape) - len(gamma.shape)], 0)
        )
    else:
        xFp32_1 = xFp32.reshape(
            (*xFp32.shape[: len(xFp32.shape) - len(gamma.shape)], -1)
        )
    rstd = torch.rsqrt(
        xFp32_1.pow(2)
        .mean(-1, keepdim=True)
        .reshape(
            (
                *xFp32.shape[: len(xFp32.shape) - len(gamma.shape)],
                *([1] * len(gamma.shape)),
            )
        )
        + eps
    )

    tmpX = xFp32 * rstd
    if x1.dtype == torch.bfloat16:
        tmpX = tmpX.type(torch.bfloat16).type(torch.float32)
        y2 = (tmpX * gamma.type(torch.float32)).type(x1.dtype)
    elif x.dtype == torch.float16:
        tmpX = tmpX.type(torch.float16)
        y2 = tmpX * gamma
    else:
        y2 = tmpX * gamma
    y1 = y2.type(torch.float32).clone()
    return y1, y2, rstd, x


def add_rms_cast_v2(x1, x2, gamma, eps):
    """950 系列参考路径：残差和不提前转回低精度，全程 FP32 归约。

    返回四元组 (y1, y2, rstd, x)：y1 为 FP32 归一化输出，y2 为其 cast 结果。
    """
    calc_dtype = torch.float32
    x = x1.type(calc_dtype) + x2.type(calc_dtype)

    if 0 in x.shape[: len(x.shape) - len(gamma.shape)]:
        x_1 = x.reshape((*x.shape[: len(x.shape) - len(gamma.shape)], 0))
    else:
        x_1 = x.reshape((*x.shape[: len(x.shape) - len(gamma.shape)], -1))
    rstd = torch.rsqrt(
        x_1.pow(2)
        .mean(-1, keepdim=True)
        .reshape(
            (*x.shape[: len(x.shape) - len(gamma.shape)], *([1] * len(gamma.shape)))
        )
        + eps
    )

    tmpX = x * rstd
    y1 = tmpX * gamma.type(calc_dtype)
    # clone 防止 y1 与 y2 共享存储：Promote 模式下 x1.dtype 已抬到 fp32，
    # y1.type(x1.dtype) 会原样返回 y1，导致框架对 y2 施加 fp16 溢出语义时
    # 连带污染 y1（真值）。显式 clone 隔离两个输出。
    y2 = y1.type(x1.dtype).clone()
    return y1, y2, rstd, x.type(x1.dtype).clone()


def add_rms_norm_cast_golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
    """Kernel/GEIR golden：numpy 入/出，返回 [y1, y2, rstd, x]。"""
    x1_t = _np_to_torch(x1)
    x2_t = _np_to_torch(x2)
    gamma_t = _np_to_torch(gamma)

    if _is_910b_or_93(kwargs.get("short_soc_version")):
        y1, y2, rstd, x = add_rms_cast_v1(x1_t, x2_t, gamma_t, epsilon)
    else:
        y1, y2, rstd, x = add_rms_cast_v2(x1_t, x2_t, gamma_t, epsilon)

    return [
        y1.numpy(),
        _torch_to_numpy(y2, x1.dtype),
        rstd.numpy(),
        _torch_to_numpy(x, x1.dtype),
    ]


def aclnn_add_rms_norm_cast_golden(
    x1,
    x2,
    gamma,
    epsilon=1e-6,
    y1Out=None,
    y2Out=None,
    rstdOut=None,
    xOut=None,
    **kwargs,
):
    """ACLNN / E2E golden：torch 入/出，返回 [y1, y2, rstd, x]。

    参数位置跟随 aclnnAddRmsNormCastGetWorkspaceSize（不含 workspaceSize 与
    executor）；可选的输出占位参数不参与计算。E2E 的 torch_npu 接口
    npu_add_rms_norm_cast(x1, x2, gamma, epsilon) 复用本入口。
    """
    if _is_910b_or_93(kwargs.get("short_soc_version")):
        y1, y2, rstd, x = add_rms_cast_v1(x1, x2, gamma, epsilon)
    else:
        y1, y2, rstd, x = add_rms_cast_v2(x1, x2, gamma, epsilon)
    return [y1, y2, rstd, x]


class AddRmsNormCastXpuRef:
    """torch 三方标杆（GPU 参考，用于 cross_check）。

    输入为设备侧 torch.Tensor：FP32 残差求和，torch.rms_norm 生成 FP32 归一化
    输出 y1，再 cast 得到 y2；rstd 与残差和 x 单独计算并在输出阶段恢复输入 dtype。
    """

    def __call__(self, x1, x2, gamma, epsilon=1e-6, **kwargs):
        calc_dtype = torch.float32
        x = torch.add(x1.to(calc_dtype), x2.to(calc_dtype))
        y1 = torch.rms_norm(x, tuple(gamma.shape), gamma.to(calc_dtype), epsilon)
        y2 = y1.to(x1.dtype)

        outer_shape = x.shape[: x.dim() - gamma.dim()]
        rstd = torch.rsqrt(
            x.reshape((*outer_shape, gamma.numel())).square().mean(-1, keepdim=True)
            + epsilon
        ).reshape((*outer_shape, *([1] * gamma.dim())))
        return [y1, y2, rstd, x.to(x1.dtype)]


class AddRmsNormCastSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    golden = staticmethod(add_rms_norm_cast_golden)
    third_party = {"torch": AddRmsNormCastXpuRef}


class AclnnAddRmsNormCastSpec:
    """aclnn 流程：torch 入/出。"""

    golden = staticmethod(aclnn_add_rms_norm_cast_golden)
    third_party = {"torch": AddRmsNormCastXpuRef}


class TorchAddRmsNormCastSpec(AclnnAddRmsNormCastSpec):
    """e2e 流程：torch_npu.npu_add_rms_norm_cast，输入与输出顺序与 aclnn 一致。"""

    pass
