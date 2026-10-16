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


# 保留函数式 golden 注册：Kernel 接收 NumPy 输入，ACLNN / Torch 接收 Torch 输入。
# Torch 使用 op-plugin 真实公开的接口名。
__golden__ = {
    "kernel": {"add_rms_norm": "add_rms_norm_golden"},
    "aclnn": {"aclnnAddRmsNorm": "aclnn_add_rms_norm_golden"},
    "e2e": {"torch_npu.npu_add_rms_norm": "aclnn_add_rms_norm_golden"},
}


# TestSpec 在现有 CPU golden 之外补充 GPU third_party。
# Kernel 与 GEIR 共用算子名；ACLNN 和 Torch 则按实际 API 名查找对应的 Spec。
__spec__ = {
    "add_rms_norm": "AddRmsNormSpec",  # Kernel / GEIR
    "aclnnAddRmsNorm": "AclnnAddRmsNormSpec",
    "torch_npu.npu_add_rms_norm": "TorchAddRmsNormSpec",
}


def add_rms_norm_golden(
    x1,
    x2,
    gamma,  # inputs
    epsilon: float = 1e-6,  # attributes
    **kwargs,
):
    """
    Golden function for add_rms_norm.
    All the parameters (names and order) follow @add_rms_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    # Kernel 测试传入 NumPy 数组。BF16 用相同的 16 位存储做位视图转换，
    # 不经过 FP16 数值转换，避免改变 BF16 的数值范围和原始数据。
    x1_dtype = x1.dtype
    if x1_dtype.name == "bfloat16":
        x1_tensor = torch.from_numpy(x1.view(np.float16)).view(torch.bfloat16)
        x2_tensor = torch.from_numpy(x2.view(np.float16)).view(torch.bfloat16)
        gamma_tensor = torch.from_numpy(gamma.view(np.float16)).view(torch.bfloat16)
    else:
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        gamma_tensor = torch.from_numpy(gamma)

    # AddRmsNormCast 也复用此函数，通过 _post_action 请求额外的 FP32 输出。
    # 保留主线已有的 SoC 分支：A2/A3 使用 v1 的中间舍入，950 使用 v2。
    post_action = kwargs.get("_post_action")
    short_soc_version = kwargs.get("short_soc_version")
    if short_soc_version in ["Ascend910B", "Ascend910_93"]:
        y_tensor, var_tensor, x_tensor, y_fp32_tensor = add_rms_golden_v1(
            x1_tensor, x2_tensor, gamma_tensor, epsilon, post_action
        )
    else:
        y_tensor, var_tensor, x_tensor, y_fp32_tensor = add_rms_golden_v2(
            x1_tensor, x2_tensor, gamma_tensor, epsilon, post_action
        )

    # Kernel golden 的返回类型必须是 NumPy；y/x 保持输入 dtype，
    # rstd 保持计算 dtype。普通模式的输出顺序为 y、rstd、残差和 x。
    if x1_dtype.name == "bfloat16":
        y = y_tensor.view(torch.float16).numpy().view(x1_dtype)
        x = x_tensor.view(torch.float16).numpy().view(x1_dtype)
    else:
        y = y_tensor.numpy()
        x = x_tensor.numpy()
    rstd = var_tensor.numpy()
    if post_action == "cast":  # add_rms_norm_cast
        y_fp32 = y_fp32_tensor.numpy()
        return y_fp32, y, rstd, x
    else:
        return y, rstd, x


def add_rms_golden_v1(x1, x2, gamma, eps, post_action):
    import torch

    # A2/A3 原有参考路径：先按输入 dtype 保存残差和，再提升到 FP32 归约。
    # 该中间舍入与 950 不同，不能直接替换成 v2 的计算过程。
    if x1.dtype == torch.bfloat16:
        x = (x1.type(torch.float32) + x2.type(torch.float32)).type(x1.dtype)
    else:
        x = x1 + x2

    # gamma.shape 对应输入末尾的归约维。将这些维度展平后求均方，
    # 外部维度 A 保留；A 含 0 时显式指定 0，避免 reshape(..., -1) 歧义。
    xFp32 = x.type(torch.float32)
    if 0 in xFp32.shape[: len(xFp32.shape) - len(gamma.shape)]:  # A 轴存在 0
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
    # rstd = 1 / sqrt(mean(x^2) + eps)，形状为外部维度加归约维上的 1，
    # 因此可以广播回 x。先乘 rstd，再按旧架构规则舍入并乘 gamma。
    tmpX = xFp32 * rstd

    if x1.dtype == torch.bfloat16:
        tmpX = tmpX.type(torch.bfloat16).type(torch.float32)
        y = (tmpX * gamma.type(torch.float32)).type(x1.dtype)
    elif x.dtype == torch.float16:
        tmpX = tmpX.type(torch.float16)
        y = tmpX * gamma
    else:
        y = tmpX * gamma
    y_fp32 = y.type(torch.float32) if post_action == "cast" else None
    return y, rstd, x, y_fp32


def add_rms_golden_v2(x1, x2, gamma, eps, post_action):
    import torch

    # 950 普通模式：FP16/BF16/FP32 输入均在 FP32 中计算。
    # TTK Promote 可能传入 FP64，必须保留，不能降回 FP32 破坏高精度参考。
    # Cast 模式仍沿用原有 FP32 语义。
    calc_dtype = (
        torch.float64
        if post_action != "cast" and torch.float64 in (x1.dtype, x2.dtype, gamma.dtype)
        else torch.float32
    )
    # x1+x2 的中间结果不提前转回低精度；归一化使用未舍入的残差和。
    x = x1.type(calc_dtype) + x2.type(calc_dtype)
    # 归约范围为 gamma 覆盖的末尾各维，支持多维 gamma 和空 Tensor。
    # 外部维度含 0 时避免使用 -1；归约维为空时 mean 自然产生 NaN rstd。
    if 0 in x.shape[: len(x.shape) - len(gamma.shape)]:  # A 轴存在 0
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
    # 按公式先乘 rstd，再乘 gamma，避免 x*gamma 提前溢出。
    tmpX = x * rstd

    y_fp32 = tmpX * gamma.type(calc_dtype)
    # 只在输出阶段恢复 y/x 的输入 dtype；rstd 保留计算精度。
    # y_fp32 这个历史名称也用于 Cast 复用；普通 Promote 模式下它可为 FP64。
    y = y_fp32.type(x1.dtype)
    x = x.type(x1.dtype)
    return y, rstd, x, y_fp32


def aclnn_add_rms_norm_golden(
    x1, x2, gamma, epsilon=1e-6, yOut=None, rstdOut=None, xOut=None, **kwargs
):
    """ACLNN and Torch reference using the existing SoC-specific math."""
    # ACLNN 的可选输出占位参数不参与计算；Torch 直接复用此入口。
    # 两条通路都接收 Torch Tensor，因此无需 Kernel 入口的 NumPy 转换。
    short_soc_version = (
        str(kwargs.get("short_soc_version", "")).lower().replace("_", "")
    )
    if short_soc_version in ("ascend910b", "ascend91093"):
        y, rstd, x, _ = add_rms_golden_v1(x1, x2, gamma, epsilon, None)
    else:
        y, rstd, x, _ = add_rms_golden_v2(x1, x2, gamma, epsilon, None)
    return y, rstd, x


class AddRmsNormXpuRef:
    """Torch add + rms_norm GPU reference for Ascend 950 normal mode.

    Inputs are Torch tensors on the target device. Keep the FP32 residual sum
    until normalization and return y/rstd/x without modifying the inputs.
    This is a composite reference, not a native GPU fused AddRmsNorm API.
    """

    def __call__(self, x1, x2, gamma, epsilon=1e-6, **kwargs):
        import torch

        # xpu-server 已把输入放到 GPU；保留 950 的 FP32 残差加法语义，
        # 不先把 x1+x2 舍入回 FP16/BF16。FP64 输入仍保留其计算精度。
        calc_dtype = (
            torch.float64
            if torch.float64 in (x1.dtype, x2.dtype, gamma.dtype)
            else torch.float32
        )
        x = torch.add(x1.to(calc_dtype), x2.to(calc_dtype))
        # 使用框架的 RMSNorm 接口生成 y，让 GPU 选择其原生实现。
        # gamma.shape 覆盖末尾归约维；只在最终输出阶段恢复输入 dtype。
        y = torch.rms_norm(x, tuple(gamma.shape), gamma.to(calc_dtype), epsilon)
        # torch.rms_norm 只返回 y，额外计算接口要求的 rstd，并保留归约维。
        outer_shape = x.shape[: x.dim() - gamma.dim()]
        rstd = torch.rsqrt(
            x.reshape((*outer_shape, gamma.numel())).square().mean(-1, keepdim=True)
            + epsilon
        ).reshape((*outer_shape, *([1] * gamma.dim())))
        return y.to(x1.dtype), rstd, x.to(x1.dtype)


class AddRmsNormSpec:
    # CPU golden 用于高精度对照；third_party 指定 GPU 端需要实例化的参考类。
    # 不自定义 tolerance，继续采用 TTK 的默认精度标准。
    golden = staticmethod(add_rms_norm_golden)
    third_party = {"torch": AddRmsNormXpuRef}


class AclnnAddRmsNormSpec(AddRmsNormSpec):
    # GPU 参考相同，仅将 CPU golden 换成接收 Torch Tensor 的入口。
    golden = staticmethod(aclnn_add_rms_norm_golden)


class TorchAddRmsNormSpec(AclnnAddRmsNormSpec):
    # Torch API 与 ACLNN 的输入及 y/rstd/x 输出顺序一致，可以直接继承。
    pass
