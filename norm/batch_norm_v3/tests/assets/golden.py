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

"""BatchNormV3 golden：BatchNormV3 单算子参考实现。

kernel / GEIR = ``batch_norm_v3``（numpy 入/出）：训练语义下
y = (x - mean) / sqrt(var + eps) * weight + bias，mean / var 为按 channel 的
批统计；running_mean / running_var 按 momentum 动量更新；save_mean 为批均值、
save_rstd（或 save_var，见下）为批统计量。真值在 FP32（fp16/bf16 输入）或
FP64（fp32 输入）中完成。

aclnn = ``aclnnBatchNorm``（torch 入/出）：输出 [y, saveMean, saveInvstd]
（训练）或 [y]（推理）；runningMean / runningVar 为原位输入输出，不参与比对。

e2e = ``torch.ops.aten.native_batch_norm``（torch 入/出）：输出
[y, save_mean, save_invstd]，与 torch ATen 原生 batch norm 语义一致。

芯片差异：第 5 个输出在 950 系列为 input 标准差倒数 rstd =
1/sqrt(var(x)+eps)；在 Ascend910B / Ascend910_93 为 input 方差 var(x)。
"""

import numpy as np
import torch

__spec__ = {
    "batch_norm_v3": "BatchNormV3Spec",  # Kernel / GEIR
    "aclnnBatchNorm": "AclnnBatchNormSpec",  # ACLNN
    "torch.ops.aten.native_batch_norm.default": "TorchBatchNormV3Spec",  # E2E
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
        tensor = tensor.to(torch.bfloat16)
        return tensor.view(torch.float16).numpy().view(np_dtype)
    return np.ascontiguousarray(tensor.numpy().astype(np_dtype, copy=False))


def _compute_dtype(x_dtype, promote=True):
    """计算精度：fp16/bf16 → fp32；fp32 → fp64。

    promote=False 时保持输入 dtype——三方（GPU）按 case 初始 dtype 计算，与被测 NPU 同精度。
    """
    if not promote:
        return x_dtype
    if x_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return torch.float64


def _is_save_var(short_soc_version):
    """第 5 个输出是否为方差（而非 rstd）。"""
    soc = str(short_soc_version or "").lower().replace("_", "")
    return soc in ("ascend910b", "ascend91093")


def _to_nchw(x, data_format):
    """NHWC / NDHWC → NCHW / NCDHW（其余格式 channel 在 dim 1，无需置换）。"""
    if data_format == "NHWC":
        return x.permute(0, 3, 1, 2)
    if data_format == "NDHWC":
        return x.permute(0, 4, 1, 2, 3)
    return x


def _back_from_nchw(y, data_format):
    """NCHW / NCDHW → NHWC / NDHWC。"""
    if data_format == "NHWC":
        return y.permute(0, 2, 3, 1)
    if data_format == "NDHWC":
        return y.permute(0, 2, 3, 4, 1)
    return y


def _get_format(kwargs, x_ndim, fmts_key):
    """从 kwargs 元信息解析 x 的数据格式，缺省按 rank 推断（ND 系列 channel=1）。"""
    fmts = (
        kwargs.get(fmts_key)
        or kwargs.get("input_formats")
        or kwargs.get("input_ori_formats")
    )
    if isinstance(fmts, (list, tuple)) and fmts and isinstance(fmts[0], str):
        return str(fmts[0])
    if x_ndim == 5:
        return "NCDHW"
    if x_ndim == 4:
        return "NCHW"
    return "ND"


def _batch_norm_core(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    data_format,
    short_soc_version=None,
    promote=True,
):
    """核心计算（torch 张量入/出），返回 (y, running_mean, running_var, save_mean, save_invstd)。

    save_invstd 按芯片语义：950 系列为 rstd，910B/910_93 为方差。
    running_mean / running_var 返回更新后的值（推理时原样返回输入）。
    promote=False 时按输入 dtype 计算（三方 GPU 与 NPU 同精度）。
    """
    calc_dtype = _compute_dtype(x.dtype, promote)
    x_c = x.to(calc_dtype)
    weight_c = weight.to(calc_dtype)
    bias_c = bias.to(calc_dtype)
    running_mean_c = running_mean.to(calc_dtype).clone()
    running_var_c = running_var.to(calc_dtype).clone()

    x_nchw = _to_nchw(x_c, data_format)
    res = torch.ops.aten.native_batch_norm(
        input=x_nchw,
        weight=weight_c,
        bias=bias_c,
        running_mean=running_mean_c,
        running_var=running_var_c,
        training=bool(training),
        momentum=float(momentum),
        eps=float(eps),
    )
    y = _back_from_nchw(res[0], data_format)

    save_mean = res[1] if res[1] is not None and res[1].numel() > 0 else None
    save_invstd = res[2] if res[2] is not None and res[2].numel() > 0 else None

    if save_invstd is not None and _is_save_var(short_soc_version):
        # 910B / 910_93：输出方差 = 1 / rstd^2 - eps
        save_invstd = save_invstd.pow(-2) - float(eps)

    return y, running_mean_c, running_var_c, save_mean, save_invstd


def batch_norm_v3_golden(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    epsilon=1e-5,
    momentum=0.1,
    is_training=True,
    **kwargs,
):
    """Kernel/GEIR golden：numpy 入/出，返回 [y, running_mean, running_var, save_mean, save_invstd]。"""
    x_t = _np_to_torch(x)
    weight_t = _np_to_torch(weight)
    bias_t = _np_to_torch(bias)
    running_mean_t = _np_to_torch(running_mean)
    running_var_t = _np_to_torch(running_var)

    data_format = _get_format(kwargs, x_t.dim(), "input_formats")
    short_soc = kwargs.get("short_soc_version")

    y, rmean, rvar, save_mean, save_invstd = _batch_norm_core(
        x_t,
        weight_t,
        bias_t,
        running_mean_t,
        running_var_t,
        is_training,
        momentum,
        epsilon,
        data_format,
        short_soc,
    )

    c_numel = _to_nchw(x_t, data_format).shape[1] if x_t.dim() >= 2 else x_t.numel()
    outputs = [
        _torch_to_numpy(y, x.dtype),
        _torch_to_numpy(rmean, running_mean.dtype),
        _torch_to_numpy(rvar, running_var.dtype),
        _torch_to_numpy(save_mean, np.float32)
        if save_mean is not None
        else np.zeros((c_numel,), np.float32),
        _torch_to_numpy(save_invstd, np.float32)
        if save_invstd is not None
        else np.zeros((c_numel,), np.float32),
    ]
    return outputs


def aclnn_batch_norm_golden(
    input,
    weight,
    bias,
    runningMean,
    runningVar,
    training,
    momentum,
    eps,
    output=None,
    saveMean=None,
    saveInvstd=None,
    **kwargs,
):
    """ACLNN golden：torch 入/出。

    参数位置跟随 aclnnBatchNormGetWorkspaceSize（不含 workspaceSize 与 executor）。
    训练返回 [y, saveMean, saveInvstd]；推理（saveMean/saveInvstd 不写有效内容）
    仅返回 [y]。
    """
    data_format = _get_format(kwargs, input.dim(), "tensor_formats")
    y, _rmean, _rvar, save_mean, save_invstd = _batch_norm_core(
        input,
        weight,
        bias,
        runningMean,
        runningVar,
        training,
        momentum,
        eps,
        data_format,
        kwargs.get("short_soc_version"),
    )

    y_out = y.to(input.dtype)
    if not training:
        return [y_out]

    save_mean_out = (
        save_mean.to(saveMean.dtype) if saveMean is not None else save_mean.float()
    )
    save_invstd_out = (
        save_invstd.to(saveInvstd.dtype)
        if saveInvstd is not None
        else save_invstd.float()
    )
    return [y_out, save_mean_out, save_invstd_out]


def native_batch_norm_golden(
    input, weight, bias, running_mean, running_var, training, momentum, eps, **kwargs
):
    """E2E golden：torch.ops.aten.native_batch_norm，torch 入/出。

    参数位置跟随 aten::native_batch_norm 的 schema 形参序；返回
    [y, save_mean, save_invstd]（save_invstd = 1/sqrt(var(x)+eps)，与 torch
    ATen 语义一致，不随芯片二次换算；running_mean/running_var 为原位更新，
    不进返回与比对）。
    """
    data_format = _get_format(kwargs, input.dim(), "tensor_formats")
    y, _rmean, _rvar, save_mean, save_invstd = _batch_norm_core(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        data_format,
    )

    y_out = y.to(input.dtype)
    if not training:
        return [y_out]
    save_mean_out = save_mean.float() if save_mean is not None else running_mean.float()
    save_invstd_out = (
        save_invstd.float() if save_invstd is not None else running_var.float()
    )
    return [y_out, save_mean_out, save_invstd_out]


class BatchNormV3XpuRef:
    """torch 三方标杆（GPU 参考，用于 cross_check，5 输出形态）。

    输入为设备侧 torch.Tensor：训练语义下与 golden 走同一 native_batch_norm。
    """

    def __call__(
        self,
        x,
        weight,
        bias,
        running_mean,
        running_var,
        *,
        momentum=0.1,
        epsilon=1e-5,
        is_training=True,
        **kwargs,
    ):
        data_format = _get_format(kwargs, x.dim(), "input_formats")
        y, rmean, rvar, save_mean, save_invstd = _batch_norm_core(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            is_training,
            momentum,
            epsilon,
            data_format,
            kwargs.get("short_soc_version"),
            promote=False,
        )
        if save_mean is None:
            # 推理（is_training=False）：native_batch_norm 的 save_mean/save_invstd
            # 为空张量，与 kernel golden 一样按通道数补零，保持 5 输出形态，
            # 不能像 aclnn/e2e 三方那样退化为 1 输出早返回。
            c_numel = _to_nchw(x, data_format).shape[1] if x.dim() >= 2 else x.numel()
            save_mean = torch.zeros(c_numel, dtype=torch.float32, device=x.device)
            save_invstd = torch.zeros(c_numel, dtype=torch.float32, device=x.device)
        return [
            y.to(x.dtype),
            rmean.to(running_mean.dtype),
            rvar.to(running_var.dtype),
            save_mean.float(),
            save_invstd.float(),
        ]


class AclnnBatchNormXpuRef:
    """torch 三方标杆（aclnnn 3 输出形态）。"""

    def __call__(
        self,
        input,
        weight,
        bias,
        runningMean,
        runningVar,
        training,
        momentum,
        eps,
        **kwargs,
    ):
        data_format = _get_format(kwargs, input.dim(), "tensor_formats")
        y, _rmean, _rvar, save_mean, save_invstd = _batch_norm_core(
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training,
            momentum,
            eps,
            data_format,
            kwargs.get("short_soc_version"),
        )
        if not training:
            return [y.to(input.dtype)]
        return [y.to(input.dtype), save_mean.float(), save_invstd.float()]


class TorchBatchNormV3XpuRef:
    """torch 三方标杆（e2e 3 输出形态，rstd 语义与 torch ATen 一致）。"""

    def __call__(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        **kwargs,
    ):
        data_format = _get_format(kwargs, input.dim(), "tensor_formats")
        y, _rmean, _rvar, save_mean, save_invstd = _batch_norm_core(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            data_format,
        )
        if not training:
            return [y.to(input.dtype)]
        return [y.to(input.dtype), save_mean.float(), save_invstd.float()]


class BatchNormV3Spec:
    """kernel/GEIR 流程：numpy 入/出，5 输出。"""

    golden = staticmethod(batch_norm_v3_golden)
    third_party = {"torch": BatchNormV3XpuRef}


class AclnnBatchNormSpec:
    """aclnn 流程：torch 入/出，训练 3 输出 / 推理 1 输出。"""

    golden = staticmethod(aclnn_batch_norm_golden)
    third_party = {"torch": AclnnBatchNormXpuRef}


class TorchBatchNormV3Spec:
    """e2e 流程：torch 入/出，训练 3 输出（y/save_mean/save_invstd）。"""

    golden = staticmethod(native_batch_norm_golden)
    third_party = {"torch": TorchBatchNormV3XpuRef}
