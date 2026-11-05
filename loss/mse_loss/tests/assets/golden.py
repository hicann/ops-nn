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

import functools
import numpy as np
import torch

__golden__ = {
    "kernel": {"mse_loss": "mse_loss_golden", "mse_loss_grad": "mse_loss_grad_golden"},
    "aclnn": {
        "aclnnMseLoss": "aclnn_mse_loss_golden",
    },
}

_REDUCTION_INT_TO_STR = {0: "none", 1: "mean", 2: "sum"}

# aclnnMseLossOut 只经 __spec__ 注册（loader 的 TestSpec 优先于 __golden__，
# 双注册会产生绕过 spec 的死配置）。aclnnMseLoss 保持原 __golden__ 路径不变。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _red_str(v):
    """reduction 归一到字符串：kernel/e2e 通路本来就是 str，aclnn 通路是 int64。"""
    if isinstance(v, str):
        return v.strip().lower()
    return _REDUCTION_INT_TO_STR.get(int(v), "mean")


def aclnn_mse_loss_golden(self_t, target_t, reduction, out_t=None, **kwargs):
    import torch

    red = _REDUCTION_INT_TO_STR.get(int(reduction), "mean")
    diff = self_t - target_t
    sqr = diff * diff
    if red == "sum":
        result = sqr.to(torch.float32).sum()
    elif red == "mean":
        result = sqr.to(torch.float32).sum() / sqr.numel()
    else:
        result = sqr
    # Ascend ReduceSumOp produces nan when input contains inf (overflow in sqr)
    if red in ("sum", "mean") and torch.isinf(sqr.to(torch.float32)).any():
        result = torch.tensor(float("nan"), dtype=torch.float32)
    if out_t is not None and hasattr(out_t, "dtype"):
        result = result.to(out_t.dtype)
    return result


def aclnn_mse_loss_out_golden(
    self_t, target_t, reduction, out_t=None, out_dtype=None, **kwargs
):
    """aclnnMseLossOut：mean/sum 归约 0 轴（broadcast 去首维），none 逐元素
    （语义见 docs/aclnnMseLossOut.md 与 aclnn_mse_loss_out.cpp）。

    计算档 = TTK 交付档，不自行抬精度；末位照声明 out dtype 落取值网格与
    溢出语义（fp16 out 放不下时 golden 与 NPU 同为 ±inf），容器回到交付档
    （直返 fp16 会在 np.promote_types(fp16, bfloat16) 抛 TypeError）。
    """
    red = _red_str(reduction)
    diff = self_t - target_t
    sqr = diff * diff
    if sqr.numel() == 0:
        # 空张量: NPU 空路径实测输出全 1.0, 不走归约
        shape = sqr.shape[1:] if red in ("sum", "mean") else sqr.shape
        result = torch.ones(shape, dtype=diff.dtype, device=diff.device)
    elif red == "sum":
        result = sqr.sum(dim=0)
    elif red == "mean":
        result = sqr.sum(dim=0) / sqr.shape[0]
    else:
        result = sqr
    dt = out_dtype
    if dt is None and out_t is not None and hasattr(out_t, "dtype"):
        dt = out_t.dtype
    if dt is not None:
        result = result.to(_to_torch_dtype(dt)).to(diff.dtype)
    return result


def _flatten(nested):
    """递归平铺嵌套 tuple/list（tensor_dtypes 可能按 TensorList 嵌套）。"""
    if isinstance(nested, (str, bytes)):  # 裸字符串是单元素，不可逐字符迭代
        yield nested
        return
    for x in nested:
        if isinstance(x, (tuple, list)):
            yield from _flatten(x)
        else:
            yield x


def _to_torch_dtype(dt):
    """dtype 归一到 torch.dtype：接受 torch.dtype 与 CSV 声明字符串（'float16' 等）。"""
    if isinstance(dt, torch.dtype):
        return dt
    return {
        "float": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "double": torch.float64,
        "float64": torch.float64,
    }[str(dt).strip().lower()]


def _third_party_mse_out(self_t, target_t, reduction, out_t=None):
    """三方腿：torch 原生接口、原生 dtype，不自行抬 fp32。
    与 NPU 链路（MSELossV2→ReduceMean/Sum→Cast，中间均落输入 dtype）逐算子
    同构；抬 fp32 跳过中间舍入会让三方误差虚低、cross_check 三比值误判 NPU。
    空张量输出全 1.0（与 NPU 实测对齐）。"""
    red = _red_str(reduction)
    # F.mse_loss 要求同 dtype：混合输入按 result_type 对齐 NPU PromoteType
    work = torch.result_type(self_t, target_t)
    d = torch.nn.functional.mse_loss(
        self_t.to(work), target_t.to(work), reduction="none"
    )
    if d.numel() == 0:
        shape = d.shape[1:] if red in ("sum", "mean") else d.shape
        out = torch.ones(shape, dtype=d.dtype, device=d.device)
    elif red == "mean":
        out = d.mean(dim=0)
    elif red == "sum":
        out = d.sum(dim=0)
    else:
        out = d
    if out_t is not None and hasattr(out_t, "dtype"):
        out = out.to(out_t.dtype)
    return out


class _MseLossOutCompose:
    """aclnnMseLossOut 三方标杆：入参按 aclnn 头文件顺序 (self, target, reduction, out)。

    ⚠️ `self` 仅位置参数：aclnn 首参名就叫 self，服务端按名绑定时 self=<tensor>
    会以关键字进 **_；positional-only 声明后关键字 self 落进 **kw，不撞实例参数。
    """

    def __init__(self, /, reduction=1, **_):
        self._red = _red_str(reduction)

    def __call__(self, /, *args, **kw):
        x = args[0] if len(args) > 0 else kw.get("self", kw.get("input"))
        t = args[1] if len(args) > 1 else kw.get("target")
        red = args[2] if len(args) > 2 else kw.get("reduction", self._red)
        out_t = args[3] if len(args) > 3 else kw.get("out")
        return [_third_party_mse_out(x, t, red, out_t)]


class MseLossOutAclnnSpec:
    """aclnn 通路 spec（aclnnMseLossOut）：golden 收 CPU torch.Tensor、返 Tensor。

    签名取 aclnn_mse_loss_out.h 参数序（self/target/reduction/out）。
    golden 末位照声明 out dtype 落网格+溢出语义，dtype 来源按可靠性：
    out 实张量 dtype（纯输出不被 Promote 抬档，探针实证以声明 dtype 送达）
    → kwargs tensor_dtypes 末位。不取 kwargs["output_dtypes"]：aclnn 通路
    不下发该键。不砍回时 fp16 out 溢出位 golden 有限、NPU ±inf，
    cross_check Path A 必误判 NPU（L0_003）。
    """

    def golden(self, target, reduction=1, out=None, **kwargs):
        # 可靠性排序: out 实张量 dtype → tensor_dtypes 末位(纯输出声明档)
        out_dtype = getattr(out, "dtype", None)
        if out_dtype is None:
            td = kwargs.get("tensor_dtypes")
            if td is not None:
                flat = [x for x in _flatten(td) if x is not None]
                if flat:
                    out_dtype = flat[-1]
        return [
            aclnn_mse_loss_out_golden(self, target, reduction, out, out_dtype=out_dtype)
        ]

    third_party = {"torch": _MseLossOutCompose}
    tolerance = _TOL


__spec__ = {
    "aclnnMseLossOut": "MseLossOutAclnnSpec",
}


def mse_loss_golden(input0, input1, *, reduction="mean", **kwargs):
    """
    Golden function for mse_loss.
    All the parameters (names and order) follow @mse_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    ori_dtype = input0.dtype
    reshape_input0 = [functools.reduce(lambda x, y: x * y, input0.shape)]
    reshape_input1 = [functools.reduce(lambda x, y: x * y, input1.shape)]
    predict = np.reshape(input0, reshape_input0)
    label = np.reshape(input1, reshape_input1)

    if "float16" in ori_dtype.name:
        predict = predict.astype("float32")
        label = label.astype("float32")

    reduce_elts = 1.0
    for i in reshape_input0:
        reduce_elts *= i
    cof = reduce_elts ** (-1)

    axis_d = tuple(range(len(reshape_input0)))
    res = np.subtract(predict, label)
    res_sqr = np.multiply(res, res)

    y = 0.0
    if reduction == "mean":
        y = np.sum(res_sqr, axis=axis_d, keepdims=False)
        y = np.multiply(y, cof)
    elif reduction == "sum":
        y = np.sum(res_sqr, axis=axis_d, keepdims=False)
    elif reduction == "none":
        y = res_sqr

    if "float16" in ori_dtype.name:
        y = y.astype(ori_dtype, copy=False)

    return y


def mse_loss_grad_golden(predict, label, dout, *, reduction="mean", **kwargs):
    ori_dtype = predict.dtype
    p_shape = predict.shape
    l_shape = label.shape
    d_shape = dout.shape

    shape_list = [p_shape, l_shape, d_shape]
    max_shape = shape_list[0]
    for s in shape_list[1:]:
        max_shape = (
            np.broadcast(max_shape, s).shape
            if hasattr(np, "broadcast")
            else [max(s1, s2) for s1, s2 in zip(max_shape, s)]
        )

    predict = np.broadcast_to(predict, max_shape)
    label = np.broadcast_to(label, max_shape)
    dout = np.broadcast_to(dout, max_shape)

    if ori_dtype.name in ("float16", "bfloat16"):
        predict = predict.astype("float32")
        label = label.astype("float32")
        dout = dout.astype("float32")

    sub_res = np.subtract(predict, label)
    if "mean" == reduction:
        reduce_elts = 1.0
        for i in p_shape:
            reduce_elts *= i
        cof = (reduce_elts ** (-1)) * 2.0
    else:
        cof = 2.0

    norm_grad = sub_res * cof
    grad_res = norm_grad * dout
    return grad_res.astype(ori_dtype, copy=False)
