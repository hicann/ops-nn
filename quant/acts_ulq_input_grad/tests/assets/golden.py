#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
ActsULQInputGrad TTK **kernel** 级插件（Golden + Input）。

算子定义（acts_ulq_input_grad_def.cpp，3 输入 1 输出，无属性）：
    输入: y_grad, clamp_min_mask, clamp_max_mask
    输出: x_grad
逐元素门控乘法：
    signal = clamp_min_mask * clamp_max_mask       (乘子严格 {0,1})
    x_grad = y_grad * signal

Golden 用 torch 小算子拼接（本文件按需求要求以 torch 拼接编写；TTK kernel 级
契约要求返回 numpy，故内部用 torch 计算、末尾转回 numpy —— 即 numpy+torch 混用，
与 math/sort_with_index 同款模式）。

dtype 语义（对齐 kernel 实际计算 acts_ulq_input_grad.h：纯 Reg::Mul，无饱和逻辑）：
  - kernel 两条路径均为纯 IEEE 乘法（K2 half 直算 / K1·K3·K4 fp32 算后 cast）：
      signal = min_mask * max_mask;  x_grad = y_grad * signal
  - fp16 与 fp32 一致按 IEEE 传播 inf/nan：inf*0=nan、nan*任意=nan、inf*1=inf。
    **kernel 无任何 inf→65504 饱和处理**（dump 实证 L1_337：inf×0 → NPU 输出 nan）。
  - 掩码在 QAT 中严格 {0,1}、y_grad 为有限梯度；inf/nan y_grad 属 fuzz 极端输入，
    其结果按 IEEE 传播（与 canndev 动态路径 vmul 语义一致）。

Input 插件：将浮点掩码 snap 到严格 {0,1}（阈值 0.5），bool 掩码天然 {0,1}；
y_grad 原样保留（保留 input_data_ranges 注入的 inf/nan）。
"""

import numpy as np
import torch

__golden__ = {"kernel": {"acts_ulq_input_grad": "acts_ulq_input_grad_golden"}}
__input__ = {"kernel": {"acts_ulq_input_grad": "acts_ulq_input_grad_input"}}


def acts_ulq_input_grad_golden(y_grad, clamp_min_mask, clamp_max_mask, **kwargs):
    """
    Golden for acts_ulq_input_grad. 参数顺序同 def.cpp 输入（不含输出）。

    镜像 kernel 实际计算（纯 IEEE 乘法，无饱和）：
        signal = min_mask * max_mask;  x_grad = y_grad * signal
    在 float32 上拼接后 cast 回输出 dtype——对 {0,1} 掩码语义与 kernel 的
    half 直算(K2)/fp32 算后 cast(K1/K3/K4) 两条路径结果一致，且 inf/nan 按 IEEE 传播
    （inf*0=nan），与 NPU dump 实证一致。

    **kwargs: input_dtypes / output_dtypes / ...（kernel 级元信息）
    返回 numpy.ndarray（x_grad，dtype/shape 与 y_grad 一致）。
    """
    out_dtype = np.dtype(y_grad.dtype)

    # numpy -> torch（保留 inf/nan）
    yg = torch.from_numpy(np.ascontiguousarray(y_grad))
    mn = torch.from_numpy(np.ascontiguousarray(clamp_min_mask))
    mx = torch.from_numpy(np.ascontiguousarray(clamp_max_mask))

    # 纯 IEEE 门控乘法（torch 小算子拼接），fp16/fp32 一致，无饱和：
    signal = mn.to(torch.float32) * mx.to(torch.float32)
    x_grad = yg.to(torch.float32) * signal  # inf*0=nan 自然按 IEEE 传播

    return x_grad.cpu().numpy().astype(out_dtype, copy=False)


def acts_ulq_input_grad_input(y_grad, clamp_min_mask, clamp_max_mask, **kwargs):
    """
    Input for acts_ulq_input_grad。返回 [y_grad, clamp_min_mask, clamp_max_mask]。

    - y_grad：原样保留（保留 input_data_ranges 注入的 inf/nan/极值）。
    - 掩码：snap 到严格 {0,1}（阈值 0.5）。范围 (0,0)→全 0、(1,1)→全 1、
            (0,1)→随机 {0,1}；bool 掩码天然 {0,1}，阈值化幂等。
    """

    def to_binary(mask):
        b = (mask.astype(np.float32) > 0.5).astype(mask.dtype, copy=False)
        return b

    return [y_grad, to_binary(clamp_min_mask), to_binary(clamp_max_mask)]


# 自校验：signal∈{0,1} 位精确；fp16 饱和 / fp32 IEEE 语义
def _selfcheck():
    print("=" * 60)
    print("acts_ulq_input_grad golden 自校验（torch 拼接）")
    print("=" * 60)
    ok = True

    # 1) fp32 基本门控 + IEEE nan 透传
    yg = np.array([1.0, 2.0, np.nan, -3.0], dtype=np.float32)
    mn = np.array([1, 1, 1, 0], dtype=np.float32)
    mx = np.array([1, 0, 1, 1], dtype=np.float32)
    out = acts_ulq_input_grad_golden(yg, mn, mx)
    exp = np.array([1.0, 0.0, np.nan, 0.0], dtype=np.float32)
    m = np.array_equal(np.nan_to_num(out, nan=-9), np.nan_to_num(exp, nan=-9))
    print(f"[{'PASS' if m else 'FAIL'}] fp32 门控+NaN透传: out={out}")
    ok &= m

    # 2) fp16 门控直通位精确
    yg = np.array([3.5, 3.5, 3.5], dtype=np.float16)
    mn = np.array([1, 1, 0], dtype=np.float16)
    mx = np.array([1, 0, 1], dtype=np.float16)
    out = acts_ulq_input_grad_golden(yg, mn, mx)
    exp = np.array([3.5, 0.0, 0.0], dtype=np.float16)
    m = np.array_equal(out, exp)
    print(f"[{'PASS' if m else 'FAIL'}] fp16 门控位精确: out={out}")
    ok &= m

    # 3) fp16 inf/nan 按 IEEE（掩码全 1 → 透传，无饱和）
    yg = np.array([np.inf, -np.inf, np.nan, 100.0], dtype=np.float16)
    mn = np.ones(4, dtype=np.float16)
    mx = np.ones(4, dtype=np.float16)
    out = acts_ulq_input_grad_golden(yg, mn, mx)
    exp = np.array([np.inf, -np.inf, np.nan, 100.0], dtype=np.float16)
    m = np.array_equal(np.nan_to_num(out, nan=-9), np.nan_to_num(exp, nan=-9))
    print(f"[{'PASS' if m else 'FAIL'}] fp16 inf/nan IEEE透传(全1掩码): out={out}")
    ok &= m

    # 4) fp16 掩码 0 位：inf/nan × 0 = nan（IEEE，无门控归零；对齐 kernel 纯 vmul）
    yg = np.array([np.inf, np.nan, -np.inf], dtype=np.float16)
    mn = np.array([0, 0, 1], dtype=np.float16)
    mx = np.array([1, 1, 0], dtype=np.float16)
    out = acts_ulq_input_grad_golden(yg, mn, mx)
    # inf*0=nan, nan*0=nan, -inf*0=nan
    m = bool(np.all(np.isnan(out.astype(np.float32))))
    print(f"[{'PASS' if m else 'FAIL'}] fp16 掩码0位 inf/nan×0=nan(IEEE): out={out}")
    ok &= m

    # 5) Input 插件：float mask snap 到 {0,1}
    mn = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float16)
    mx = np.array([0.9, 0.1, 0.8, 0.4], dtype=np.float16)
    yg = np.ones(4, dtype=np.float16)
    _, b_mn, b_mx = acts_ulq_input_grad_input(yg, mn, mx)
    m = set(np.unique(b_mn).tolist()) <= {0.0, 1.0} and set(
        np.unique(b_mx).tolist()
    ) <= {0.0, 1.0}
    print(f"[{'PASS' if m else 'FAIL'}] Input snap {{0,1}}: mn={b_mn}, mx={b_mx}")
    ok &= m

    # 6) bool 掩码路径（K1/K3）
    yg = np.array([2.0, 2.0], dtype=np.float32)
    mn = np.array([True, False])
    mx = np.array([True, True])
    out = acts_ulq_input_grad_golden(yg, mn, mx)
    exp = np.array([2.0, 0.0], dtype=np.float32)
    m = np.array_equal(out, exp)
    print(f"[{'PASS' if m else 'FAIL'}] bool 掩码(K1/K3): out={out}")
    ok &= m

    print("=" * 60)
    print(f"All tests passed: {ok}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selfcheck() else 1)
