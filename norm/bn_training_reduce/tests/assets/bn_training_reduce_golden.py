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
# BNTrainingReduce golden — BN 训练统计量归约算子（torch 小算子拼接实现）
# 依据：reference/cann/ops-nn/norm/bn_training_reduce/
#       docs/aclnnBatchNormReduce.md「计算公式」（唯一口径，逐字）：
#         sum_i       = Σ_{n=0..N-1} Σ_{h=0..H-1} Σ_{w=0..W-1} x_(n,i,h,w)
#         squareSum_i = Σ_{n=0..N-1} Σ_{h=0..H-1} Σ_{w=0..W-1} x_(n,i,h,w)^2
#       + reference/cann/canndev/ops/built-in/tbe/impl/bn_training_reduce.py
#         （_reduce_compute_nd / _reduce_compute_5hd：保留 C 轴，归约其余轴；
#          FP16/BF16 输入先提升 FP32 再累加）
#       + reference/ascend/op-plugin/op_plugin/ops/aclops/BatchNormReduceKernelNpu.cpp
#         （输出强制 FP32；非 FP32 输入先 cast FP32）
# 拼接口径（PyTorch 无独立 reduce 单算子，参考 research/opensource/pytorch/code_design.md）：
#   官方 batch_norm 训练 reduce 环节为 torch.sum / torch.mul 组合实现（reduce 语义），
#   本 golden 以 torch 基础算子（sum / mul）逐项拼接 CANN 公式：
#     sum       = torch.sum(x, dim=axis)                # 沿非通道轴求和
#     squareSum = torch.sum(x * x, dim=axis)            # 沿非通道轴求平方和
# 数值口径：
#   - 输入 dtype 支持 FLOAT32 / FLOAT16 / BFLOAT16；输出恒为 FLOAT32（GEIR / aclnn 一致）
#   - 计算一律在 FP32 进行（FP16/BF16 先提升，BF16 用 FP32 承载等价计算）
#   - 归约轴 = 除通道轴（dim=1）外的全部轴；aclnn 接口场景为 4D NCHW -> axis=(0,2,3)
#   - 支持 rank>=2 的 NCHW 系输入（GE 层 2~4D），rank!=4 时归约 axis = 除 1 外全部
#   - 空 Tensor 语义（aclnn）：输出置 0（本 golden 不特判，由空输入自然得到 0 输出）
# 输出顺序：sum, square_sum（与 aclnnBatchNormReduce(x, sum, squareSum) 一致）
# ----------------------------------------------------------------------------
from typing import Optional, Sequence

import numpy as np
import torch

__golden__ = {
    "kernel": {"bn_training_reduce": "bn_training_reduce_golden"},
    "aclnn": {"aclnnBatchNormReduce": "bn_training_reduce_golden"},
}


def _as_np(x) -> Optional[np.ndarray]:
    """把 torch.Tensor / np.ndarray 统一转成 numpy 数组（保持原 dtype；bf16 以 fp32 承载）。"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu()
        if arr.dtype == torch.bfloat16:
            return arr.float().numpy()
        return arr.numpy()
    return np.asarray(x)


def _to_fp32(x) -> Optional[np.ndarray]:
    """提升到 FP32（fp16/bfloat16 输入；BF16 按 uint16 位级读值承载）。"""
    x = _as_np(x)
    if x is None:
        return None
    if str(x.dtype) == "float16":
        return x.astype(np.float32)
    if str(x.dtype) == "bfloat16":
        return x.view(np.uint16).astype(np.float32) * 0.0 + x.astype(np.float32)
    return x.astype(np.float32)


def _to_torch_fp32(x) -> torch.Tensor:
    """把 numpy 数组提升为 torch FP32 张量（小算子拼接的计算载体）。"""
    arr = _to_fp32(x)
    assert arr is not None, "输入不能为空"
    return torch.from_numpy(arr)


def bn_training_reduce_golden(
    x, sum=None, square_sum=None, format: str = "NCHW", **kwargs
) -> list:
    """
    Golden function for bn_training_reduce kernel (BN training statistics reduce).

    Inputs:
        x: torch.Tensor|np.ndarray，rank>=2 的 NCHW 系张量 [N, C, ...]
           dtype 支持 FLOAT32 / FLOAT16 / BFLOAT16（BF16 以 FP32 承载）
        sum / square_sum: 可选输出张量（golden 不消费，仅用于签名对齐）
        format: 输入布局，支持 "NCHW"（rank 2~4，C=dim1）/ "NHWC"（rank 4，C=dim3）/
                "NCDHW"（rank 5，C=dim1）；默认 "NCHW"

    Returns:
        [sum, square_sum]（两个 np.ndarray，shape [C]，dtype FLOAT32）
        - sum:        沿非通道轴的元素和
        - square_sum: 沿非通道轴的平方和
    """
    x_np = _as_np(x)
    assert x_np is not None, "必选输入 x 不能为空"
    assert x_np.ndim >= 2, "x 需 rank>=2"

    fmt = str(format).upper()

    # C 轴索引按格式决定（NCHW/NCDHW → dim 1；NHWC → dim 3，即末维）
    if fmt == "NHWC":
        c_axis = x_np.ndim - 1
    elif fmt in ("NCHW", "NCDHW"):
        c_axis = 1
    else:
        raise ValueError(f"不支持的 format: {format}（支持 NCHW / NHWC / NCDHW）")
    assert c_axis < x_np.ndim, f"format={format} 与 rank={x_np.ndim} 不匹配"

    # 归约轴：除通道轴外的全部轴；保留 C 轴：输出 [C]
    axis: Sequence[int] = tuple(i for i in range(x_np.ndim) if i != c_axis)
    out_shape = (int(x_np.shape[c_axis]),)

    # ---- torch 小算子拼接（sum / mul，严格照 aclnnBatchNormReduce 公式） ----
    x_t = _to_torch_fp32(x_np)  # FP16/BF16 提升 FP32 再累加
    axis_t = tuple(axis)
    sum_t = torch.sum(x_t, dim=axis_t, keepdim=False)  # Σ x
    square_sum_t = torch.sum(torch.mul(x_t, x_t), dim=axis_t, keepdim=False)  # Σ x²

    sum_out = sum_t.numpy().reshape(out_shape).astype(np.float32)
    square_sum_out = square_sum_t.numpy().reshape(out_shape).astype(np.float32)

    return [sum_out, square_sum_out]


if __name__ == "__main__":
    # 冒烟自测（本地运行验证，不影响 TTK 插件入口）
    def _check(name, got, want, tol=1e-3):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
        assert np.allclose(got, want, rtol=tol, atol=tol), (
            f"{name}: value mismatch\n{got}\n{want}"
        )
        print(f"PASS: {name}")

    # case1: FP32 NCHW [2,3,4,5]
    x1 = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    s1, sq1 = bn_training_reduce_golden(x1)
    _check("fp32 nchw sum", s1, x1.sum(axis=(0, 2, 3)))
    _check("fp32 nchw square_sum", sq1, (x1 * x1).sum(axis=(0, 2, 3)))

    # case2: FP16 [1,4,2,3]（验证 FP16 -> FP32 提升）
    x2 = torch.randn(1, 4, 2, 3, dtype=torch.float16)
    s2, sq2 = bn_training_reduce_golden(x2)
    x2f = x2.float().numpy()
    _check("fp16 sum", s2, x2f.sum(axis=(0, 2, 3)), tol=1e-2)
    _check("fp16 square_sum", sq2, (x2f * x2f).sum(axis=(0, 2, 3)), tol=1e-2)

    # case3: BF16 [2,3,1,1]（H=W=1 边界；BF16 以 FP32 承载）
    x3 = torch.randn(2, 3, 1, 1, dtype=torch.bfloat16)
    s3, sq3 = bn_training_reduce_golden(x3)
    x3f = x3.float().numpy()
    _check("bf16 sum", s3, x3f.sum(axis=(0, 2, 3)), tol=1e-2)
    _check("bf16 square_sum", sq3, (x3f * x3f).sum(axis=(0, 2, 3)), tol=1e-2)

    # case4: 大值 FP16 [1,2,64,64]（验证 FP32 累加防溢出）
    x4 = torch.full((1, 2, 64, 64), 1000.0, dtype=torch.float16)
    s4, sq4 = bn_training_reduce_golden(x4)
    n_elems = 1 * 64 * 64
    _check(
        "fp16 large sum",
        s4,
        np.full((2,), 1000.0 * n_elems, dtype=np.float32),
        tol=1e-2,
    )
    _check(
        "fp16 large square_sum",
        sq4,
        np.full((2,), 1e6 * n_elems, dtype=np.float32),
        tol=5e-2,
    )

    # case5: 2D 输入 [4,3]（rank2 NCHW，C=3，归约 axis=(0,)）
    x5 = np.arange(12, dtype=np.float32).reshape(4, 3)
    s5, sq5 = bn_training_reduce_golden(x5)
    _check("rank2 sum", s5, x5.sum(axis=0))
    _check("rank2 square_sum", sq5, (x5 * x5).sum(axis=0))

    # case6: NHWC 4D [2,4,5,3]（format=NHWC，C=末维 dim3，归约 axis=(0,1,2)）
    x6 = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
    s6, sq6 = bn_training_reduce_golden(x6, format="NHWC")
    _check("nhwc sum", s6, x6.sum(axis=(0, 1, 2)))
    _check("nhwc square_sum", sq6, (x6 * x6).sum(axis=(0, 1, 2)))

    # case7: NCDHW 5D [1,3,2,4,5]（format=NCDHW，C=dim1，归约 axis=(0,2,3,4)）
    x7 = np.arange(1 * 3 * 2 * 4 * 5, dtype=np.float32).reshape(1, 3, 2, 4, 5)
    s7, sq7 = bn_training_reduce_golden(x7, format="NCDHW")
    _check("ncdhw sum", s7, x7.sum(axis=(0, 2, 3, 4)))
    _check("ncdhw square_sum", sq7, (x7 * x7).sum(axis=(0, 2, 3, 4)))

    # case8: 非法 format 应抛 ValueError
    try:
        bn_training_reduce_golden(x1, format="UNKNOWN")
        raise AssertionError("expected ValueError for UNKNOWN format")
    except ValueError:
        print("PASS: unknown format raises ValueError")

    print("ALL SMOKE TESTS PASSED")
