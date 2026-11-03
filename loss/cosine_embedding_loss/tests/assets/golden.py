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

"""Torch CPU golden for CosineEmbeddingLoss."""

import numpy as np
import torch

__spec__ = {
    "cosine_embedding_loss": "CosineEmbeddingLossKernelSpec",
}

__golden__ = {"kernel": {"cosine_embedding_loss": "__golden_cosine_embedding_loss"}}

EPS = 1.0e-12

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _resolve(kwargs, margin, reduction):
    attrs = kwargs.get("attrs")
    if isinstance(attrs, dict):
        margin = attrs.get("margin", margin)
        reduction = attrs.get("reduction", reduction)
    return float(margin), str(reduction)


def _torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype)
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }.get(name, torch.float32)


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return _torch_dtype(dtype) if dtype is not None else default


def _to_compute_tensor(value, compute_dtype):
    """Convert to torch tensor, lifting integer inputs to the compute dtype."""
    tensor = torch.as_tensor(np.asarray(value))
    if not tensor.is_floating_point():
        return tensor.to(compute_dtype)
    return tensor


def cosine_embedding_loss_golden(
    x1, x2, target, margin=0.0, reduction="mean", **kwargs
):
    margin, reduction = _resolve(kwargs, margin, reduction)
    a = _to_compute_tensor(x1, torch.float32)
    b = _to_compute_tensor(x2, torch.float32)
    # 超大广播 shape 分块计算，避免本地 CPU golden 物化数百 GB 而 OOM
    if reduction != "none" and _needs_chunking(a, b):
        # 与未分块路径保持一致的 compute_dtype 推导：先 promote(fp32, a.dtype)，
        # 再与输出 dtype 二次 promote（保留 TTK Promote 提供的更宽精度，如 fp64）。
        compute_dtype = torch.promote_types(torch.float32, a.dtype)
        out_dtype = _output_dtype(kwargs, 0, compute_dtype)
        compute_dtype = torch.promote_types(compute_dtype, out_dtype)
        total, total_n = _chunked_loss(a, b, target, margin, reduction, compute_dtype)
        if reduction == "sum":
            return total.to(out_dtype).reshape(1).numpy()
        n = total_n if total_n > 0 else 1
        return (total / n).to(out_dtype).reshape(1).numpy()

    a, b = torch.broadcast_tensors(a, b)
    if a.dim() < 2:
        raise ValueError("broadcast rank of x1 and x2 must be at least 2")
    # Compute in at least fp32, retaining the wider dtype TTK Promote supplies
    # (fp64 for a fp32 case). Integer x1/x2 inputs are lifted through the
    # promoted output dtype so the true value stays at the promoted precision.
    compute_dtype = torch.promote_types(torch.float32, a.dtype)
    out_dtype = _output_dtype(kwargs, 0, compute_dtype)
    compute_dtype = torch.promote_types(compute_dtype, out_dtype)
    a = a.to(compute_dtype)
    b = b.to(compute_dtype)
    t = _to_compute_tensor(target, compute_dtype)

    dot = torch.sum(a * b, dim=1)
    s1 = torch.sum(a * a, dim=1)
    s2 = torch.sum(b * b, dim=1)
    eps = torch.tensor(EPS, dtype=compute_dtype)
    denom = torch.sqrt(s1 + eps) * torch.sqrt(s2 + eps)
    cos = dot / denom

    cos, t = torch.broadcast_tensors(cos, t)
    pos = torch.tensor(1.0, dtype=compute_dtype) - cos
    neg = torch.maximum(
        torch.tensor(0.0, dtype=compute_dtype),
        cos - torch.tensor(margin, dtype=compute_dtype),
    )
    loss = torch.where(
        t == 1.0, pos, torch.where(t == -1.0, neg, torch.zeros_like(pos))
    )

    if reduction == "none":
        return loss.to(out_dtype).numpy()
    if reduction == "sum":
        return torch.sum(loss).reshape(1).to(out_dtype).numpy()
    denom_n = loss.numel() if loss.numel() > 0 else 1
    return (torch.sum(loss) / denom_n).reshape(1).to(out_dtype).numpy()


def __golden_cosine_embedding_loss(x1, x2, target, **kwargs):
    return [cosine_embedding_loss_golden(x1, x2, target, **kwargs)]


_cosine_embedding_loss_spec_golden = __golden_cosine_embedding_loss


# broadcast 物化超过该元素数时改用分块，避免超大 shape 广播 OOM
# （如 x1(179,1648330) 与 x2(3,7,1,4,6,1,1) 广播到 ~5e11 元素会分配数百 GB，
# 三方 GPU 与本地 CPU golden 均会 OOM）。
_CHUNK_THRESHOLD = 4_000_000
# 分块列数：每块广播子块峰值 ≈ (除 reduce 维外的行数) × CHUNK_COLS。
# 8192 使 c_s_023(1648330 列) 只需 ~201 块，避免 1024 时 1610 块的巨额循环开销。
_CHUNK_COLS = 8192


def _chunked_loss(
    x1, x2, target, margin, reduction, compute_dtype, chunk_cols=_CHUNK_COLS
):
    """通用分块 cosine loss：沿 x1 元素最多的维分块，每块只物化 CHUNK 大小的
    广播子块后逐块 reduce 累加。与未分块路径数学语义一致，仅累加顺序变化。

    仅支持 reduction sum/mean（none 的输出本身与广播同大小，分块无法降低峰值）。
    返回 (结果标量 tensor, 元素总数)。
    """
    target = torch.as_tensor(np.asarray(target), dtype=compute_dtype, device=x1.device)
    chunk_axis = max(range(x1.dim()), key=lambda i: x1.shape[i])
    cols = x1.shape[chunk_axis]
    total = torch.zeros((), dtype=compute_dtype, device=x1.device)
    total_n = 0
    for s in range(0, cols, chunk_cols):
        e = min(s + chunk_cols, cols)
        idx = [slice(None)] * x1.dim()
        idx[chunk_axis] = slice(s, e)
        x1b = x1[tuple(idx)]
        # target 若在对应轴上有同大小切片，则同步切片（如 c_s_023 的 target 尾维）
        tc = target.dim() - (x1.dim() - chunk_axis)
        if target.dim() > 1 and 0 <= tc < target.dim() and target.shape[tc] == cols:
            tidx = [slice(None)] * target.dim()
            tidx[tc] = slice(s, e)
            targ_b = target[tuple(tidx)]
        else:
            targ_b = target
        b1b, b2b = torch.broadcast_tensors(x1b, x2)
        dot_b = torch.sum(b1b * b2b, dim=1, dtype=compute_dtype)
        s1_b = torch.sum(b1b * b1b, dim=1, dtype=compute_dtype)
        s2_b = torch.sum(b2b * b2b, dim=1, dtype=compute_dtype)
        eps = torch.tensor(EPS, dtype=compute_dtype)
        denom_b = torch.sqrt(s1_b + eps) * torch.sqrt(s2_b + eps)
        cos_b = dot_b / denom_b
        cos_b, t_b = torch.broadcast_tensors(cos_b, targ_b)
        pos_b = torch.tensor(1.0, dtype=compute_dtype) - cos_b
        neg_b = torch.maximum(
            torch.tensor(0.0, dtype=compute_dtype),
            cos_b - torch.tensor(margin, dtype=compute_dtype),
        )
        loss_b = torch.where(
            t_b == 1.0, pos_b, torch.where(t_b == -1.0, neg_b, torch.zeros_like(pos_b))
        )
        total = total + loss_b.sum()
        total_n += loss_b.numel()
    return total, total_n


def _needs_chunking(x1, x2):
    """是否应分块：广播物化超过阈值，且 x1 为 2 维、x2 多维（超大 shape 典型结构）。"""
    bshape = torch.broadcast_shapes(x1.shape, x2.shape)
    numel = 1
    for s in bshape:
        numel *= s
    return numel > _CHUNK_THRESHOLD and x1.dim() == 2 and x2.dim() > 2


class _CosineEmbeddingLossCompose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, margin=0.0, reduction="mean", **kwargs):
        self.margin, self.reduction = _resolve(kwargs, margin, reduction)

    def __call__(self, x1, x2, target, **kwargs):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        target = target.to(torch.float32)
        # 超大广播 shape 分块计算，避免三方 GPU 物化数百 GB 而 OOM
        if self.reduction != "none" and _needs_chunking(x1, x2):
            total, total_n = _chunked_loss(
                x1, x2, target, self.margin, self.reduction, torch.float32
            )
            if self.reduction == "sum":
                return [total.to(torch.float32).reshape(1)]
            n = total_n if total_n > 0 else 1
            return [(total / n).to(torch.float32).reshape(1)]

        x1, x2 = torch.broadcast_tensors(x1, x2)
        dot = torch.sum(x1 * x2, dim=1, dtype=torch.float32)
        s1 = torch.sum(x1 * x1, dim=1, dtype=torch.float32)
        s2 = torch.sum(x2 * x2, dim=1, dtype=torch.float32)
        denom = torch.sqrt(s1 + EPS) * torch.sqrt(s2 + EPS)
        cos = dot / denom
        cos, target = torch.broadcast_tensors(cos, target)
        pos = 1.0 - cos
        neg = torch.maximum(
            torch.zeros((), dtype=torch.float32, device=cos.device),
            cos - self.margin,
        )
        loss = torch.where(
            target == 1.0,
            pos,
            torch.where(target == -1.0, neg, torch.zeros_like(pos)),
        )
        if self.reduction == "none":
            return [loss.to(torch.float32)]
        if self.reduction == "sum":
            return [torch.sum(loss, dtype=torch.float32).reshape(1)]
        denom_n = loss.numel() if loss.numel() > 0 else 1
        return [(torch.sum(loss, dtype=torch.float32) / denom_n).reshape(1)]


class CosineEmbeddingLossKernelSpec:
    golden = _cosine_embedding_loss_spec_golden
    third_party = {"torch": _CosineEmbeddingLossCompose}
    tolerance = _KERNEL_TOLERANCE


# 【不存在】aclnn 通路: CMakeLists.txt 使用 ACLNNTYPE aclnn_exclude.
# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 CosineEmbeddingLoss.
