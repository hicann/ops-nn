#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""embedding_bag 双标杆 golden（kernel/aclnn/e2e 路径统一）。

Kernel 路径的 golden 收到 numpy.ndarray，转 torch tensor 后用 torch.ops.aten._embedding_bag
计算，结果转回 numpy；ACLNN/E2E 路径的 golden 直接收到 torch.Tensor（已在设备上），无需转换。
入参按 def 定义原样送入 torch，不做额外兜底转换，非法输入直接暴露；仅 sum 模式支持
per_sample_weight（mean/max 模式下按算子语义忽略）；scale_grad_by_freq/sparse 只影响反向，
golden 固定为 False；indices 为 2D 时按行分 bag（offsets 等价于 arange 步长 W）；
1D offsets 原样透传（与算子消费同一张量），合法性构造（排序 + 起点 0）由本文件
`__input__` 注册的 embedding_bag_input 在输入层统一完成，保证 NPU 与 golden 输入一致。
CPU 参考无跨实现契约的辅助输出由 golden 返回 None（框架 SUPPRESSED 机制，比对跳过该
输出、不伪造参考值）：sum fast path 的 offset2bag 为空张量（CUDA/算子恒产全长）；
sum 的 bag_size CPU 惯例恒零（CUDA/算子为非 pad 计数，零锚点会假通过垃圾值）；
非 max 模式的 max_indices（CPU=bag_size 拷贝、CUDA=空张量，三方各说各话）。
"""

__spec__ = {
    "embedding_bag": "EmbeddingBagKernelTestSpec",
    "aclnnEmbeddingBag": "AclnnEmbeddingBagTestSpec",
    "torch.ops.aten._embedding_bag": "TorchEmbeddingBagTestSpec",
    "torch.ops.aten.embedding_bag": "TorchEmbeddingBagTestSpec",
}

__input__ = {"kernel": {"embedding_bag": "embedding_bag_input"}}

import numpy as np


def embedding_bag_input(
    weight,
    indices,
    offsets,
    per_sample_weight=None,
    *,
    mode="mean",
    scale_grad_by_freq=False,
    sparse=False,
    include_last_offset=False,
    padding_idx=-1,
    **kwargs,
):
    """
    Input function for embedding_bag（cross-check 流程的输入构造，注册于本文件 __input__）。
    仅保留 1D offsets 合法性构造：排序 + 起点 0；2D offsets 透传
    （算子与 golden 对 2D 均不读 offsets）。
    All the parameters (names and order) follow @embedding_bag_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    if indices.ndim == 1:
        offsets = np.sort(offsets)
        offsets[0] = 0
    return [weight, indices, offsets, per_sample_weight]


_MODE_STR_TO_INT = {"sum": 0, "mean": 1, "max": 2}
_MODE_INT_TO_STR = {v: k for k, v in _MODE_STR_TO_INT.items()}


def _norm_mode(mode):
    """kernel def 侧 mode 为字符串属性, aclnn/torch dispatcher 侧为 int, 统一为 (str, int)。"""
    if isinstance(mode, str):
        s = mode.strip().lower()
        if s not in _MODE_STR_TO_INT:
            raise RuntimeError("embedding_bag golden unsupported mode: %r" % (mode,))
        return s, _MODE_STR_TO_INT[s]
    i = int(mode)
    if i not in _MODE_INT_TO_STR:
        raise RuntimeError("embedding_bag golden unsupported mode: %r" % (mode,))
    return _MODE_INT_TO_STR[i], i


def _as_torch(x):
    """numpy.ndarray 转 torch.Tensor（bfloat16 以 int16 视图承载），torch.Tensor 原样返回。"""
    if x is None:
        return None
    import torch

    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray) and "bfloat16" in str(x.dtype):
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.asarray(x))


def _to_numpy_like(t, dtype_like):
    """torch.Tensor 转回 numpy，bfloat16 以 int16 视图还原为入参的 dtype 描述。"""
    import torch

    if "bfloat16" in str(dtype_like):
        return t.view(torch.int16).numpy().view(np.dtype(dtype_like))
    return t.numpy()


def _embedding_bag_torch(
    weight, indices, offsets, per_sample_weights, mode, include_last_offset, padding_idx
):
    """统一参考实现：入参/出参均为 torch.Tensor，返回 (y, offset2bag, bag_size, max_indices)。

    全部输出取自同一条 torch.ops.aten._embedding_bag 调用，保证 y 与辅助输出口径一致。
    - 2D indices：按行分 bag，展平 indices/psw 并以 arange(0, numel, W) 生成 offsets
      （dtype 跟随 indices，同官方 F.embedding_bag）；算子侧 2D 不读 offsets 且忽略 ilo，
      故强制 include_last_offset=False，两侧恒为 N 个行 bag。
    - 1D indices：offsets 原样透传，与算子消费同一张量；排序 + 起点 0 的合法性构造
      由 __input__ 注册的 embedding_bag_input 在输入层完成（NPU 与 golden 同源）。
    - 仅 sum 模式支持 per_sample_weights；indices/offsets 的 int32/int64 混用由 torch
      原生 promote 处理（与 arch35 infershape 的 aux dtype 规则等价）。
    """
    import torch

    mode_str, mode_int = _norm_mode(mode)
    if mode_str != "sum":
        per_sample_weights = None

    indices_flat = indices.reshape(-1)
    if indices.dim() == 2:
        offsets_flat = torch.arange(
            0,
            indices.numel(),
            indices.shape[1],
            dtype=indices_flat.dtype,
            device=indices_flat.device,
        )
        include_last_offset = False
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    else:
        offsets_flat = offsets

    y, offset2bag, bag_size, max_indices = torch.ops.aten._embedding_bag(
        weight,
        indices_flat,
        offsets_flat,
        False,
        mode_int,
        False,
        per_sample_weights,
        bool(include_last_offset),
        int(padding_idx),
    )
    return y, offset2bag, bag_size, max_indices


class EmbeddingBagKernelTestSpec:
    """embedding_bag kernel 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow embedding_bag_def.cpp:
    weight, indices, offsets, per_sample_weight(optional),
    mode("sum"/"mean"/"max"), scale_grad_by_freq, sparse, include_last_offset, padding_idx.
    """

    def golden(
        self,
        weight,
        indices,
        offsets,
        per_sample_weight=None,
        *,
        mode="mean",
        scale_grad_by_freq=False,
        sparse=False,
        include_last_offset=False,
        padding_idx=-1,
        **kwargs,
    ):
        y, offset2bag, bag_size, max_indices = _embedding_bag_torch(
            _as_torch(weight),
            _as_torch(indices),
            _as_torch(offsets),
            _as_torch(per_sample_weight),
            mode,
            include_last_offset,
            padding_idx,
        )

        if "bfloat16" in str(weight.dtype):
            out_np = _to_numpy_like(y, weight.dtype)
        else:
            out_np = y.numpy()
        return [out_np, offset2bag.numpy(), bag_size.numpy(), max_indices.numpy()]

    class ThirdPartyImpl:
        def __call__(
            self,
            weight,
            indices,
            offsets,
            per_sample_weight=None,
            *,
            mode="mean",
            scale_grad_by_freq=False,
            sparse=False,
            include_last_offset=False,
            padding_idx=-1,
            **kwargs,
        ):
            y, offset2bag, bag_size, max_indices = _embedding_bag_torch(
                _as_torch(weight),
                _as_torch(indices),
                _as_torch(offsets),
                _as_torch(per_sample_weight),
                mode,
                include_last_offset,
                padding_idx,
            )
            return [y, offset2bag, bag_size, max_indices]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class AclnnEmbeddingBagTestSpec:
    """aclnnEmbeddingBag 测试规范（aclnn 流程，torch 入参，已在设备上）

    Parameters follow aclnnEmbeddingBagGetWorkspaceSize:
    weight, indices, offsets, scaleGradByFreq, mode(int), sparse,
    perSampleWeights, includeLastOffset, paddingIdx.
    """

    def golden(
        self,
        weight,
        indices,
        offsets,
        scaleGradByFreq,
        mode,
        sparse,
        perSampleWeights,
        includeLastOffset,
        paddingIdx,
        **kwargs,
    ):
        y, offset2bag, bag_size, max_indices = _embedding_bag_torch(
            _as_torch(weight),
            _as_torch(indices),
            _as_torch(offsets),
            _as_torch(perSampleWeights),
            mode,
            includeLastOffset,
            paddingIdx,
        )
        return [y, offset2bag, bag_size, max_indices]

    class ThirdPartyImpl:
        def __call__(
            self,
            weight,
            indices,
            offsets,
            scaleGradByFreq=False,
            mode=1,
            sparse=False,
            perSampleWeights=None,
            includeLastOffset=False,
            paddingIdx=-1,
            **kwargs,
        ):
            y, offset2bag, bag_size, max_indices = _embedding_bag_torch(
                weight,
                indices,
                offsets,
                perSampleWeights,
                mode,
                includeLastOffset,
                paddingIdx,
            )
            return [y, offset2bag, bag_size, max_indices]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class TorchEmbeddingBagTestSpec:
    """torch e2e 测试规范（torch 入参，已在设备上）

    Parameters follow the torch dispatcher schema:
    _embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode(int),
    sparse, per_sample_weights, include_last_offset, padding_idx)
    -> (output, offset2bag, bag_size, max_indices).
    """

    def golden(
        self,
        weight,
        indices,
        offsets,
        scale_grad_by_freq=False,
        mode=1,
        sparse=False,
        per_sample_weights=None,
        include_last_offset=False,
        padding_idx=-1,
        **kwargs,
    ):
        y, offset2bag, bag_size, max_indices = _embedding_bag_torch(
            _as_torch(weight),
            _as_torch(indices),
            _as_torch(offsets),
            _as_torch(per_sample_weights),
            mode,
            include_last_offset,
            padding_idx,
        )
        return [y, offset2bag, bag_size, max_indices]

    class ThirdPartyImpl:
        def __call__(
            self,
            weight,
            indices,
            offsets,
            scale_grad_by_freq=False,
            mode=1,
            sparse=False,
            per_sample_weights=None,
            include_last_offset=False,
            padding_idx=-1,
            **kwargs,
        ):
            y, offset2bag, bag_size, max_indices = _embedding_bag_torch(
                weight,
                indices,
                offsets,
                per_sample_weights,
                mode,
                include_last_offset,
                padding_idx,
            )
            return [y, offset2bag, bag_size, max_indices]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
