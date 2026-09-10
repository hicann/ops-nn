#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""cross_entropy_sum_exp_and_index_logit 算子在三条测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，直接用 numpy 计算后返回 numpy；
ACLNN 的 golden 收到 torch.Tensor（可同时兼容 numpy），先 detach().cpu()
转 numpy 计算后返回 numpy；
torch（E2E）的 golden 收到 torch.Tensor，计算逻辑对齐
pta_test/test_cross_entropy_sum_exp_and_index_logit.py 的 cross_entropy_golden
（CPU FP64 高精度参考，BF16/FP32 输入统一抬到 double 计算后转回输出 dtype），
接口即 torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit。
类形式 golden 返回 list（每个输出一个元素）。

格式遵循 ops-test-kit ttk-how-write-plugin skill / ttk/test_spec 规范：
- __spec__ 显式注册（op_name/api_name -> 类名），loader AST 静态扫描、惰性 exec；
- 参数名与顺序对齐各流程接口输入（不含输出 / workspace 占位）。
"""

import numpy as np


def _to_numpy(t):
    """兼容 torch.Tensor / numpy 数组输入。"""
    if t is None:
        return t
    if hasattr(t, "detach"):
        import torch

        t = t.detach().cpu()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.numpy()
    return np.asarray(t)


def _as_cpu_tensor(t):
    """转成 CPU torch.Tensor：兼容 numpy / torch.Tensor（任意设备）。"""
    import torch

    if isinstance(t, torch.Tensor):
        if t.is_floating_point() and t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.detach().cpu()
    return torch.as_tensor(np.asarray(t))


def _as_python_int(value):
    """标量（python int / torch 标量 tensor）统一转 int。"""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _cross_entropy_numpy_golden(logits, tgt, gmax, s, e):
    """numpy fp32 精度的核心计算，返回 5 元素 list。

    logits: (n, v) float32,  tgt: (n,) int64,  gmax: (n,) float32.
    s, e: int, vocab 区间。
    """
    target_shape = tgt.shape
    logits_shape = logits.shape

    v_local = logits.shape[-1]
    n = logits.size // v_local
    if e == 0:
        e = v_local

    logits_2d = logits.reshape(n, v_local)
    target_1d = tgt.reshape(n)
    gmax_1d = gmax.reshape(n)

    shifted = logits_2d - gmax_1d[:, None]

    target_mask_1d = ((target_1d < s) | (target_1d >= e)).astype(np.int32)

    target_offset_1d = np.where(target_mask_1d.astype(bool), 0, target_1d - s).astype(
        np.int32
    )

    gathered = logits_2d[np.arange(n), target_offset_1d.astype(np.int64)]
    predicted_1d = np.where(
        target_mask_1d.astype(bool), np.float32(0.0), gathered - gmax_1d
    )

    exp_logits = np.exp(shifted)
    sum_exp_1d = exp_logits.sum(axis=-1)

    return [
        predicted_1d.reshape(target_shape).astype(np.float32),
        sum_exp_1d.reshape(target_shape).astype(np.float32),
        exp_logits.reshape(logits_shape).astype(np.float32),
        target_offset_1d.reshape(target_shape).astype(np.int32),
        target_mask_1d.reshape(target_shape).astype(np.int32),
    ]


def _cross_entropy_torch_golden(logits, tgt, gmax, s, e):
    """torch float64 高精度核心计算，返回 5 元素 list。

    logits: (n, v) float64,  tgt: (n,) int64,  gmax: (n,) float64.
    s, e: int, vocab 区间。
    """
    import torch

    target_shape = tgt.shape
    logits_shape = logits.shape

    v_local = logits.shape[-1]
    n = logits.numel() // v_local
    if e == 0:
        e = v_local

    logits_2d = logits.reshape(n, v_local)
    target_1d = tgt.reshape(n)
    gmax_1d = gmax.reshape(n)

    shifted = logits_2d - gmax_1d.unsqueeze(-1)

    target_mask_1d = ((target_1d < s) | (target_1d >= e)).to(torch.int32)

    target_offset_1d = torch.where(
        target_mask_1d.bool(),
        torch.zeros_like(target_1d, dtype=torch.int32),
        (target_1d - s).to(torch.int32),
    )

    gathered = logits_2d[torch.arange(n), target_offset_1d.long()]
    predicted_1d = torch.where(
        target_mask_1d.bool(), torch.zeros_like(gathered), gathered - gmax_1d
    )

    exp_logits = torch.exp(shifted)
    sum_exp_1d = exp_logits.sum(dim=-1)

    return [
        predicted_1d.reshape(target_shape).float(),
        sum_exp_1d.reshape(target_shape).float(),
        exp_logits.reshape(logits_shape).float(),
        target_offset_1d.reshape(target_shape).int(),
        target_mask_1d.reshape(target_shape).int(),
    ]


class CrossEntropySumExpAndIndexLogitKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray。

    参数名与顺序对齐 cross_entropy_sum_exp_and_index_logit_def.cpp 输入（不含输出）。
    """

    @staticmethod
    def golden(
        vocab_parallel_logits,
        target,
        global_logits_max,
        *,
        vocab_start_index=0,
        vocab_end_index=0,
        **kwargs,
    ):
        """
        Parameters follow cross_entropy_sum_exp_and_index_logit_def.cpp without outputs.

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name
        """
        logits = vocab_parallel_logits.astype(np.float32)
        tgt = target.astype(np.int64)
        gmax = global_logits_max.astype(np.float32)
        s = int(vocab_start_index)
        e = int(vocab_end_index)
        return _cross_entropy_numpy_golden(logits, tgt, gmax, s, e)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
    }


class AclnnCrossEntropySumExpAndIndexLogitSpec:
    """ACLNN 流程 — golden 收到 torch.Tensor（已在设备上），兼容 numpy。

    参数名与顺序对齐 aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize
    （不含 workspaceSize 和 executor）：3 个输入张量 + 2 个 int64 标量属性 +
    5 个输出张量占位（框架按该顺序以 10 个位置参数调用，输出占位被忽略）。

    接口：aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize，参数顺序：
      (vocabParallelLogits, target, globalLogitsMax,
       vocabStartIndex, vocabEndIndex,
       predictedLogitsOut, sumExpLogitsOut, expLogitsOut, targetOffsetOut, targetMaskOut)
    """

    @staticmethod
    def golden(
        vocabParallelLogits,
        target,
        globalLogitsMax,
        vocabStartIndex,
        vocabEndIndex,
        predictedLogitsOut,
        sumExpLogitsOut,
        expLogitsOut,
        targetOffsetOut,
        targetMaskOut,
        **kwargs,
    ):
        """
        Parameters follow aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize
        without workspaceSize & executor. The 5 trailing output tensors are
        placeholders and ignored here.

        **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
                 use_torch, short_soc_version, testcase_name
        """
        logits = _to_numpy(vocabParallelLogits).astype(np.float32)
        tgt = _to_numpy(target)
        if tgt.dtype != np.int64:
            tgt = tgt.astype(np.int64)
        gmax = _to_numpy(globalLogitsMax).astype(np.float32)
        s = int(vocabStartIndex)
        e = int(vocabEndIndex)
        return _cross_entropy_numpy_golden(logits, tgt, gmax, s, e)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
    }


class TorchCrossEntropySumExpAndIndexLogitSpec:
    """torch（E2E）流程 — golden 收到 torch.Tensor。

    接口对齐 pta_test/test_cross_entropy_sum_exp_and_index_logit.py：
        torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit(
            vocab_parallel_logits, target, global_logits_max,
            vocab_start_index, vocab_end_index)
        -> (predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask)

    计算逻辑对齐该脚本的 cross_entropy_golden：CPU FP64 高精度参考。
    """

    @staticmethod
    def golden(
        vocab_parallel_logits,
        target,
        global_logits_max,
        vocab_start_index,
        vocab_end_index,
        **kwargs,
    ):
        """
        Parameters follow the torch.ops interface above (no outputs).

        **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
                 use_torch, short_soc_version, testcase_name
        """
        logits = _as_cpu_tensor(vocab_parallel_logits).double()
        tgt = _as_cpu_tensor(target).long()
        gmax = _as_cpu_tensor(global_logits_max).double()
        s = _as_python_int(vocab_start_index)
        e = _as_python_int(vocab_end_index)
        return _cross_entropy_torch_golden(logits, tgt, gmax, s, e)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
    }


__spec__ = {
    "cross_entropy_sum_exp_and_index_logit": "CrossEntropySumExpAndIndexLogitKernelSpec",
    "aclnnCrossEntropySumExpAndIndexLogit": "AclnnCrossEntropySumExpAndIndexLogitSpec",
    "torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit": "TorchCrossEntropySumExpAndIndexLogitSpec",
}
