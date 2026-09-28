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
"""
TTK kernel input generator for CTCLossV2Grad (arch35 / Ascend950).

【为何需要自定义输入生成】
CTCLossV2Grad 的两个输入 neg_log_likelihood 与 log_alpha 是 CTC【前向】的中间产物，
彼此以及与 (log_probs, targets, lengths, blank) 之间存在严格数学约束——随机填充会
让反向公式 grad=(exp(lp)-exp(res+nll-lp))*grad_out 中的 `res+nll-lp` 完全失真，
真值与 kernel 都会算出无意义结果，无法形成有效比对。

因此本插件用【同一次合法前向】产生 nll/log_alpha：
    nll, log_alpha = torch.ops.aten._ctc_loss(log_probs, targets, input_lengths,
                                              target_lengths, blank, zero_infinity)
再连同（重新生成为合法值的）log_probs/targets/grad_out 一起，按 def.cpp 顺序返回，
覆盖框架的随机初值，确保 golden 与 kernel 观测到完全一致且自洽的输入。

【确定性】seed = md5(testcase_name)，与 kernel/golden 观测同一组输入、可复现。

【长度来源】input_lengths/target_lengths 经 CSV `attributes` 以确定 tuple 注入
（pickup_by_names），抵达本插件时已是确定值，直接采用；不再随机。
"""

import hashlib
import numpy as np

__input__ = {"kernel": {"ctc_loss_v2_grad": "ctc_loss_v2_grad_input"}}


def _seed_from_name(name):
    return int(hashlib.md5(str(name).encode()).hexdigest(), 16) % (2**32)


def _int_list(arr):
    return [int(x) for x in np.asarray(arr).reshape(-1).tolist()]


def ctc_loss_v2_grad_input(
    grad_out,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    *,
    blank=0,
    reduction="mean",
    zero_infinity=False,
    **kwargs,
):
    """按 def.cpp 顺序生成 CTCLossV2Grad 的 7 个输入，保证 nll/log_alpha 与前向自洽。

    返回值 dtype/shape 必须与 CSV input_shapes/input_dtypes 完全一致：
        grad_out(N,), log_probs(T,N,C), targets(N,S), input_lengths(N,),
        target_lengths(N,), neg_log_likelihood(N,), log_alpha(N,T,2*S+1)
    其中 S = targets.shape[1] = max(target_lengths)。
    """
    import torch

    name = kwargs.get("testcase_name", "ctc_loss_v2_grad")
    rng = np.random.default_rng(_seed_from_name(name))

    lp_arr = np.asarray(log_probs)
    tgt_arr = np.asarray(targets)
    T, N, C = int(lp_arr.shape[0]), int(lp_arr.shape[1]), int(lp_arr.shape[2])

    lp_dtype = lp_arr.dtype  # 浮点声明 dtype（fp32/fp16/bf16）
    tgt_dtype = tgt_arr.dtype  # 整型声明 dtype（int32/int64）
    grad_out_dtype = np.asarray(grad_out).dtype
    nll_dtype = np.asarray(neg_log_likelihood).dtype
    la_dtype = np.asarray(log_alpha).dtype

    # ---- 长度：采用注入的确定值；缺省则退化为合法构造 ----
    il = _int_list(input_lengths)
    tl = _int_list(target_lengths)
    if len(il) != N or any(v <= 0 for v in il):
        il = [T] * N
    il = [min(int(v), T) for v in il]  # input_length 不得超过 T

    # targets 形状 (N, S)；S 取列宽（= max(target_lengths)）
    if tgt_arr.ndim == 2:
        S = int(tgt_arr.shape[1])
    else:
        S = int(max(tl)) if tl else 0
    if len(tl) != N:
        tl = [min(S, il_i) for il_i in il]
    tl = [max(0, min(int(v), S)) for v in tl]  # 0 <= target_length <= S

    # ---- log_probs：随机张量做 log_softmax(dim=2) 得到合法对数概率 ----
    raw = rng.standard_normal((T, N, C)).astype(np.float32)
    lp_t = torch.log_softmax(torch.from_numpy(raw), dim=2)  # (T,N,C) fp32

    # ---- targets：在 [0,C)\{blank} 采样；padding 位也填合法值（超出 target_length 不参与） ----
    tgt_2d = np.zeros((N, S), dtype=np.int64)
    if C > 1 and S > 0:
        pool = np.array([c for c in range(C) if c != int(blank)], dtype=np.int64)
        for r in range(N):
            tgt_2d[r, :] = pool[rng.integers(0, len(pool), size=S)]
    elif S > 0:
        # C==1 的极端场景：只有 blank，可用标签为空，填 0
        tgt_2d[:, :] = 0
    tgt_i64 = torch.from_numpy(tgt_2d)

    # ---- 前向：产生与输入自洽的 nll / log_alpha ----
    nll_t, la_t = torch.ops.aten._ctc_loss(
        lp_t, tgt_i64, il, tl, int(blank), bool(zero_infinity)
    )

    # log_alpha 第三维应为 2*S+1；如与声明不符则对齐（一般恒等）
    exp_last = 2 * S + 1
    la_np = la_t.numpy().astype(np.float32)
    if la_np.shape[-1] != exp_last:
        fixed = np.full((N, T, exp_last), -np.inf, dtype=np.float32)
        m = min(exp_last, la_np.shape[-1])
        fixed[:, :, :m] = la_np[:, :, :m]
        la_np = fixed

    # ---- grad_out：逐 batch 非均匀上游梯度 ----
    go_np = rng.uniform(-1.0, 1.0, size=(N,)).astype(np.float32)

    # ---- 按声明 dtype 落地 ----
    def cast(arr, dt):
        return np.asarray(arr).astype(dt, copy=False)

    out_grad_out = cast(go_np, grad_out_dtype)
    out_log_probs = cast(lp_t.numpy(), lp_dtype)
    out_targets = cast(tgt_2d, tgt_dtype)
    out_input_lengths = cast(np.asarray(il), np.asarray(input_lengths).dtype)
    out_target_lengths = cast(np.asarray(tl), np.asarray(target_lengths).dtype)
    out_nll = cast(nll_t.numpy(), nll_dtype)
    out_log_alpha = cast(la_np, la_dtype)

    return [
        out_grad_out,
        out_log_probs,
        out_targets,
        out_input_lengths,
        out_target_lengths,
        out_nll,
        out_log_alpha,
    ]
