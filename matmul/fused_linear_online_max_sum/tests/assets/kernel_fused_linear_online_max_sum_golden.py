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
import torch

try:
    from ml_dtypes import bfloat16 as ml_bfloat16
except ImportError:
    ml_bfloat16 = None


def _to_numpy(tensor):
    if tensor.dtype == torch.bfloat16 and ml_bfloat16 is not None:
        return tensor.view(torch.int16).numpy().view(ml_bfloat16)
    return tensor.numpy()


def _from_numpy(arr):
    if ml_bfloat16 is not None and arr.dtype == ml_bfloat16:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


# 精度规则：
# 1. golden：FP16/BF16 浮点输入在计算前全部升精度到 FP32（matmul 前完成），
#    后续 max/sub/exp/sum 全程 FP32；output[5] 仅在产出时降回输入 dtype。
# Spec.tolerance 按 dtype 路由：浮点输出运行时以 --compare close 走 isclose
# （atol+rtol 混合容差：|a-g|<=atol+rtol*|g|，rtol 按 dtype 内置 fp32=1e-4、
# fp16/bf16=1e-3，atol=1e-8，允许 0.1% 元素超差）；Spec.tolerance 的
# stat_rel_err 为未带 CLI 时的兜底标准；int/uint -> binary_equal。


__spec__ = {"fused_linear_online_max_sum": "FusedLinearOnlineMaxSumKernelSpec"}


class FusedLinearOnlineMaxSumKernelSpec:
    @staticmethod
    def golden(
        input,
        weight,
        target,
        vocab_start_index=0,
        vocab_end_index=0,
        vocab_parallel_logits_out_flag=False,
        **kwargs,
    ):
        input_t = _from_numpy(input)
        weight_t = _from_numpy(weight)
        target_t = _from_numpy(target)
        vpl = torch.matmul(input_t.to(torch.float32), weight_t.t().to(torch.float32))
        logits_max_local = torch.max(vpl, dim=-1)[0]
        logits_sub = vpl - logits_max_local.unsqueeze(dim=-1)
        target_mask = (target_t < vocab_start_index) | (target_t >= vocab_end_index)
        masked_target = target_t.clone() - vocab_start_index
        masked_target[target_mask] = 0
        bt = target_t.shape[0]
        pad_num = (bt + 7) // 8 * 8 - bt
        if pad_num > 0:
            target_pad = torch.nn.functional.pad(target_t, (0, pad_num), "constant", 0)
            target_mask_pad = (target_pad < vocab_start_index) | (
                target_pad >= vocab_end_index
            )
        else:
            target_mask_pad = target_mask.clone()
        tm_uint8 = target_mask_pad.to(torch.uint8).cpu().numpy().reshape([-1, 8])
        tm_flip = np.flip(tm_uint8, axis=1)
        target_mask_out = np.packbits(tm_flip, axis=1).reshape(-1)
        vocab_width = vpl.shape[-1]
        logits_2d = logits_sub.view(-1, vocab_width)
        gather_index = masked_target.reshape(-1).to(torch.int64).unsqueeze(1)
        predicted_logits_1d = torch.gather(logits_2d, 1, gather_index).squeeze(1)
        predicted_logits = predicted_logits_1d.reshape(target_t.shape)
        predicted_logits[target_mask] = 0.0
        sum_exp_logits = logits_sub.exp().sum(dim=-1)
        results = [
            logits_max_local.numpy().astype(np.float32),
            sum_exp_logits.numpy().astype(np.float32),
            predicted_logits.numpy().astype(np.float32),
            target_mask_out.astype(np.uint8),
            masked_target.numpy(),
        ]
        if vocab_parallel_logits_out_flag:
            results.append(_to_numpy(vpl.to(input_t.dtype)))
        else:
            results.append(np.array([], dtype=np.float32))
        return results

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "uint8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
