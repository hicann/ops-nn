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

"""Kernel golden：FP16/BF16 浮点输入在计算前升精度到 FP32，后续全程 FP32；
output[5] 仅在产出时降回输入 dtype。
"""

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


__spec__ = {"fused_linear_online_max_sum": "FusedLinearOnlineMaxSumKernelSpec"}


class FusedLinearOnlineMaxSumTorchThirdParty:
    """GPU 侧分解式 torch 参考实现（cross_check 三方腿，远端 XPU server 执行）。"""

    def __call__(
        self,
        input,
        weight,
        target,
        vocab_start_index=0,
        vocab_end_index=0,
        vocab_parallel_logits_out_flag=False,
        device=None,
        **kwargs,
    ):
        vpl = torch.matmul(input.to(torch.float32), weight.t().to(torch.float32))
        logits_max_local = torch.max(vpl, dim=-1)[0]
        logits_sub = vpl - logits_max_local.unsqueeze(dim=-1)
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0
        bt = target.shape[0]
        pad_num = (bt + 7) // 8 * 8 - bt
        if pad_num > 0:
            target_pad = torch.nn.functional.pad(target, (0, pad_num), "constant", 0)
            target_mask_pad = (target_pad < vocab_start_index) | (
                target_pad >= vocab_end_index
            )
        else:
            target_mask_pad = target_mask
        tm = target_mask_pad.to(torch.uint8).reshape(-1, 8)
        bit_weights = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=tm.device
        )
        target_mask_out = (tm * bit_weights).sum(dim=1).to(torch.uint8)
        vocab_width = vpl.shape[-1]
        logits_2d = logits_sub.view(-1, vocab_width)
        gather_index = masked_target.reshape(-1).to(torch.int64).unsqueeze(1)
        predicted_logits = (
            torch.gather(logits_2d, 1, gather_index).squeeze(1).reshape(target.shape)
        )
        predicted_logits[target_mask] = 0.0
        sum_exp_logits = logits_sub.exp().sum(dim=-1)
        results = [
            logits_max_local.to(torch.float32),
            sum_exp_logits.to(torch.float32),
            predicted_logits.to(torch.float32),
            target_mask_out,
            masked_target,
        ]
        if vocab_parallel_logits_out_flag:
            results.append(vpl.to(input.dtype))
        else:
            results.append(torch.empty(0, dtype=torch.float32))
        return results


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

    @staticmethod
    def input(
        input,
        weight,
        target,
        vocab_start_index=0,
        vocab_end_index=0,
        vocab_parallel_logits_out_flag=False,
        **kwargs,
    ):
        # L0_008（接近数相减）：全 1 输入配 weight[1,0]=1.0078125（bf16 可精确表示），
        # 锁定 matmul 的 FP32 计算路径——若提前降精度，max/sub 之差会被舍入吞掉。
        if kwargs.get("testcase_name") == "L0_008":
            input = np.ones_like(input)
            weight = np.ones_like(weight)
            weight[1, 0] = 1.0078125
            target = np.zeros_like(target)
        return (input, weight, target)

    third_party = {"torch": FusedLinearOnlineMaxSumTorchThirdParty}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "uint8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
