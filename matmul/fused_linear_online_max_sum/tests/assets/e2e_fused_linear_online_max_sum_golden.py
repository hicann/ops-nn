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
"""E2E golden spec for torch_npu.fused_linear_online_max_sum.

E2E 路径：入参为 torch.Tensor（CPU），返回 torch.Tensor 列表。
框架 (_to_numpy_result) 负责将 torch.Tensor 转为 numpy 供精度比对。
"""

import torch

# 精度规则：
# 1. golden：FP16/BF16 浮点输入在计算前全部升精度到 FP32（matmul 前完成），
#    后续 max/sub/exp/sum 全程 FP32；output[5] 仅在产出时降回输入 dtype。
# 2. e2e 属性为 snake_case，output_logits 取 vocab_parallel_logits_out_flag 或
#    return_logits 之或。
# Spec.tolerance 按 dtype 路由：浮点输出运行时以 --compare close 走 isclose
# （atol+rtol 混合容差：|a-g|<=atol+rtol*|g|，rtol 按 dtype 内置 fp32=1e-4、
# fp16/bf16=1e-3，atol=1e-8，允许 0.1% 元素超差）；Spec.tolerance 的
# stat_rel_err 为未带 CLI 时的兜底标准；int/uint -> binary_equal。


__spec__ = {"torch_npu.fused_linear_online_max_sum": "E2EFusedLinearOnlineMaxSumSpec"}


class E2EFusedLinearOnlineMaxSumSpec:
    """E2E 流程 golden — 收到 torch.Tensor，返回 torch.Tensor 列表。"""

    @staticmethod
    def golden(
        input,
        weight,
        target,
        vocab_start_index=0,
        vocab_end_index=0,
        vocab_parallel_logits_out_flag=False,
        return_logits=False,
        **kwargs,
    ):
        output_logits = vocab_parallel_logits_out_flag or return_logits
        # matmul 在 float32 下计算，避免 fp16 精度损失
        vpl = torch.matmul(input.to(torch.float32), weight.t().to(torch.float32))
        logits_max_local = torch.max(vpl, dim=-1)[0]
        logits_sub = vpl - logits_max_local.unsqueeze(dim=-1)

        # target mask: True 表示 target 不在 [vocab_start, vocab_end) 范围内
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # pad target 到 8 的倍数，便于 packbits
        bt = target.shape[0]
        pad_num = (bt + 7) // 8 * 8 - bt
        if pad_num > 0:
            target_pad = torch.nn.functional.pad(target, (0, pad_num), "constant", 0)
            target_mask_pad = (target_pad < vocab_start_index) | (
                target_pad >= vocab_end_index
            )
        else:
            target_mask_pad = target_mask.clone()

        # 每 8 个 mask 位打包为 1 字节：位 i 乘权重 2^i，首元素为 LSB
        tm = target_mask_pad.to(torch.uint8).reshape(-1, 8)
        weights = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=tm.device
        )
        target_mask_out = (tm * weights).sum(dim=1).to(torch.uint8)

        # gather predicted logits
        vocab_width = vpl.shape[-1]
        logits_2d = logits_sub.view(-1, vocab_width)
        gather_index = masked_target.reshape(-1).to(torch.int64).unsqueeze(1)
        predicted_logits_1d = torch.gather(logits_2d, 1, gather_index).squeeze(1)
        predicted_logits = predicted_logits_1d.reshape(target.shape)
        predicted_logits[target_mask] = 0.0
        sum_exp_logits = logits_sub.exp().sum(dim=-1)

        results = [
            logits_max_local.to(torch.float32),
            sum_exp_logits.to(torch.float32),
            predicted_logits.to(torch.float32),
            target_mask_out,
            masked_target,
        ]
        if output_logits:
            results.append(vpl.to(input.dtype))
        else:
            # optional 输出未请求：golden 返回 None 占位 -> 框架比对按 SUPPRESSED 放行
            # （返回空 tensor 会与 NPU 侧 None 槽位形成 NO_OUTPUT 误判）
            results.append(None)
        return results

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "uint8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
