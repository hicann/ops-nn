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

# 精度规则：
# 1. golden：FP16/BF16 浮点输入在计算前全部升精度到 FP32（matmul 前完成），
#    后续 max/sub/exp/sum 全程 FP32；output[5] 仅在产出时降回输入 dtype。
# 2. aclnn 属性为 camelCase（vocabStartIndex/vocabEndIndex），且拿不到 optional
#    输出槽位信息，恒产出 output[5]（flag=False 用例该输出不参与比对，无副作用）。
# Spec.tolerance 按 dtype 路由：浮点输出运行时以 --compare close 走 isclose
# （atol+rtol 混合容差：|a-g|<=atol+rtol*|g|，rtol 按 dtype 内置 fp32=1e-4、
# fp16/bf16=1e-3，atol=1e-8，允许 0.1% 元素超差）；Spec.tolerance 的
# stat_rel_err 为未带 CLI 时的兜底标准；int/uint -> binary_equal。


__spec__ = {"aclnnFusedLinearOnlineMaxSum": "FusedLinearOnlineMaxSumSpec"}


class FusedLinearOnlineMaxSumSpec:
    @staticmethod
    def golden(
        input,
        weight,
        target,
        vocabStartIndex=0,
        vocabEndIndex=0,
        logitsMaxLocalOut=None,
        sumExpLogitsLocalOut=None,
        predictedLogitsLocalOut=None,
        targetMaskOut=None,
        maskedTargetOut=None,
        vocabParallelLogitsOutOptional=None,
        **kwargs,
    ):
        vpl = torch.matmul(input.to(torch.float32), weight.t().to(torch.float32))
        logits_max_local = torch.max(vpl, dim=-1)[0]
        logits_sub = vpl - logits_max_local.unsqueeze(dim=-1)
        target_mask = (target < vocabStartIndex) | (target >= vocabEndIndex)
        masked_target = target.clone() - vocabStartIndex
        masked_target[target_mask] = 0
        bt = target.shape[0]
        pad_num = (bt + 7) // 8 * 8 - bt
        if pad_num > 0:
            target_pad = torch.nn.functional.pad(target, (0, pad_num), "constant", 0)
            target_mask_pad = (target_pad < vocabStartIndex) | (
                target_pad >= vocabEndIndex
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
        predicted_logits = predicted_logits_1d.reshape(target.shape)
        predicted_logits[target_mask] = 0.0
        sum_exp_logits = logits_sub.exp().sum(dim=-1)

        results = [
            logits_max_local,
            sum_exp_logits,
            predicted_logits,
            torch.from_numpy(target_mask_out.astype(np.uint8)),
            masked_target,
        ]
        if vocabParallelLogitsOutOptional is not None:
            results.append(vpl.to(input.dtype))
        return results

    @staticmethod
    def customize_inputs(
        input,
        weight,
        target,
        vocabStartIndex=0,
        vocabEndIndex=0,
        logitsMaxLocalOut=None,
        sumExpLogitsLocalOut=None,
        predictedLogitsLocalOut=None,
        targetMaskOut=None,
        maskedTargetOut=None,
        vocabParallelLogitsOutOptional=None,
        **kwargs,
    ):
        tensor_dtypes = kwargs.get("tensor_dtypes", ())
        if len(tensor_dtypes) >= 2:
            import torch

            dt_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            dt0 = str(tensor_dtypes[0])
            str(tensor_dtypes[1])
            if dt0 in dt_map:
                input = input.to(dt_map[dt0])
                weight = weight.to(dt_map[dt0])
            if dt0 == "bfloat16" and vocabParallelLogitsOutOptional is not None:
                vocabParallelLogitsOutOptional = vocabParallelLogitsOutOptional.to(
                    torch.bfloat16
                )
        return (
            input,
            weight,
            target,
            vocabStartIndex,
            vocabEndIndex,
            logitsMaxLocalOut,
            sumExpLogitsLocalOut,
            predictedLogitsLocalOut,
            targetMaskOut,
            maskedTargetOut,
            vocabParallelLogitsOutOptional,
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "uint8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
