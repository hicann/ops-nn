#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Golden reference for sigmoid_cross_entropy_with_logits_grad.

gradient = (sigmoid(predict) - target) * dout

Kernel 语义：非 float32 输入先提升到 float32 计算，结果用 CAST_RINT 转回原 dtype；
fp16/bf16 输入在比较前完成 dtype 往返，float32 输入保持原 dtype。
"""

import numpy as np

__spec__ = {
    "sigmoid_cross_entropy_with_logits_grad": "SigmoidCrossEntropyWithLogitsGradKernelSpec",
}

_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_BF16_NUMPY = None
try:
    import ml_dtypes

    _BF16_NUMPY = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover - TTK 环境必然提供
    _BF16_NUMPY = None


def _is_bfloat16(dtype):
    if _BF16_NUMPY is not None and dtype == _BF16_NUMPY:
        return True
    return str(dtype) in ("bfloat16", "<V2")


def _to_float32(arr):
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32)


def _cast_back(arr, orig_dtype):
    if orig_dtype == np.float32:
        return arr
    if orig_dtype == np.float16:
        return arr.astype(np.float16)
    if _is_bfloat16(orig_dtype) and _BF16_NUMPY is not None:
        return arr.astype(_BF16_NUMPY)
    return arr


class SigmoidCrossEntropyWithLogitsGradKernelSpec:
    """Kernel/GEIR 共用 TestSpec（CSV op_name = sigmoid_cross_entropy_with_logits_grad）。

    输入按 def.cpp 顺序位置传入（predict, target, dout），无属性、无可选输入。
    golden 接收 numpy，返回 numpy list。
    """

    @staticmethod
    def golden(predict, target, dout, **kwargs):
        import torch

        orig_dtype = predict.dtype
        p = torch.from_numpy(_to_float32(predict))
        t = torch.from_numpy(_to_float32(target))
        d = torch.from_numpy(_to_float32(dout))
        # dout 只需元素总数与 predict 相等，shape 可能不同；reshape 到 predict shape 后逐元素计算
        d = d.reshape(p.shape)
        grad = (torch.sigmoid(p) - t) * d
        return [_cast_back(grad.numpy(), orig_dtype)]

    class ThirdPartyImpl:
        """torch provider：独立 Torch 基础算子组合实现，供 GEIR cross_check。"""

        def __init__(self, **kwargs):
            pass

        def __call__(self, predict, target, dout, **kwargs):
            import torch

            orig_dtype = predict.dtype
            p = predict.float() if predict.dtype != torch.float32 else predict
            t = target.float() if target.dtype != torch.float32 else target
            d = dout.float() if dout.dtype != torch.float32 else dout
            # dout 只需元素总数与 predict 相等，shape 可能不同；reshape 到 predict shape 后逐元素计算
            d = d.reshape(p.shape)
            grad = (torch.sigmoid(p) - t) * d
            if orig_dtype == torch.float16:
                grad = grad.to(torch.float16)
            elif orig_dtype == torch.bfloat16:
                grad = grad.to(torch.bfloat16)
            return [grad]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = _TOLERANCE
