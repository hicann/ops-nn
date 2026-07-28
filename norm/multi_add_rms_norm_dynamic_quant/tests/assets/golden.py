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

__golden__ = {
    "kernel": {
        "multi_add_rms_norm_dynamic_quant": "multi_add_rms_norm_dynamic_quant_golden"
    }
}


def _to_torch(arr):
    """numpy(含 ml_dtypes.bfloat16) -> torch tensor(原 dtype)。"""
    name = arr.dtype.name
    if name == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16).view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _from_torch(t, np_dtype_name):
    if np_dtype_name == "bfloat16":
        import ml_dtypes

        return t.view(torch.int16).numpy().view(np.uint16).view(ml_dtypes.bfloat16)
    return t.numpy()


def multi_add_rms_norm_dynamic_quant_golden(
    x1,
    x2,
    gamma,
    smooth_scale1=None,
    smooth_scale2=None,
    epsilon: float = 1e-6,
    **kwargs,
):
    """
    Golden for multi_add_rms_norm_dynamic_quant (arch35).
    参数顺序遵循 def(输入 + attr,无输出)。x1 为 TensorList => list[np.ndarray]。
    语义:
      x = (x1[0]+x2) + x1[1] + ...     (全程 fp32 累加,A2 顺序:x2 第二个加;x 输出 cast 回 dtype)
      rstd = 1/sqrt(mean(x^2)+eps)     (fp32,末轴)
      normed = x * rstd * gamma        (fp32) => y 输出(cast dtype)
      对每个 smooth_scale_j:
        q = normed * smooth_j
        scale_j = max(|q|)/127         (逐行,fp32)
        y_j = round_half_even(q/scale_j) -> int8
      无 smooth 时 q = normed。
    返回 (y1, y2, x, y, scale1, scale2),对齐 def 输出顺序。
    """
    in_dtype_name = x1[0].dtype.name if isinstance(x1, (list, tuple)) else x1.dtype.name

    x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]
    x1_t = [_to_torch(a) for a in x1_list]
    x2_t = _to_torch(x2)
    gamma_t = _to_torch(gamma)

    # A2 权威语义(对齐 canndev normal/single_row_kernel:逐个 Cast fp32 再 Add,全程 fp32):
    # 注释"1.将x1的第一个tensor和x2相加 2.将x1剩余的相加" => x = (x1[0]+x2) + x1[1]+...+x1[n-1],x2 第二个加。
    # (fp32 加法足够精确、顺序不影响末位 cast;此处照 A2 顺序书写以忠实对齐。)
    x_fp32 = x1_t[0].to(torch.float32) + x2_t.to(torch.float32)  # x1[0] + x2
    for t in x1_t[1:]:
        x_fp32 = x_fp32 + t.to(torch.float32)  # += x1[i]
    x_out_t = x_fp32.to(x1_t[0].dtype)

    var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + float(epsilon))
    normed = x_fp32 * rstd * gamma_t.to(torch.float32)  # y 输出(cast前)
    y_out_t = normed.to(x1_t[0].dtype)

    def quant(smooth):
        if smooth is not None:
            q = normed * _to_torch(smooth).to(torch.float32)
        else:
            q = normed
        scale = q.abs().amax(dim=-1, keepdim=True) / 127.0
        yq = torch.round(q / scale)  # round-half-to-even,对齐 CAST_RINT
        yq = torch.clamp(yq, -128, 127).to(torch.int8)
        return yq, scale.squeeze(-1).to(torch.float32)

    y1_t, scale1_t = quant(smooth_scale1)
    y1 = y1_t.numpy()
    scale1 = scale1_t.numpy()
    x = _from_torch(x_out_t, in_dtype_name)
    y = _from_torch(y_out_t, in_dtype_name)
    # y2/scale2 有效 <=> smooth_scale2 存在(smoothNum==2)。对齐 A2/arch35 内核(hasY2Scale2_=hasSmoothScale2_,
    # smoothNum<2 时内核不写 y2/scale2、留 buf 初值)。故 smoothNum<2 返回 None 哨兵 —— TTK 比对会跳过它们
    # (core_modules/npu/op/comparison.py: None/str 哨兵 continue 跳过),避免拿内核未写的垃圾值误判 FAIL。
    if smooth_scale2 is not None:
        y2_t, scale2_t = quant(smooth_scale2)
        y2 = y2_t.numpy()
        scale2 = scale2_t.numpy()
    else:
        y2, scale2 = None, None
    return y1, y2, x, y, scale1, scale2


if __name__ == "__main__":
    # 自测:小例

    np.random.seed(0)
    N, D, k = 4, 64, 3
    x1 = [np.random.randn(N, D).astype(np.float16) for _ in range(k)]
    x2 = np.random.randn(N, D).astype(np.float16)
    gamma = np.random.randn(D).astype(np.float16)
    s1 = np.abs(np.random.randn(D)).astype(np.float16) + 0.5
    s2 = np.abs(np.random.randn(D)).astype(np.float16) + 0.5
    y1, y2, x, y, sc1, sc2 = multi_add_rms_norm_dynamic_quant_golden(
        x1, x2, gamma, s1, s2, epsilon=1e-5
    )
    print("y1", y1.shape, y1.dtype, "range", y1.min(), y1.max())
    print("x", x.shape, x.dtype, "y", y.shape, "scale1", sc1.shape, sc1.dtype, sc1[:2])
    assert y1.dtype == np.int8 and sc1.dtype == np.float32
    print("SELF-TEST OK")
