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
Golden function for ActULQClampMaxGrad operator.

同时作为 TTK kernel 模式的外部 golden 插件使用：
    python3 -m ttk kernel -i xxx.csv --plugin golden.py
TTK 靠 AST 静态解析顶层 __golden__ 字典（键与 CSV 的 op_name 一致），
并以 golden_func(*context.input_arrays, **kwargs) 方式调用入口函数。
"""

# Standard Packages
from typing import Tuple

# Third-Party Packages
import numpy as np
import torch

# TTK 外部插件注册：op_name -> 本文件入口函数名（绝对 import only，勿加相对 import）
__golden__ = {"kernel": {"act_ulq_clamp_max_grad": "ttk_act_ulq_clamp_max_grad_golden"}}


def act_ulq_clamp_max_grad_golden(
    y_grad: np.ndarray, clamp_max_mask: np.ndarray, x_clamped_loss: np.ndarray
) -> Tuple[np.ndarray]:
    """
    ActULQClampMaxGrad Golden 实现（torch 小算子拼接）

    用于 ULQ 量化感知训练中，计算 clamp 上界截断的反向梯度。

    公式（动态 kernel，实际生效路径）：
        mask' = cast(clamp_max_mask, dtype(y_grad))
        signal = |mask'|              # 对 0/1 输入等价于 mask'
        x_max_grad = x_clamped_loss + signal
        clamp_max_grad = sum(y_grad * x_max_grad)   # 全轴求和，标量输出

    torch 小算子映射（对应 TBE DSL 指令）：
        tbe.cast_to  → x.to(torch_dtype)
        tbe.vabs     → torch.abs
        tbe.vadd     → torch.add
        tbe.vmul     → torch.mul
        tbe.reduce_sum → torch.sum（全轴）

    Args:
        y_grad: 来自后续层的梯度张量，dtype=float16/float32
        clamp_max_mask: clamp 上界掩码，dtype=float16/float32/bool
        x_clamped_loss: 经过 clamp 后的损失值，dtype=float16/float32

    Returns:
        Tuple[np.ndarray]: 包含单个标量输出的元组
    """
    # 转换为 torch tensor
    y_grad_torch = torch.from_numpy(y_grad)
    clamp_max_mask_torch = torch.from_numpy(clamp_max_mask)
    x_clamped_loss_torch = torch.from_numpy(x_clamped_loss)

    # 记录输出 dtype（跟随 y_grad）
    out_dtype = y_grad_torch.dtype

    # ⚠ 全程 FP32 中间累加（与 kernel 一致）：kernel 将 fp16 输入 Cast 到 fp32 后再
    #   abs/add/mul/reduce_sum，仅最终缩位回输出 dtype。若 golden 直接在 fp16 下累加，
    #   大 shape / 大值（如 y_grad=65504 fp16 上限）中间乘积或归约和会溢出为 inf/nan，
    #   与 kernel 的有限结果差异巨大（DYN_GOLD 0.0%）。故此处统一提升到 fp32 计算。
    y_grad_f32 = y_grad_torch.to(torch.float32)
    mask_f32 = clamp_max_mask_torch.to(torch.float32)
    x_clamped_loss_f32 = x_clamped_loss_torch.to(torch.float32)

    # Step 1: signal = |mask'|
    # 对应 tbe.vabs（fp32 视图）
    signal = torch.abs(mask_f32)

    # Step 2: x_max_grad = x_clamped_loss + signal
    # 对应 tbe.vadd
    x_max_grad = torch.add(x_clamped_loss_f32, signal)

    # Step 3: y_grad * x_max_grad
    # 对应 tbe.vmul
    prod = torch.mul(y_grad_f32, x_max_grad)

    # Step 4: 全轴求和，输出标量（fp32 累加器）
    # 对应 tbe.reduce_sum（全轴）
    clamp_max_grad = torch.sum(prod)

    # Step 5: 缩位回输出 dtype（对应 kernel 末端 Cast fp32→输出 dtype）
    result = clamp_max_grad.to(out_dtype).numpy().astype(y_grad.dtype)

    return (result,)


def ttk_act_ulq_clamp_max_grad_golden(y_grad, clamp_max_mask, x_clamped_loss, **kwargs):
    """
    TTK kernel 模式入口（签名适配）。

    TTK 调用方式：golden_func(*context.input_arrays, **kwargs)，
    故 3 个输入按位置参数展开，额外 kwargs（如 full_soc_version）用 **kwargs 吸收。
    计算逻辑完全复用 act_ulq_clamp_max_grad_golden，不重复实现。

    输入顺序（与 def.cpp / proto.h 一致）：
        y_grad          (float16/float32)
        clamp_max_mask  (float16/float32/bool)
        x_clamped_loss  (float16/float32)
    返回：[clamp_max_grad]，形状 (1,)（与 CSV output_shapes 一致），dtype 跟随 y_grad
    """
    (result,) = act_ulq_clamp_max_grad_golden(y_grad, clamp_max_mask, x_clamped_loss)
    # act_ulq_clamp_max_grad_golden 返回 0 维标量；TTK CSV output_shapes 为 (1,)，reshape 对齐
    return [np.array(result).reshape(1).astype(y_grad.dtype)]
