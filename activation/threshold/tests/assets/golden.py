#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""threshold 在 kernel/geir 测试路径下的 golden 编写。

比较域设计（对齐 kernel 结构性语义）:
  - fp16/bf16/fp32 输入: threshold 属性系 f32（REG_OP .Float(0.0f)），kernel 逐元素
    static_cast<float>(x) 与 f32 threshold 比较（GT 形，op_kernel/arch35/threshold_simt.h），
    golden 对齐为 x.astype(float32) > np.float32(threshold)，布尔结果 astype 回原 dtype 得
    0.0/1.0；fp16/bf16 tie 场景下，直接在输入自身 dtype 域比较会与设备 f32 域分歧，故统一 f32 域。
  - f64 输入: kernel 声明域（fp16/bf16/fp32）不含 f64，升 f32 系降位失真，golden 保持
    f64 域数学正确比较。
  - 纯 numpy 实现无 torch 依赖；NaN 比较恒 False -> 0.0，与 kernel GT 形一致；
    bf16 经 ml_dtypes 支持 astype/比较；0-d rank 纯 numpy 运算保持原 shape。
third_party 的 torch vendor 收到 torch.Tensor，直接调同一接口，无需手动转换。
torch 侧 threshold 语义（x > threshold ? x : value）与本算子（x > threshold ? 1 : 0）不同，
无现成竞品接口，以 torch.where 小算子拼接作为三方标杆。

Registered names:
  - "threshold"                     <- CSV op_name（Kernel/GEIR/Network 路径）
  - "Threshold"                     <- GE 图模式类型名变体（按 GE op type 生成 CamelCase 名）

aclnn 通路说明: 本算子不发布 aclnn 接口（aclnnThreshold/aclnnInplaceThreshold 系 ThresholdV2
算子能力）。

Import discipline: torch 仅在函数体内 lazy import —— TTK 在 aclnn 进程内加载本文件时 aclrt
已初始化，模块级竞品框架 import 有 SIGSEGV 风险；
golden 主体纯 numpy 不依赖 torch，third_party（torch vendor）仍用 lazy import。
"""

import numpy as np

__spec__ = {"threshold": "ThresholdKernelSpec", "Threshold": "ThresholdKernelSpec"}


class ThresholdKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor"""

    # -- golden — function form --
    def golden(x, *, threshold=0.0, **kwargs):
        """
        Golden function for threshold.
        All the parameters (names and order) follow SE doc prototype definition without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            x: Input tensor, supports float16 / float32 / bfloat16 (kernel domain);
              float64 handled by defensive branch (see module docstring)
            threshold: Threshold value (f32 attribute in kernel), default 0.0
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            y: Output tensor, same dtype as x, values 0.0 or 1.0
        """
        if x.dtype == np.float64:
            # DOUBLE 场景: kernel 声明域(fp16/bf16/fp32)不含 f64,升 f32 系降位失真,
            # 保持 f64 域数学正确比较
            y = (x > float(threshold)).astype(x.dtype)
        else:
            # 对齐 kernel 语义: threshold 系 f32 属性(REG_OP .Float(0.0f)),
            # kernel 逐元素 static_cast<float>(x) 与 f32 threshold 比较(GT 形)
            y = (x.astype(np.float32) > np.float32(threshold)).astype(x.dtype)
        return [y]

    # -- third_party — dict multi-vendor --
    # GPU golden
    class ThirdPartyImpl:
        # __init__ 接收输入 x：dtype/device 相关常量前置；__call__ 只剩纯 torch.where 调用
        def __init__(self, x, *, threshold=0.0, **kwargs):
            import torch

            self.threshold = threshold
            self.one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            self.zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        def __call__(self, x, **kwargs):
            import torch

            # Custom threshold: x > threshold -> 1.0, else -> 0.0
            output = torch.where(x > self.threshold, self.one, self.zero)
            return [output]

    third_party = {
        "torch": ThirdPartyImpl,
    }

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
