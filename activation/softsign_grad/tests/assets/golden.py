#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""softsign_grad golden（TTK TestSpec + third_party 形式；商用通路 kernel/geir，不含 aclnn——REQUIREMENTS v1.2 §7.4）。

golden 原则（本交付件精度方法学，不可违背）:
  CPU 与 GPU 均运行标杆算子本体（tf.raw_ops.SoftsignGrad）；
  CPU 真值侧升精度由 TTK golden_mode=Promote 在调用侧实现（mix_tolerance/cross_check
  标准自动触发：TTK 调用 golden 前把 FP16/BF16 输入提升为 FP32、FP32 输入提升为 FP64
  [DTYPE_PROMOTE_MAP: float16/bfloat16→float32, float32→float64]，并按提升语义比对与恢复。
  FP32→FP64 档使 CPU 真值精度高于 GPU 标杆原生 FP32，从而三方比对分母（GPU 相对
  真值的误差）非零、比值有意义——这是 13 例 float32 极值 rmse 比值超限不再触发的
  根本机制），
  golden 函数本身朴素委托标杆本体、不做显式精度转换——收到什么 dtype 按该 dtype 计算
  （TF 按 T 属性原生特化）。精度分析时分别计算 NPU / GPU 相对真值的误差，再对比双方误差大小。
  golden 函数必须委托标杆算子本体，禁止以自研实现替代标杆
  （自研复刻仅适用于无标杆可用的兜底场景，且须标注低置信度）。

分工:
  golden      = 标杆算子本体朴素委托（本文件；升精度策略由 TTK golden_mode=Promote 承载；
                注册统一经 TestSpec——见 __spec__ 注释，不使用旧式 __golden__ 接口键）
  third_party = 标杆算子的设备侧原生精度执行（TTK 调用 tf.raw_ops.SoftsignGrad，cross_check 比对）

对标算子: tf.raw_ops.SoftsignGrad（REQUIREMENTS.md v1.1 §6.5/§8.1 golden 基准）
数学公式: output = gradients / (1 + |features|)^2
dtype:   FP16 / FP32 / BF16，三 tensor 同 dtype（§3.1/§4）
判据:     cross_check（L1 档比值判据：NPU 误差不劣于 GPU 标杆误差——评审统一结论，
          全算子精度执行 cross_check；mare/mere/rmse ratio 与 small_value 见 TTK
          LEVEL_PRESETS/SMALL_VALUE 表；spec.yaml numerical_tolerance 的 rtol/atol
          数值保留为数学语义记录，不作为 cross_check 判据输入）

Input Func 判断: 不需要。
  1) REQUIREMENTS 无约束性参数转换（无量化/编码参数）；
  2) features 为前向原始输入而非前向输出（§3.2），随机生成即数学自洽，
     无需从前向重算（区别于 aclnnCdistBackward 的 cdist=前向输出场景）。
"""

# golden 统一经 TestSpec 注册（按名字查找，与通路无关）：
#   "softsign_grad"           ← kernel/geir 通路（按 op_name 查找，两路共用）
#   "tf.raw_ops.SoftsignGrad" ← TF 图迁移通路（TTK e2e，按 api_name 查找——
#                               framework_api/golden.py:168 / worker.py:761）。
# 不设旧式 __golden__ 注册，依据 TTK plugin/loader.py:45 查询优先级：
#   1) TestSpec 优先命中，旧 __golden__ 兼容层（custom_plugin_manager）永不回退；
#   2) 旧系统合法 level 仅 kernel/aclnn/e2e（custom_plugin_manager.py:30-32），
#      "geir" 键本不被旧系统消费（geir 通路金标准实际一直由 TestSpec 服务）。
# e2e 条目说明：无本条目时 e2e 的 tolerance 将回落 dtype 轴默认 mix_tolerance，
# 违背全通路 cross_check 统一判据（REQUIREMENTS §8.1）；e2e golden 虽有同 API
# CPU 兜底（Priority 3，被测 API 即标杆本体），显式注册使真值/判据/三方行为
# 显式可控。ES 直调无 TTK 接口（示例级验证承载），无需注册。
__spec__ = {
    "softsign_grad": "SoftsignGradTestSpec",
    "tf.raw_ops.SoftsignGrad": "SoftsignGradTestSpec",
}

import numpy as np
import tensorflow as tf


def _benchmark_softsign_grad(gradients, features):
    """标杆算子本体: tf.raw_ops.SoftsignGrad（CPU，按输入 dtype 原生执行，TF T 属性特化）。"""
    with tf.device("/CPU:0"):
        result = tf.raw_ops.SoftsignGrad(
            gradients=tf.convert_to_tensor(gradients),
            features=tf.convert_to_tensor(features),
        )
    return result.numpy()


def softsign_grad_golden(gradients, features, **kwargs):
    """kernel / geir 流程 CPU 真值: 标杆算子本体朴素委托。

    numpy.ndarray 入参，返回 [numpy.ndarray]。
    参数名与顺序同 def.cpp: gradients, features。
    置信度: ⭐⭐⭐⭐⭐（直接委托标杆算子本体 tf.raw_ops.SoftsignGrad，非自研复刻）

    Note:
        精度策略由 TTK 决定: golden_mode=Promote（mix_tolerance/cross_check 自动触发）时
        TTK 在调用前把低精度输入提升为 FP32、调用后按提升语义比对并恢复，本函数不做
        显式精度转换——收到什么 dtype 就按该 dtype 交标杆计算（TF 按 T 原生特化）。
        异常保护: 不加 epsilon——标杆语义原样保留（NaN/Inf 按 IEEE 除法传播，
        REQUIREMENTS §2.6/§5.8 为预期行为）。
    """
    return [_benchmark_softsign_grad(np.asarray(gradients), np.asarray(features))]


_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_THIRD_PARTY = {"tf": "tf.raw_ops.SoftsignGrad"}


class SoftsignGradTestSpec:
    """softsign_grad 测试规范（kernel / geir 流程，numpy 入参）。"""

    golden = softsign_grad_golden
    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


if __name__ == "__main__":
    import sys

    # FP32 常规值: 委托标杆 + 中转 == 闭式公式
    g = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
    f = np.array([0.0, 1.0, 2.0, -3.0], dtype=np.float32)
    out = softsign_grad_golden(g, f)[0]
    expected = g / (1.0 + np.abs(f)) ** 2
    assert np.array_equal(out, expected), f"FP32 mismatch: {out} vs {expected}"

    # 朴素委托断言: fp16 输入 → 按 T 原生计算，输出保持 fp16（无显式精度转换）
    out16 = softsign_grad_golden(g.astype(np.float16), f.astype(np.float16))[0]
    assert out16.dtype == np.float16, (
        f"naive delegation must preserve dtype: {out16.dtype}"
    )

    # Promote 链路组合断言（模拟 TTK golden_mode=Promote）:
    # TTK 调用前 astype(fp32) → golden 朴素计算 → fp32 结果 → 舍回原 dtype == 闭式公式
    outp = softsign_grad_golden(
        g.astype(np.float16).astype(np.float32), f.astype(np.float16).astype(np.float32)
    )[0]
    assert outp.dtype == np.float32, f"promoted path must return fp32: {outp.dtype}"
    exp16 = (g / (1.0 + np.abs(f)) ** 2).astype(np.float16)
    assert outp.astype(np.float16).dtype == np.float16
    assert np.allclose(outp.astype(np.float16), exp16, rtol=1e-3), (
        "promote-chain roundtrip mismatch"
    )

    # 边界: x=0 → output == gradients（invariant）
    g0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    f0 = np.zeros(3, dtype=np.float32)
    assert np.array_equal(softsign_grad_golden(g0, f0)[0], g0), (
        "zero-features invariant failed"
    )

    # 边界: 有限 g ∧ ±Inf f → 0；Inf g ∧ Inf f → NaN；NaN 传播
    out_inf = softsign_grad_golden(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([np.inf, -np.inf], dtype=np.float32),
    )[0]
    assert np.all(out_inf == 0.0), f"finite g with Inf f should give 0: {out_inf}"
    out_nan = softsign_grad_golden(
        np.array([np.inf, 1.0], dtype=np.float32),
        np.array([np.inf, np.nan], dtype=np.float32),
    )[0]
    assert np.isnan(out_nan[0]), "Inf g with Inf f should give NaN"
    assert np.isnan(out_nan[1]), "NaN propagation failed"

    # 空 tensor
    out_empty = softsign_grad_golden(
        np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    )[0]
    assert out_empty.shape == (0, 3), f"empty tensor shape: {out_empty.shape}"

    # TestSpec 结构检查
    assert __spec__["softsign_grad"] == "SoftsignGradTestSpec"
    assert __spec__["tf.raw_ops.SoftsignGrad"] == "SoftsignGradTestSpec", (
        "TF 图迁移（e2e）按 api_name 解析判据，需复用同一 TestSpec 保持 cross_check 统一"
    )
    assert SoftsignGradTestSpec.third_party == {"tf": "tf.raw_ops.SoftsignGrad"}
    assert SoftsignGradTestSpec.tolerance["float16"] == {
        "standard": "cross_check",
        "level": "L1",
    }
    # 注册统一：无旧式 __golden__（TTK TestSpec 优先，旧层仅无 TestSpec 时回退）
    assert "__golden__" not in globals(), (
        "golden 应统一经 TestSpec 注册，不设 __golden__ 旧注册"
    )
    assert callable(SoftsignGradTestSpec.golden), (
        "TestSpec.golden 必须可调用（全通路金标准入口）"
    )

    print("All golden self-tests passed.", file=sys.stderr)
