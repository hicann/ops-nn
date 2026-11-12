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

"""InstanceNorm golden：实例归一化算子参考实现。

本算子为 GE 层自定义算子（TensorFlow 框架），无 TF 原生接口、无 torch 对应——
TF 侧通过小算子图（``tf.nn.moments`` + 算术算子）在 GE 图编译期经
``ScopeInstanceNormPass`` 融合为该算子。故 kernel / GEIR 的 golden 与
third_party 三方标杆均按同一 TF 小算子图语义实现（仅注册 kernel / GEIR 通路）。

语义：对每个样本每个通道，沿空间维规约求均值/方差，再做归一化：
    mean = E(x),  var = Var(x)（无偏修正关闭，即除以 N）
    y    = (x - mean) / sqrt(var + epsilon) * gamma + beta

三个输出（顺序与 def.cpp / proto.h 一致）：
    y        —— shape / dtype / format 与 x 一致
    mean     —— 维度数同 x，规约轴置 1（非规约轴 N、C 保留），dtype 同 gamma
    variance —— 同 mean

规约轴由 ``data_format`` 决定：NCHW/NCDHW/ND → 除 N、C 外的空间维；NHWC/NDHWC →
除 N、末位 C 外的空间维。ND 2D 无空间维（mean=x、var=0）。真值在 FP32 中计算，
再按各输出 dtype 回转。
"""

__spec__ = {
    "instance_norm": "InstanceNormSpec",  # Kernel / GEIR
}


def _reduce_axes_and_channel(data_format, ndim):
    """按 data_format 返回 (规约轴 tuple, 通道轴下标)。"""
    data_format = str(data_format).upper()
    if data_format == "NCHW":
        return tuple(range(2, ndim)), 1
    if data_format == "NHWC":
        return tuple(range(1, ndim - 1)), ndim - 1
    if data_format == "NCDHW":
        return tuple(range(2, ndim)), 1
    if data_format == "NDHWC":
        return tuple(range(1, ndim - 1)), ndim - 1
    if data_format == "ND":
        return tuple(range(2, ndim)), 1
    raise ValueError(f"Unsupported data_format: {data_format}")


def _instance_norm_tf(x, gamma, beta, data_format="NDHWC", epsilon=1e-6):
    """TensorFlow 小算子图实现，返回 [y, mean, variance]（均为 tf.Tensor）。

    InstanceNorm 无原生 tf 接口，TF 侧正是通过 ``tf.nn.moments`` 求 mean/var、
    再用 ``rsqrt`` 求标准差倒数、最后用算术算子（Sub/Mul/Add）完成归一的
    小算子图，GE 编译期经 ScopeInstanceNormPass 融合成本算子。本实现复刻同一
    小算子图语义。x / gamma / beta 均为 tf.Tensor（native dtype）。
    """
    import tensorflow as tf

    x_dtype = x.dtype
    gamma_dtype = gamma.dtype
    ndim = len(x.shape.as_list())
    reduce_axes, channel = _reduce_axes_and_channel(data_format, ndim)

    x_f32 = tf.cast(x, tf.float32)
    gamma_f32 = tf.cast(gamma, tf.float32)
    beta_f32 = tf.cast(beta, tf.float32)

    param_view = [1] * ndim
    param_view[channel] = int(gamma.shape.as_list()[0])
    gamma_v = tf.reshape(gamma_f32, param_view)
    beta_v = tf.reshape(beta_f32, param_view)

    if reduce_axes:
        mean, var = tf.nn.moments(x_f32, axes=list(reduce_axes), keepdims=True)
    else:
        mean = x_f32
        var = tf.zeros_like(x_f32)

    # 小算子图：rstd = rsqrt(var + eps)；y = (x - mean) * rstd * gamma + beta
    rstd = tf.math.rsqrt(var + tf.cast(epsilon, tf.float32))
    y = (x_f32 - mean) * rstd * gamma_v + beta_v

    return [
        tf.cast(y, x_dtype),
        tf.cast(mean, gamma_dtype),
        tf.cast(var, gamma_dtype),
    ]


class InstanceNormTfRef:
    """TensorFlow 三方标杆（cross_check）：以小算子图等价实现本算子。

    输入为 XPU server 端的 tf.Tensor；参数按名绑定（x / gamma / beta 张量，
    data_format / epsilon 属性）。
    """

    def __call__(self, x, gamma, beta, data_format="NDHWC", epsilon=1e-6, **kwargs):
        return _instance_norm_tf(x, gamma, beta, data_format, epsilon)


class InstanceNormSpec:
    """kernel / GEIR 流程：numpy 入/出，用 TF 小算子图实现。"""

    @staticmethod
    def golden(x, gamma, beta, data_format="NDHWC", epsilon=1e-6, **kwargs):
        """numpy 入/出，返回 [y, mean, variance]。

        入参为 numpy 数组（kernel 契约），内部转 tf.Tensor 后按 TF 小算子图
        计算，再转回 numpy（bf16/fp16/fp32 经 tf 无损回转）。
        """
        import tensorflow as tf

        x_t = tf.convert_to_tensor(x)
        gamma_t = tf.convert_to_tensor(gamma)
        beta_t = tf.convert_to_tensor(beta)

        results = _instance_norm_tf(x_t, gamma_t, beta_t, data_format, epsilon)

        return [r.numpy() for r in results]

    # 三方标杆与 golden 同一套 TF 小算子图（本算子无 TF 原生接口，无 torch 对应）
    third_party = {"tf": InstanceNormTfRef}
