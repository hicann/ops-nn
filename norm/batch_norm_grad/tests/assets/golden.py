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

"""BatchNormGrad（batch_norm_grad）算子测试规范（TTK TestSpec）。

batch_norm_grad 仅支持 TensorFlow，不注册 torch / aclnn。

注册关系：
    kernel/GEIR : op_name = batch_norm_grad（numpy 入/出）。GEIR 复用 kernel 条目。
    e2e        : api_name = tf.raw_ops.FusedBatchNormGradV3（tf 入/出）。

语义说明（kernel/GEIR 与 e2e 存在 SEMANTIC_DIFF，须分别给真值）：
    - kernel/GEIR 走自定义 AscendC BatchNormGrad 算子，reserve_space_2 在
      training 下保存的是 rstd（1/sqrt(var+eps)）而非 variance；inference 下
      保存 variance。原生 TF 无对应 rstd 语义接口，故用 TF 小算子拼接公式实现。
    - e2e 的 api_name 即原生 TF 算子 tf.raw_ops.FusedBatchNormGradV3，其
      reserve_space_2 恒为 variance，故直接调用原生 TF 接口作真值。

kernel/GEIR 公式（N = 通道内元素数，归约除通道轴外的所有轴）：
    x_hat      = (x - mean) * rstd
    dscale     = sum(dy * x_hat)
    doffset    = sum(dy)
    training  : dx = scale * rstd * (dy - (x_hat * dscale + doffset) / N)
    inference : rstd = 1/sqrt(reserve_space_2 + eps); dx = dy * scale * rstd

输出 5 路：x_backprop / scale_backprop / offset_backprop / reserve_space_4 /
reserve_space_5。kernel/GEIR 下后两路恒为空 shape(0,)（infershape 固定 {0}）；
e2e 原生 TF 下后两路为 0-D 标量 0.0。
output_mask 仅在 inference 被 tiling 采纳：training 恒计算三路真值
（tiling 不读 output_mask），故 training 下强制 [True, True, True]。
"""

import numpy as np
import tensorflow as tf

try:
    import ml_dtypes  # noqa: F401  注册 numpy bfloat16 名称
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "batch_norm_grad": "BatchNormGradTestSpec",
    "tf.raw_ops.FusedBatchNormGradV3": "BatchNormGradE2ETestSpec",
}

_TOLERANCE = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _as_numpy(value):
    """tf.Tensor / numpy / 标量 → numpy。"""
    if value is None:
        return None
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return np.asarray(value)


def _compute_dtype(*arrays):
    """计算精度：任一入参为 fp64（golden_mode=Promote 抬档）则整体 fp64，否则 fp32。"""
    for array in arrays:
        if array is not None and getattr(array, "dtype", None) == np.float64:
            return np.float64
    return np.float32


def _channel_axis(data_format, rank):
    fmt = str(data_format).upper()
    if fmt in ("NHWC", "NDHWC"):
        return rank - 1
    return 1


def _normalize_output_mask(output_mask, is_training):
    """输出掩码归一化为 3 个 bool（dx, dscale, doffset）。

    training 恒 [True, True, True]（tiling 不读 output_mask）；inference 遵循
    给定掩码，缺省取 def.cpp 默认 [True, False, False]。
    """
    if is_training:
        return [True, True, True]
    if output_mask is None:
        return [True, False, False]
    mask = [bool(v) for v in output_mask]
    if len(mask) != 3:
        raise ValueError(f"output_mask length must be 3, got {len(mask)}")
    return mask


def _dtype_from_name(name):
    """dtype 名称 → numpy dtype（适配 CSV 里可能的别名写法）。"""
    n = str(name).lower()
    if n in ("bfloat16", "dt_bfloat16", "dt_bf16", "bf16"):
        return np.dtype("bfloat16")
    if n in ("float16", "dt_float16", "half", "fp16"):
        return np.float16
    if n in ("float32", "float", "dt_float", "fp32"):
        return np.float32
    if n in ("float64", "double", "dt_double", "fp64"):
        return np.float64
    return np.dtype(name)


def _output_dtype(kwargs, index, default):
    out_dtypes = kwargs.get("output_dtypes") or []
    if index < len(out_dtypes) and out_dtypes[index] is not None:
        return str(out_dtypes[index])
    return default


def _ensure_tf(value, dtype):
    """numpy / tf.Tensor → 指定 dtype 的 tf.Tensor。"""
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype != dtype:
        tensor = tf.cast(tensor, dtype)
    return tensor


# ---------------------------------------------------------------------------
# TF 小算子拼接的核心公式（kernel/GEIR，rstd 语义）
# ---------------------------------------------------------------------------
def _tf_batch_norm_grad(
    dy, x, scale, mean, rs2, is_training, epsilon, data_format, compute_dtype
):
    """在 tf 上完成 BatchNormGrad 三路输出（dx, dscale, doffset）。

    dy/x/scale/mean/rs2 均为 compute_dtype 的 tf.Tensor。
    """
    rank = len(x.shape)
    axis = _channel_axis(data_format, rank)
    reduce_axes = [i for i in range(rank) if i != axis]
    reduce_count = 1
    for i in reduce_axes:
        reduce_count *= int(x.shape[i])

    eps_t = tf.constant(float(epsilon), dtype=compute_dtype)
    n_t = tf.constant(float(reduce_count), dtype=compute_dtype)

    bshape = [1] * rank
    bshape[axis] = int(scale.shape[0])
    scale_b = tf.reshape(scale, bshape)
    mean_b = tf.reshape(mean, bshape)
    if is_training:
        rstd = rs2
    else:
        rstd = tf.math.rsqrt(rs2 + eps_t)
    rstd_b = tf.reshape(rstd, bshape)

    x_hat = (x - mean_b) * rstd_b
    dscale = tf.reduce_sum(dy * x_hat, axis=reduce_axes)
    doffset = tf.reduce_sum(dy, axis=reduce_axes)

    dscale_b = tf.reshape(dscale, bshape)
    doffset_b = tf.reshape(doffset, bshape)
    if is_training:
        dx = scale_b * rstd_b * (dy - (x_hat * dscale_b + doffset_b) / n_t)
    else:
        dx = dy * scale_b * rstd_b
    return dx, dscale, doffset


def batch_norm_grad_golden(
    y_backprop,
    x,
    scale,
    reserve_space_1,
    reserve_space_2,
    reserve_space_3=None,
    *,
    epsilon=1e-4,
    data_format="NHWC",
    is_training=True,
    output_mask=None,
    **kwargs,
):
    """kernel/GEIR golden：TF 小算子拼接 rstd 语义公式。

    参数名与顺序跟随 def.cpp 输入（不含输出）；属性 epsilon/data_format/
    is_training/output_mask 按关键字传入。reserve_space_3 为 OPTIONAL 输入，
    不参与梯度计算（仅透传占位）。

    返回 5 路 numpy 输出：[x_backprop, scale_backprop, offset_backprop,
    reserve_space_4, reserve_space_5]；后两路为 shape(0,) 空张量。
    """
    dy_np = _as_numpy(y_backprop)
    x_np = _as_numpy(x)
    scale_np = _as_numpy(scale)
    mean_np = _as_numpy(reserve_space_1)
    rs2_np = _as_numpy(reserve_space_2)
    for name, arr in (
        ("y_backprop", dy_np),
        ("x", x_np),
        ("scale", scale_np),
        ("reserve_space_1", mean_np),
        ("reserve_space_2", rs2_np),
    ):
        if arr is None:
            raise ValueError(f"input {name} must not be None (REQUIRED)")

    compute_dtype = _compute_dtype(dy_np, x_np, scale_np, mean_np, rs2_np)
    dy_t = _ensure_tf(dy_np, compute_dtype)
    x_t = _ensure_tf(x_np, compute_dtype)
    scale_t = _ensure_tf(scale_np, compute_dtype)
    mean_t = _ensure_tf(mean_np, compute_dtype)
    rs2_t = _ensure_tf(rs2_np, compute_dtype)

    dx_t, dscale_t, doffset_t = _tf_batch_norm_grad(
        dy_t,
        x_t,
        scale_t,
        mean_t,
        rs2_t,
        bool(is_training),
        float(epsilon),
        data_format,
        compute_dtype,
    )
    dx = dx_t.numpy()
    dscale = dscale_t.numpy()
    doffset = doffset_t.numpy()

    mask = _normalize_output_mask(output_mask, bool(is_training))
    dx_dtype = _output_dtype(kwargs, 0, str(dy_np.dtype))
    dscale_dtype = _output_dtype(kwargs, 1, "float32")
    doffset_dtype = _output_dtype(kwargs, 2, "float32")

    outputs = []
    for enabled, value, dtype_name in (
        (mask[0], dx, dx_dtype),
        (mask[1], dscale, dscale_dtype),
        (mask[2], doffset, doffset_dtype),
    ):
        if not enabled:
            outputs.append(None)
        else:
            outputs.append(
                np.ascontiguousarray(
                    np.asarray(value).astype(_dtype_from_name(dtype_name), copy=False)
                )
            )
    # reserve_space_4 / reserve_space_5 恒空 shape(0,)
    outputs.append(np.empty((0,), dtype=np.float32))
    outputs.append(np.empty((0,), dtype=np.float32))
    return outputs


def batch_norm_grad_e2e_golden(
    y_backprop,
    x,
    scale,
    reserve_space_1,
    reserve_space_2,
    reserve_space_3,
    *,
    epsilon=1e-4,
    data_format="NHWC",
    is_training=True,
    **kwargs,
):
    """e2e golden：直接调用原生 TF 接口 tf.raw_ops.FusedBatchNormGradV3。

    原生 TF 语义下 reserve_space_2 恒为 variance，与 api_name 自洽；入参可
    为 tf.Tensor 或 numpy，先转 numpy 再经 tf.convert_to_tensor 喂入。返回
    5 路 numpy 输出，reserve_space_4/5 为原生 TF 产出的 0-D 标量 0.0。
    """
    dx_t, dscale_t, doffset_t, rs4_t, rs5_t = tf.raw_ops.FusedBatchNormGradV3(
        y_backprop=tf.convert_to_tensor(_as_numpy(y_backprop)),
        x=tf.convert_to_tensor(_as_numpy(x)),
        scale=tf.convert_to_tensor(_as_numpy(scale)),
        reserve_space_1=tf.convert_to_tensor(_as_numpy(reserve_space_1)),
        reserve_space_2=tf.convert_to_tensor(_as_numpy(reserve_space_2)),
        reserve_space_3=tf.convert_to_tensor(_as_numpy(reserve_space_3)),
        epsilon=float(epsilon),
        data_format=str(data_format),
        is_training=bool(is_training),
    )
    return [
        np.asarray(dx_t.numpy()),
        np.asarray(dscale_t.numpy()),
        np.asarray(doffset_t.numpy()),
        np.asarray(rs4_t.numpy(), dtype=np.float32),
        np.asarray(rs5_t.numpy(), dtype=np.float32),
    ]


# ---------------------------------------------------------------------------
# 三方标杆（third_party）
# ---------------------------------------------------------------------------
class _TfRstdThirdPartyImpl:
    """kernel/GEIR 三方标杆：TF 小算子拼接 rstd 语义公式（reserve_space_2 同算子）。"""

    def __init__(self, *, epsilon=1e-4, data_format="NHWC", is_training=True, **kwargs):
        self.epsilon = float(epsilon)
        self.data_format = str(data_format)
        self.is_training = bool(is_training)

    def __call__(
        self, y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs
    ):
        dy_np = _as_numpy(y_backprop)
        compute_dtype = _compute_dtype(dy_np)
        dx_t, dscale_t, doffset_t = _tf_batch_norm_grad(
            _ensure_tf(y_backprop, compute_dtype),
            _ensure_tf(x, compute_dtype),
            _ensure_tf(scale, compute_dtype),
            _ensure_tf(reserve_space_1, compute_dtype),
            _ensure_tf(reserve_space_2, compute_dtype),
            self.is_training,
            self.epsilon,
            self.data_format,
            compute_dtype,
        )
        return [
            tf.cast(dx_t, tf.convert_to_tensor(y_backprop).dtype),
            dscale_t,
            doffset_t,
            tf.zeros([0], dtype=tf.float32),
            tf.zeros([0], dtype=tf.float32),
        ]


class BatchNormGradTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    golden = staticmethod(batch_norm_grad_golden)
    third_party = {"tf": _TfRstdThirdPartyImpl}
    tolerance = _TOLERANCE


class BatchNormGradE2ETestSpec:
    """e2e 流程：tf.raw_ops.FusedBatchNormGradV3（原生 TF 接口）。"""

    golden = staticmethod(batch_norm_grad_e2e_golden)
    third_party = {"tf": "tf.raw_ops.FusedBatchNormGradV3"}
    tolerance = _TOLERANCE
