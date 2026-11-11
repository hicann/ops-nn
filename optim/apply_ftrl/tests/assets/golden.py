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
"""ApplyFtrl：真实 TensorFlow ApplyFtrl V1 参考 + 独立三方参考。

主干参考实现（来源：operator-artifacts 58597ec06344f5920f49e26ba203d16ba7a47592，
InplaceApplyFtrl(ApplyFtrl)/Golden/golden_thirdparty.py）保留不变：
8 个同 dtype 输入，use_locking=False，单显式输出 var；内部真实计算三态。
F16/BF16 reference 升 F32，实际 F32/F64 保持，尊重 TTK Promote 后的实际输入。
TensorFlow 延迟导入；Graph/Session/RefVariable 均为局部，不改变全局 eager 状态。

本轮移植补齐通路（不改参考数值口径）：
  - __spec__ 增加 GE 类型名键 "ApplyFtrl"（op_graph REG_OP(ApplyFtrl)），与
    "apply_ftrl" 共同指向同一 TestSpec；
  - third_party 增加 torch / tf 两条独立基础算子拼接腿（均不调用被测算子，
    不调用 golden 的 tf.raw_ops.ApplyFtrl），Torch 张量保持在原设备上计算，
    并对 accum / linear 作原地写回，保留主干既有写回合同；
  - tolerance 保留主干（operator-artifacts）的 cross_check/L1 验收要求：三方
    比对缺 XPU endpoint 时按受阻（GOLDEN_FAILURE）记录，不改成 stat_rel_err
    规避；CLI --compare 可显式覆盖用于本地回归。CSV 内每用例
    precision_tolerances / absolute_precision 单独注入，未放宽。
"""

import numpy as np


__spec__ = {
    "apply_ftrl": "ApplyFtrlTestSpec",
    "ApplyFtrl": "ApplyFtrlTestSpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Both may coexist; ``__spec__`` wins.
__golden__ = {"kernel": {"apply_ftrl": "apply_ftrl_golden"}}

_INPUT_NAMES = ("var", "accum", "linear", "grad", "lr", "l1", "l2", "lr_power")
_DTYPES = ("float16", "bfloat16", "float32")

# 保留主干 baseline 的 cross_check/L1：ApplyFtrl 的验收要求是三方交叉，不因
# 当前环境缺 XPU endpoint 而降级为 stat_rel_err；缺端点时如实记 GOLDEN_FAILURE
# （受阻）。本地/CI 可经 CLI --compare 显式覆盖，CSV per-case
# precision_tolerances / absolute_precision 仍单独注入，未放宽。
_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_TF = None


def _tf():
    """Lazily import TensorFlow (quiet) so plugin discovery stays cheap."""
    global _TF
    if _TF is None:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        _TF = tf
    return _TF


def _validate_numpy(inputs, use_locking):
    if use_locking:
        raise ValueError("ApplyFtrl only supports use_locking=False")
    dtype = inputs[0].dtype
    if dtype.name not in _DTYPES + ("float64",):
        raise TypeError("ApplyFtrl reference supports float16/bfloat16/float32/float64")
    if any(x.dtype != dtype for x in inputs):
        raise TypeError("All ApplyFtrl inputs must have the same dtype")
    if any(x.shape != inputs[0].shape for x in inputs[1:4]):
        raise ValueError("var, accum, linear and grad must have the same shape")
    if any(x.size != 1 for x in inputs[4:]):
        raise ValueError("FTRL scalar inputs must contain exactly one element")


def _compute_tf(inputs):
    """Run the real Ref ApplyFtrl once, then read its updated three states."""
    import tensorflow as tf

    var, accum, linear, grad, lr, l1, l2, lr_power = inputs
    tf_dtype = tf.as_dtype(var.dtype)
    graph = tf.compat.v1.Graph()
    with graph.as_default(), tf.device("/CPU:0"):
        v = tf.compat.v1.Variable(var, dtype=tf_dtype, use_resource=False)
        a = tf.compat.v1.Variable(accum, dtype=tf_dtype, use_resource=False)
        lin = tf.compat.v1.Variable(linear, dtype=tf_dtype, use_resource=False)
        var_op = tf.raw_ops.ApplyFtrl(
            var=v,
            accum=a,
            linear=lin,
            grad=tf.constant(grad, dtype=tf_dtype),
            lr=tf.constant(lr.reshape(()), dtype=tf_dtype),
            l1=tf.constant(l1.reshape(()), dtype=tf_dtype),
            l2=tf.constant(l2.reshape(()), dtype=tf_dtype),
            lr_power=tf.constant(lr_power.reshape(()), dtype=tf_dtype),
            use_locking=False,
            multiply_linear_by_lr=False,
        )
        init = tf.compat.v1.global_variables_initializer()
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(init)
            var_new = sess.run(var_op)
            accum_new, linear_new = sess.run((a, lin))
    return tuple(np.array(x, copy=True) for x in (var_new, accum_new, linear_new))


def _reference_states(
    var, accum, linear, grad, lr, l1, l2, lr_power, *, use_locking=False
):
    inputs = tuple(
        np.asarray(x) for x in (var, accum, linear, grad, lr, l1, l2, lr_power)
    )
    _validate_numpy(inputs, use_locking)
    input_dtype = inputs[0].dtype
    reference_dtype = (
        np.float32 if input_dtype.name in ("float16", "bfloat16") else input_dtype
    )
    # TTK 直接传 NumPy 输入：所有 TF 初值必须独立；不因模式标签再升/降精度。
    private_inputs = tuple(
        np.array(x, dtype=reference_dtype, order="C", copy=True) for x in inputs
    )
    return _compute_tf(private_inputs)


def apply_ftrl_golden(
    var, accum, linear, grad, lr, l1, l2, lr_power, *, use_locking=False, **kwargs
):
    """Kernel/GEIR golden：三态计算只映射单个显式输出 var。"""
    var_new, _, _ = _reference_states(
        var, accum, linear, grad, lr, l1, l2, lr_power, use_locking=use_locking
    )
    # 有 metadata 时按本次调用声明的输出 dtype 转换；直接调用回到 var dtype。
    # Promote 的原输出 dtype 由隔离 runtime 在上下文恢复后完整转换。
    output_dtypes = kwargs.get("output_dtypes")
    output_dtype = np.asarray(var).dtype
    if output_dtypes is not None:
        if len(output_dtypes) != 1:
            raise ValueError("ApplyFtrl requires exactly one output dtype")
        output_dtype = np.dtype(output_dtypes[0])
        if output_dtype.name not in _DTYPES + ("float64",):
            raise TypeError("Unsupported ApplyFtrl output dtype")
    return [var_new.astype(output_dtype, copy=False)]


class _ApplyFtrlTorchCompose:
    """torch third-party provider (independent, device-preserving).

    Composed from torch basic ops only — an independent formulation from the TF
    golden, and it never calls the operator under test (no aclnn, no npu
    dispatch, no reuse of DUT outputs).

    Device discipline: torch inputs are kept exactly where they are. Nothing is
    detoured through .cpu() or numpy, so a GPU leg stays on the GPU.
    Returns ONE tensor (var), matching the single Output() of the op def, and
    writes the updated accum / linear back in place (the provider contract the
    main branch already had).
    """

    def __call__(self, var, accum, linear, grad, lr, l1, l2, lr_power, **kwargs):
        import torch

        def t(x):
            if isinstance(x, torch.Tensor):
                return x
            arr = np.asarray(x)
            if arr.dtype.name == "bfloat16":
                # bit-exact bridge: numpy bf16 -> int16 bit view -> torch bf16
                return torch.from_numpy(arr.view("int16")).view(torch.bfloat16)
            return torch.from_numpy(arr)

        var_t = t(var)
        var_f = var_t.to(torch.float32)
        accum_f = t(accum).to(torch.float32)
        linear_f = t(linear).to(torch.float32)
        grad_f = t(grad).to(torch.float32)

        def sc(x):
            s = t(x).to(torch.float32).reshape(-1)
            return s[0] if s.numel() else torch.zeros((), dtype=torch.float32)

        lr_f, l1_f, l2_f, lrp_f = sc(lr), sc(l1), sc(l2), sc(lr_power)

        neg_p = -lrp_f
        accum_new = accum_f + grad_f * grad_f
        pow_old = torch.pow(accum_f, neg_p)
        pow_new = torch.pow(accum_new, neg_p)
        sigma = (pow_new - pow_old) / lr_f
        linear_new = linear_f + grad_f - sigma * var_f
        quad = pow_new / lr_f + 2.0 * l2_f
        x = torch.sign(linear_new) * l1_f - linear_new
        var_new = torch.where(
            torch.abs(linear_new) > l1_f, x / quad, torch.zeros_like(x)
        )
        # 只在最终输出/状态写回时恢复原 dtype/device。
        if isinstance(accum, torch.Tensor):
            with torch.no_grad():
                accum.copy_(accum_new.to(accum.dtype))
        if isinstance(linear, torch.Tensor):
            with torch.no_grad():
                linear.copy_(linear_new.to(linear.dtype))
        return [var_new.to(var_t.dtype)]


class _ApplyFtrlTfCompose:
    """TensorFlow third-party provider (independent, device-preserving).

    Composed from TF tensor primitives (add / mul / pow / sign / abs / where).
    It deliberately does NOT call tf.raw_ops.ApplyFtrl: the kernel golden
    already runs that raw op, so re-using it here would make the third-party
    leg same-source and worthless as an independent check.

    Carrier: TTK hands this leg device-placed tf.Tensors, so nothing is pulled
    back to the host — the computation stays on the assigned device and the
    result is returned as a tf.Tensor.

    Returns ONE tensor (var), matching the single Output() of the op def.
    """

    def __call__(self, var, accum, linear, grad, lr, l1, l2, lr_power, **kwargs):
        tf = _tf()

        def cv(x):
            return x if tf.is_tensor(x) else tf.convert_to_tensor(np.asarray(x))

        var_t = cv(var)
        out_dtype = var_t.dtype
        var_f = tf.cast(var_t, tf.float32)
        accum_f = tf.cast(cv(accum), tf.float32)
        linear_f = tf.cast(cv(linear), tf.float32)
        grad_f = tf.cast(cv(grad), tf.float32)

        def sc(x):
            s = tf.reshape(tf.cast(cv(x), tf.float32), [-1])
            return tf.cond(
                tf.size(s) > 0,
                lambda: s[0],
                lambda: tf.constant(0.0, tf.float32),
            )

        lr_f, l1_f, l2_f, lrp_f = sc(lr), sc(l1), sc(l2), sc(lr_power)

        neg_p = tf.negative(lrp_f)
        accum_new = tf.add(accum_f, tf.multiply(grad_f, grad_f))
        pow_old = tf.pow(accum_f, neg_p)
        pow_new = tf.pow(accum_new, neg_p)
        sigma = tf.divide(tf.subtract(pow_new, pow_old), lr_f)
        linear_new = tf.subtract(tf.add(linear_f, grad_f), tf.multiply(sigma, var_f))
        quad = tf.add(tf.divide(pow_new, lr_f), tf.multiply(2.0, l2_f))
        x = tf.subtract(tf.multiply(tf.sign(linear_new), l1_f), linear_new)
        var_new = tf.where(
            tf.greater(tf.abs(linear_new), l1_f),
            tf.divide(x, quad),
            tf.zeros_like(x),
        )
        return [tf.cast(var_new, out_dtype)]


class ApplyFtrlTestSpec:
    """Kernel/GEIR 共用；三态计算只映射单个显式输出 var。"""

    golden = staticmethod(apply_ftrl_golden)

    third_party = {
        "torch": _ApplyFtrlTorchCompose,
        "tf": _ApplyFtrlTfCompose,
    }
    tolerance = _TOLERANCE


# 通路交付情况
# 已注册: kernel + GEIR —— __spec__ 同时注册 CSV op_name "apply_ftrl" 与
#   op_graph REG_OP 类型名 "ApplyFtrl"，两者指向同一 ApplyFtrlTestSpec。
#   __golden__ 保留旧 loader 的 kernel 键。
# 参考实现: 主干 operator-artifacts 适配原样保留（tf.raw_ops.ApplyFtrl, V1
#   multiply_linear_by_lr=False，隔离 Graph/Session/RefVariable，一次执行读回
#   var/accum/linear 三态，非 NumPy 公式）。本轮未改公式、未改输出数。
# 输出数: 仍只返回 1 个显式输出 var —— @apply_ftrl_def.cpp 只声明 Output("var")，
#   accum / linear 是 device 原位状态，不是参与比对的输出。
# aclnn: 不注册 —— 标准安装包 libopapi.so 无 aclnnApplyFtrlGetWorkspaceSize 符号
#   （正式目录 ExtendCfgInfo 产出的是 Inner 接口；experimental 下的手写 op_api
#   需自定义算子包）。源码存在不等于标准包已部署；解除阻塞需在部署后的
#   libopapi.so 上复测符号并核对签名，属待验证，不等于永久不支持。
# e2e: 不注册 —— framework/apply_ftrl_tf_plugin.cpp 是 CANN 侧图转换映射，
#   tf.raw_ops.ApplyFtrl 本身可用，但当前环境无 Ascend TF adapter(tfplugin /
#   npu_device)，无法证明 TF 图下沉命中，记为待验证。
# 三方腿: torch + tf 两条独立基础算子拼接，均不复用 golden 的 raw op；torch 腿
#   保留原 dtype/device 并对 accum/linear 原地写回，tf 腿保持在 endpoint 设备上。
# GPU 三方实跑: 当前环境 ttk remote 未配置 endpoint，third_party 已声明但三方比对
#   结果记为未验证 —— 不等于不支持，也不等于已通过。
