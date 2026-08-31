"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

InplaceApplyFtrl TestSpec (GEIR pathway).

FTRL-Proximal V1 就地更新。

精度策略：
  - golden：FP16/BF16 输入在计算前统一升精度到 FP32，计算完成后降回原精度。
    计算调用 tf.raw.ops.ApplyFtrl（与 kernel 核对过的参考真值），进程内直接调用。
  - ThirdPartyImpl：调用 TF ApplyFtrl 在原生 dtype 下计算，不做升精度操作。
"""

import numpy as np

try:
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
except ImportError:
    tf = None

__spec__ = {"inplace_apply_ftrl": "InplaceApplyFtrlTestSpec"}


def _to_scalar(x):
    arr = np.asarray(x)
    return float(arr.item()) if arr.size == 1 else float(arr.flat[0])


def _np_to_tf_dtype(np_dtype):
    if np_dtype == np.dtype(np.float64):
        return tf.float64
    if np_dtype == np.dtype(np.float32):
        return tf.float32
    if np_dtype == np.dtype(np.float16):
        return tf.float16
    from ml_dtypes import bfloat16

    if np_dtype == np.dtype(bfloat16):
        return tf.bfloat16
    return tf.float32


def _compute(var, accum, linear, grad, lr, l1, l2, lr_power, tf_dtype):
    """Run tf.raw_ops.ApplyFtrl in-process, return (var_new, accum_new, linear_new)."""
    g = tf.compat.v1.Graph()
    with g.as_default():
        v = tf.compat.v1.Variable(var, dtype=tf_dtype, use_resource=False)
        a = tf.compat.v1.Variable(accum, dtype=tf_dtype, use_resource=False)
        lin = tf.compat.v1.Variable(linear, dtype=tf_dtype, use_resource=False)
        gd = tf.compat.v1.constant(grad, dtype=tf_dtype)
        var_op = tf.raw_ops.ApplyFtrl(
            var=v,
            accum=a,
            linear=lin,
            grad=gd,
            lr=tf.constant(_to_scalar(lr), dtype=tf_dtype),
            l1=tf.constant(_to_scalar(l1), dtype=tf_dtype),
            l2=tf.constant(_to_scalar(l2), dtype=tf_dtype),
            lr_power=tf.constant(_to_scalar(lr_power), dtype=tf_dtype),
            use_locking=False,
        )
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            var_val = sess.run(var_op)
            accum_val = sess.run(a)
            linear_val = sess.run(lin)
    return (np.asarray(var_val), np.asarray(accum_val), np.asarray(linear_val))


def inplace_apply_ftrl_golden(var, accum, linear, grad, lr, l1, l2, lr_power, **kwargs):
    """FTRL-Proximal V1 就地更新 golden.

    FP16/BF16 输入在计算前统一升精度到 FP32；其余 dtype（含 TTK Promote
    升上来的 FP64/FP32）按传入精度直接计算，不做降级。计算完成后降回
    调用方期望的输出 dtype。返回 [var_new, accum_new, linear_new].
    """
    if tf is None:
        raise RuntimeError("tensorflow is required for InplaceApplyFtrl golden")

    out_dtype = var.dtype
    calc_dtype = var.dtype

    # 仅 FP16/BF16 升精到 FP32；FP32/FP64 保持原样（尊重 TTK Promote 的升精结果）
    if calc_dtype in (np.dtype(np.float16),):
        calc_dtype = np.dtype(np.float32)
    else:
        try:
            from ml_dtypes import bfloat16 as _bf16

            if calc_dtype == np.dtype(_bf16):
                calc_dtype = np.dtype(np.float32)
        except ImportError:
            pass
        # numpy 侧 bfloat16 以 2 字节 void(V) 形式出现
        if calc_dtype.kind == "V" and calc_dtype.itemsize == 2:
            calc_dtype = np.dtype(np.float32)

    if calc_dtype != var.dtype:
        var = var.astype(calc_dtype)
        accum = accum.astype(calc_dtype)
        linear = linear.astype(calc_dtype)
        grad = grad.astype(calc_dtype)
        lr = np.asarray(lr).astype(calc_dtype)
        l1 = np.asarray(l1).astype(calc_dtype)
        l2 = np.asarray(l2).astype(calc_dtype)
        lr_power = np.asarray(lr_power).astype(calc_dtype)

    tf_dtype = _np_to_tf_dtype(calc_dtype)

    var_new, accum_new, linear_new = _compute(
        var, accum, linear, grad, lr, l1, l2, lr_power, tf_dtype
    )

    return [
        var_new.astype(out_dtype),
        accum_new.astype(out_dtype),
        linear_new.astype(out_dtype),
    ]


class _TfInplaceApplyFtrl:
    """TF ApplyFtrl reference — computes in native dtype, no FP32 promotion."""

    def __init__(self, *, use_locking=False, **kwargs):
        self.use_locking = bool(use_locking)

    def __call__(self, var, accum, linear, grad, lr, l1, l2, lr_power, **kwargs):
        if tf is None:
            raise RuntimeError(
                "tensorflow is required for InplaceApplyFtrl ThirdPartyImpl"
            )

        def to_np(t):
            if hasattr(t, "numpy"):
                try:
                    return t.numpy()
                except (TypeError, RuntimeError):
                    pass
            if hasattr(t, "detach"):
                t = t.detach().cpu()
            if hasattr(t, "float") and str(getattr(t, "dtype", "")) == "torch.bfloat16":
                return t.float().numpy()
            return np.asarray(t)

        var_np = to_np(var)
        accum_np = to_np(accum)
        linear_np = to_np(linear)
        grad_np = to_np(grad)
        lr_np = to_np(lr)
        l1_np = to_np(l1)
        l2_np = to_np(l2)
        lr_power_np = to_np(lr_power)

        tf_dtype = _np_to_tf_dtype(var_np.dtype)

        return _compute(
            var_np,
            accum_np,
            linear_np,
            grad_np,
            lr_np,
            l1_np,
            l2_np,
            lr_power_np,
            tf_dtype,
        )


class InplaceApplyFtrlTestSpec:
    """InplaceApplyFtrl kernel TestSpec (GEIR pathway)."""

    golden = staticmethod(inplace_apply_ftrl_golden)
    third_party = {"tf": _TfInplaceApplyFtrl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    v = rng.standard_normal(8).astype(np.float32)
    a = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.1
    ln = rng.standard_normal(8).astype(np.float32)
    g = rng.standard_normal(8).astype(np.float32)
    vo, ao, lo = InplaceApplyFtrlTestSpec.golden(v, a, ln, g, 0.1, 0.01, 0.001, -0.5)
    assert vo.shape == v.shape and vo.dtype == v.dtype
    g0 = np.zeros_like(g)
    _, ao0, _ = InplaceApplyFtrlTestSpec.golden(v, a, ln, g0, 0.1, 0.01, 0.001, -0.5)
    assert np.array_equal(ao0, a), "accum invariant violated"
    print("golden self-smoke OK")
