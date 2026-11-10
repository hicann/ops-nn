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
    计算调用 CPU tf.raw_ops.ApplyFtrl；TTK Promote 传入的 FP32/FP64 保持其精度。
  - third_party["tf"]：在输入 Tensor 所在设备调用等价的 ResourceApplyFtrl，
    使用私有状态变量并返回原 dtype 的 [var, accum, linear]，不修改输入。
    FP16/FP32 原生计算；BF16 因 TF GPU 未注册 FTRL 内核，在同设备以 FP32
    计算再回写 BF16，符合本算子 README 的内部 FP32 计算语义。
  - float16/bfloat16/float32 的三个输出均使用 TTK cross_check / L1：
    由运行时解析 L1 误差比值及逐 dtype 小值域标准。CPU Golden 与原 dtype
    远端三方独立执行；float64 仅用于 Promote 参考，不是算子声明的输出 dtype。
"""

import numpy as np

__spec__ = {"inplace_apply_ftrl": "InplaceApplyFtrlTestSpec"}

_INPUT_NAMES = ("var", "accum", "linear", "grad", "lr", "l1", "l2", "lr_power")
_REFERENCE_DTYPES = ("float16", "bfloat16", "float32", "float64")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        from ml_dtypes import bfloat16

        # NumPy cannot directly receive Torch BF16; restore BF16 before TF sees it.
        return value.float().numpy().astype(bfloat16)
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        from ml_dtypes import bfloat16

        array = array.view(bfloat16)
    return array


def _prepare_inputs(values, use_locking):
    if use_locking:
        raise ValueError("InplaceApplyFtrl only supports use_locking=False")
    var = _to_numpy(values[0])
    inputs = [var]
    for i, value in enumerate(values[1:], start=1):
        # Python scalar arguments carry no tensor dtype; retain the existing convenience API.
        if (
            i >= 4
            and isinstance(value, (float, int))
            and not isinstance(value, np.generic)
        ):
            value = np.asarray(value, dtype=var.dtype)
        inputs.append(_to_numpy(value))
    if var.dtype.name not in _REFERENCE_DTYPES:
        raise TypeError(
            "InplaceApplyFtrl reference supports float16/bfloat16/float32/float64"
        )
    for name, value in zip(_INPUT_NAMES, inputs):
        if value.dtype != var.dtype:
            raise TypeError(f"{name} must have the same dtype as var")
    if var.ndim > 8:
        raise ValueError("InplaceApplyFtrl state rank must be at most 8")
    if any(value.shape != var.shape for value in inputs[1:4]):
        raise ValueError("var, accum, linear and grad must have the same shape")
    for name, value in zip(_INPUT_NAMES[4:], inputs[4:]):
        if value.shape not in ((), (1,)):
            raise ValueError(f"{name} must be a scalar (0D) or shape [1]")
    return inputs


def _compute(inputs):
    """Run real TensorFlow ApplyFtrl in a local graph without changing eager mode."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow is required for InplaceApplyFtrl reference"
        ) from exc

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
        )
        init = tf.compat.v1.global_variables_initializer()
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(init)
            var_new = sess.run(var_op)
            accum_new, linear_new = sess.run((a, lin))
    return tuple(
        np.array(value, copy=True) for value in (var_new, accum_new, linear_new)
    )


def inplace_apply_ftrl_golden(
    var, accum, linear, grad, lr, l1, l2, lr_power, *, use_locking=False, **kwargs
):
    """Return three states; promote FP16/BF16 to FP32 and preserve FP32/FP64 Promote inputs."""
    inputs = _prepare_inputs(
        (var, accum, linear, grad, lr, l1, l2, lr_power), use_locking
    )
    out_dtype = inputs[0].dtype
    calc_dtype = np.float32 if out_dtype.name in ("float16", "bfloat16") else out_dtype
    private_inputs = [np.array(value, dtype=calc_dtype, copy=True) for value in inputs]
    return [value.astype(out_dtype) for value in _compute(private_inputs)]


class _TfInplaceApplyFtrl:
    """Device-local TF FTRL V1; BF16 uses FP32 resources and BF16 outputs."""

    def __init__(self, *, use_locking=False, **kwargs):
        if use_locking:
            raise ValueError("InplaceApplyFtrl only supports use_locking=False")
        self.use_locking = False

    def __call__(
        self,
        var,
        accum,
        linear,
        grad,
        lr,
        l1,
        l2,
        lr_power,
        *,
        use_locking=False,
        **kwargs,
    ):
        import tensorflow as tf

        if self.use_locking or use_locking:
            raise ValueError("InplaceApplyFtrl only supports use_locking=False")
        inputs = (var, accum, linear, grad, lr, l1, l2, lr_power)
        if not all(tf.is_tensor(value) for value in inputs):
            raise TypeError("TF third_party requires provider Tensor inputs")
        if var.dtype.name not in _REFERENCE_DTYPES:
            raise TypeError("InplaceApplyFtrl requires floating-point inputs")
        for name, value in zip(_INPUT_NAMES, inputs):
            if value.dtype != var.dtype:
                raise TypeError(f"{name} must have the same dtype as var")
            if value.device != var.device:
                raise ValueError(f"{name} must be on the same device as var")
        if var.shape.rank > 8:
            raise ValueError("InplaceApplyFtrl state rank must be at most 8")
        if any(value.shape != var.shape for value in inputs[1:4]):
            raise ValueError("var, accum, linear and grad must have the same shape")
        for name, value in zip(_INPUT_NAMES[4:], inputs[4:]):
            if value.shape not in ((), (1,)):
                raise ValueError(f"{name} must be a scalar (0D) or shape [1]")

        # ResourceApplyFtrl implements the same FTRL V1 update as ApplyFtrl,
        # accepting eager provider Tensors without a numpy/CPU graph boundary.
        # Keep all resources and scalar reshapes on the remotely assigned device.
        with tf.device(var.device):
            # TF 2.21 training_ops.cc registers GPU FTRL for half/float/double,
            # but not BF16. Use the operator's documented internal FP32 compute
            # for BF16 (README "internal fp32" constraint), on the same device.
            compute_inputs = (
                [tf.cast(value, tf.float32) for value in inputs]
                if var.dtype == tf.bfloat16
                else inputs
            )
            states = [
                tf.Variable(value, trainable=False) for value in compute_inputs[:3]
            ]
            if any(state.device != var.device for state in states):
                raise RuntimeError(
                    "TF third_party resources moved off the input device"
                )
            update = tf.raw_ops.ResourceApplyFtrl(
                var=states[0].handle,
                accum=states[1].handle,
                linear=states[2].handle,
                grad=compute_inputs[3],
                lr=tf.reshape(compute_inputs[4], ()),
                l1=tf.reshape(compute_inputs[5], ()),
                l2=tf.reshape(compute_inputs[6], ()),
                lr_power=tf.reshape(compute_inputs[7], ()),
                use_locking=False,
            )
            with tf.control_dependencies([] if update is None else [update]):
                outputs = [tf.cast(state.read_value(), var.dtype) for state in states]
            if any(value.device != var.device for value in outputs):
                raise RuntimeError("TF third_party outputs moved off the input device")
            return outputs


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
    idx = np.arange(8, dtype=np.float32)
    v = idx * 0.25 - 1.0
    a = idx * 0.125 + 0.1
    ln = idx * 0.1 - 0.4
    g = idx * 0.05 - 0.2
    vo, ao, lo = InplaceApplyFtrlTestSpec.golden(v, a, ln, g, 0.1, 0.01, 0.001, -0.5)
    assert vo.shape == v.shape and vo.dtype == v.dtype
    g0 = np.zeros_like(g)
    _, ao0, _ = InplaceApplyFtrlTestSpec.golden(v, a, ln, g0, 0.1, 0.01, 0.001, -0.5)
    assert np.allclose(ao0, a, rtol=1e-6, atol=1e-7), "accum invariant violated"
    print("golden self-smoke OK")
