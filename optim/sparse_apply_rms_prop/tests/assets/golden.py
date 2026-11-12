# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import hashlib

import numpy as np
import tensorflow as tf

__spec__ = {
    "sparse_apply_rms_prop": "SparseApplyRMSPropTestSpec",
    "tf.raw_ops.ResourceSparseApplyRMSProp": "TfResourceSparseApplyRMSPropSpec",
}


def _stable_seed(name):
    """Deterministic 64-bit seed derived from the testcase name (0 when absent)."""
    text = str(name) if name is not None else ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sparse_apply_rms_prop_golden(
    var,
    ms,
    mom,
    lr,
    rho,
    momentum,
    epsilon,
    grad,
    indices,
    *,
    use_locking=False,
    **kwargs,
):
    """
    Kernel golden for sparse_apply_rms_prop via tf.raw_ops.ResourceSparseApplyRMSProp.

    For float16/bfloat16 inputs, computation is performed in float32 and then cast back.

    Parameters follow @sparse_apply_rms_prop_def.cpp definition order.

    Returns a tuple of numpy.ndarray: (var_out, ms_out, mom_out)
    """
    orig_dtype = var.dtype

    if indices.size == 0:
        return var, ms, mom

    def to_idx(arr):
        return tf.constant(arr, dtype=tf.int32 if arr.dtype == np.int32 else tf.int64)

    is_fp16 = str(orig_dtype) == "float16" or (
        hasattr(orig_dtype, "name") and orig_dtype.name == "float16"
    )
    is_bf16 = str(orig_dtype) == "bfloat16" or (
        hasattr(orig_dtype, "name") and orig_dtype.name == "bfloat16"
    )
    needs_upcast = is_fp16 or is_bf16
    compute_dtype = np.float32 if needs_upcast else orig_dtype

    # Filter OOB indices (TF's ResourceSparseApplyRMSProp rejects them)
    N_var = var.shape[0]
    valid_mask = (indices >= 0) & (indices < N_var)
    if not np.all(valid_mask):
        if np.any(valid_mask):
            indices = indices[valid_mask]
            grad = grad[valid_mask]
        else:
            return var, ms, mom

    if indices.size == 0:
        return var, ms, mom

    # Deduplicate indices (keep first occurrence; kernel requires unique indices)
    _, first_idx = np.unique(indices, return_index=True)
    order = np.sort(first_idx)
    indices = indices[order]
    grad = grad[order]

    def safe_cast(arr, target_dtype):
        if arr.dtype != target_dtype:
            return arr.astype(target_dtype)
        return arr.copy()

    var_c = safe_cast(var, compute_dtype)
    ms_c = safe_cast(ms, compute_dtype)
    mom_c = safe_cast(mom, compute_dtype)
    grad_c = safe_cast(grad, compute_dtype)
    lr_c = safe_cast(lr, compute_dtype)
    rho_c = safe_cast(rho, compute_dtype)
    momentum_c = safe_cast(momentum, compute_dtype)
    epsilon_c = safe_cast(epsilon, compute_dtype)

    v_var = tf.Variable(var_c, name="var")
    v_ms = tf.Variable(ms_c, name="ms")
    v_mom = tf.Variable(mom_c, name="mom")

    lr_val = lr_c.item() if hasattr(lr_c, "item") else float(lr_c)
    rho_val = rho_c.item() if hasattr(rho_c, "item") else float(rho_c)
    momentum_val = (
        momentum_c.item() if hasattr(momentum_c, "item") else float(momentum_c)
    )
    epsilon_val = epsilon_c.item() if hasattr(epsilon_c, "item") else float(epsilon_c)

    tf.raw_ops.ResourceSparseApplyRMSProp(
        var=v_var.handle,
        ms=v_ms.handle,
        mom=v_mom.handle,
        lr=lr_val,
        rho=rho_val,
        momentum=momentum_val,
        epsilon=epsilon_val,
        grad=tf.constant(grad_c),
        indices=to_idx(indices),
        use_locking=use_locking,
    )

    out_var = v_var.read_value().numpy()
    out_ms = v_ms.read_value().numpy()
    out_mom = v_mom.read_value().numpy()

    if needs_upcast:
        out_var = out_var.astype(orig_dtype, copy=False)
        out_ms = out_ms.astype(orig_dtype, copy=False)
        out_mom = out_mom.astype(orig_dtype, copy=False)

    return out_var, out_ms, out_mom


class SparseApplyRMSPropTestSpec:
    """Kernel/GEIR TestSpec using TensorFlow ResourceSparseApplyRMSProp on CPU."""

    @staticmethod
    def customize_inputs(
        var, ms, mom, lr, rho, momentum, epsilon, grad, indices, **kwargs
    ):
        """约束输入：合法 indices 唯一化（越界 -1 保留）。

        kernel 契约要求 indices 唯一（README 约束说明），故在输入侧把合法
        索引无放回重抽为唯一值，避免算子 Grid-Stride 对重复行写竞争；
        越界索引保持不变。超参数（lr/rho/momentum/epsilon/ms）原样透传，
        对标 TF 与 IEEE 754 传播语义，不做数值约束。
        """
        N = var.shape[0]
        M = indices.shape[0]
        if M > 0 and N > 0:
            valid_mask = (indices >= 0) & (indices < N)
            valid_count = int(np.sum(valid_mask))
            if valid_count > 0 and valid_count <= N:
                rng = np.random.default_rng(_stable_seed(kwargs.get("testcase_name")))
                unique_valid = rng.choice(N, size=valid_count, replace=False).astype(
                    indices.dtype
                )
                indices = indices.copy()
                indices[valid_mask] = unique_valid
        return [var, ms, mom, lr, rho, momentum, epsilon, grad, indices]

    @staticmethod
    def golden(
        var,
        ms,
        mom,
        lr,
        rho,
        momentum,
        epsilon,
        grad,
        indices,
        *,
        use_locking=False,
        **kwargs,
    ):
        result = sparse_apply_rms_prop_golden(
            var,
            ms,
            mom,
            lr,
            rho,
            momentum,
            epsilon,
            grad,
            indices,
            use_locking=use_locking,
        )
        return list(result)

    class ThirdPartyImpl:
        """TensorFlow provider implementation used by optional cross-checks."""

        def __init__(self, *, use_locking=False, **kwargs):
            self.use_locking = use_locking

        def __call__(
            self, var, ms, mom, lr, rho, momentum, epsilon, grad, indices, **kwargs
        ):
            impl = SparseApplyRMSPropTfGatherScatter(
                var,
                ms,
                mom,
                grad,
                indices,
                lr,
                rho,
                momentum,
                epsilon,
                use_locking=self.use_locking,
            )
            return impl()

    third_party = {"tf": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class SparseApplyRMSPropTfGatherScatter:
    """gather/scatter 版 third_party（对齐 SIMT-LongTails SparseApplyRMSPropKernelSpec.ThirdPartyImpl）。

    FP16/BF16 输入统一提升为 FP32 计算，完成更新后再转回输入 dtype 回写，
    与 kernel/README 的“内部 FP32 计算、末尾回写”约定一致。

    注意：use_locking 仅透传存储，本实现是单次 gather/scatter 更新、不提供互斥锁，
    该属性在本参考中被忽略（与算子契约“indices 唯一”一致，无并发重复行场景）。
    """

    def __init__(
        self,
        var,
        ms,
        mom,
        grad,
        indices,
        lr,
        rho,
        momentum,
        epsilon,
        *,
        use_locking=False,
        **kwargs,
    ):
        # Unify input container types to tf tensors first: the GEIR/Kernel
        # third_party leg receives torch.Tensor, whose .dtype / comparisons /
        # tf ops would otherwise fail (e.g. torch.int32 fed to tf.cast).
        var = tf.convert_to_tensor(var)
        ms = tf.convert_to_tensor(ms)
        mom = tf.convert_to_tensor(mom)
        grad = tf.convert_to_tensor(grad)
        indices = tf.convert_to_tensor(indices)
        lr = tf.convert_to_tensor(lr)
        rho = tf.convert_to_tensor(rho)
        momentum = tf.convert_to_tensor(momentum)
        epsilon = tf.convert_to_tensor(epsilon)

        self.compute_dtype = (
            tf.float32 if var.dtype in (tf.float16, tf.bfloat16) else var.dtype
        )

        n_var = tf.cast(tf.shape(var)[0], indices.dtype)
        valid = (indices >= 0) & (indices < n_var)
        indices = tf.boolean_mask(indices, valid)
        grad = tf.boolean_mask(grad, valid)
        self.v_var = tf.Variable(var, name="var")
        self.v_ms = tf.Variable(ms, name="ms")
        self.v_mom = tf.Variable(mom, name="mom")
        self.grad = grad
        self.indices = indices
        self.lr = tf.cast(tf.reshape(lr, []), self.compute_dtype)
        self.rho = tf.cast(tf.reshape(rho, []), self.compute_dtype)
        self.momentum = tf.cast(tf.reshape(momentum, []), self.compute_dtype)
        self.epsilon = tf.cast(tf.reshape(epsilon, []), self.compute_dtype)
        self.use_locking = use_locking

    def __call__(self, **kwargs):
        v_rows = tf.cast(tf.gather(self.v_var, self.indices), self.compute_dtype)
        m_rows = tf.cast(tf.gather(self.v_ms, self.indices), self.compute_dtype)
        o_rows = tf.cast(tf.gather(self.v_mom, self.indices), self.compute_dtype)
        g_rows = tf.cast(self.grad, self.compute_dtype)

        one_minus_rho = tf.ones_like(self.rho) - self.rho
        ms_new = self.rho * m_rows + one_minus_rho * g_rows * g_rows
        self.v_ms.assign(
            tf.tensor_scatter_nd_update(
                self.v_ms,
                tf.expand_dims(self.indices, 1),
                tf.cast(ms_new, self.v_ms.dtype),
            )
        )

        mom_new = self.momentum * o_rows + self.lr * g_rows * tf.math.rsqrt(
            ms_new + self.epsilon
        )
        self.v_mom.assign(
            tf.tensor_scatter_nd_update(
                self.v_mom,
                tf.expand_dims(self.indices, 1),
                tf.cast(mom_new, self.v_mom.dtype),
            )
        )

        var_new = v_rows - mom_new
        self.v_var.assign(
            tf.tensor_scatter_nd_update(
                self.v_var,
                tf.expand_dims(self.indices, 1),
                tf.cast(var_new, self.v_var.dtype),
            )
        )

        return [self.v_var, self.v_ms, self.v_mom]


class TfResourceSparseApplyRMSPropSpec:
    """ResourceSparseApplyRMSProp E2E TestSpec，返回三个原地更新结果。"""

    @staticmethod
    def _run(
        var,
        ms,
        mom,
        lr,
        rho,
        momentum,
        epsilon,
        grad,
        indices,
        use_locking,
        device,
    ):
        with tf.device(device):
            var_tensor = tf.convert_to_tensor(var)
            dtype = var_tensor.dtype
            compute_dtype = tf.float32 if dtype in (tf.float16, tf.bfloat16) else dtype
            var_resource = tf.Variable(
                tf.cast(var_tensor, compute_dtype), trainable=False
            )
            ms_resource = tf.Variable(
                tf.cast(tf.convert_to_tensor(ms), compute_dtype), trainable=False
            )
            mom_resource = tf.Variable(
                tf.cast(tf.convert_to_tensor(mom), compute_dtype), trainable=False
            )
            grad_tensor = tf.cast(tf.convert_to_tensor(grad), compute_dtype)
            indices_tensor = tf.convert_to_tensor(indices)
            # Filter out-of-bound indices: tf.raw_ops.ResourceSparseApplyRMSProp
            # raises InvalidArgument on them, while the kernel and the other
            # reference paths skip OOB rows and leave them unchanged.
            n_var = tf.cast(tf.shape(var_tensor)[0], indices_tensor.dtype)
            valid = (indices_tensor >= 0) & (indices_tensor < n_var)
            indices_tensor = tf.boolean_mask(indices_tensor, valid)
            grad_tensor = tf.boolean_mask(grad_tensor, valid)

            def scalar(value):
                return tf.cast(
                    tf.reshape(tf.convert_to_tensor(value), []), compute_dtype
                )

            tf.raw_ops.ResourceSparseApplyRMSProp(
                var=var_resource.handle,
                ms=ms_resource.handle,
                mom=mom_resource.handle,
                lr=scalar(lr),
                rho=scalar(rho),
                momentum=scalar(momentum),
                epsilon=scalar(epsilon),
                grad=grad_tensor,
                indices=indices_tensor,
                use_locking=bool(use_locking),
            )
            return [
                tf.cast(tf.identity(var_resource), dtype),
                tf.cast(tf.identity(ms_resource), dtype),
                tf.cast(tf.identity(mom_resource), dtype),
            ]

    @staticmethod
    def golden(
        var,
        ms,
        mom,
        lr,
        rho,
        momentum,
        epsilon,
        grad,
        indices,
        use_locking=False,
        name=None,
        **kwargs,
    ):
        del name, kwargs
        return TfResourceSparseApplyRMSPropSpec._run(
            var,
            ms,
            mom,
            lr,
            rho,
            momentum,
            epsilon,
            grad,
            indices,
            use_locking,
            "/CPU:0",
        )

    # TensorFlow 未提供 ResourceSparseApplyRMSProp 的 GPU kernel；三方侧复用
    # 上方由 TensorFlow gather/scatter/math 小算子组成的实现。
    third_party = {"tf": SparseApplyRMSPropTfGatherScatter}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
