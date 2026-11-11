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
"""TestSpec definitions for ApplyAdamD (Kernel / GEIR pathway).

ApplyAdamD is the three-output ("D") variant of Adam: unlike ApplyAdam, which
declares a single explicit output `var`, @apply_adam_d_def.cpp declares
Output("var") / Output("m") / Output("v"), so this golden returns THREE tensors.

Update rule (same protocol as TF ApplyAdam, beta-power convention):
    m_new   = beta1 * m + (1 - beta1) * grad
    v_new   = beta2 * v + (1 - beta2) * grad * grad
    lr_hat  = lr * sqrt(1 - beta2_power) / (1 - beta1_power)
    num     = beta1 * m_new + (1 - beta1) * grad   if use_nesterov else m_new
    var_new = var - lr_hat * num / (epsilon + sqrt(v_new))

Verified against @apply_adam_d_dag.h: CalcMt / CalcVt produce m_new / v_new,
CalcLr produces lr_hat, CalcVarTWithLr divides by (sqrt(v_t) + epsilon) and
CalcMtLookAhead supplies the Nesterov numerator — i.e. the Nesterov branch
divides by sqrt(v_new), never by a separate bias-corrected "v_hat".

Registered names:
  - "apply_adam_d" <- CSV op_name (snake_case CSV dialect)
  - "ApplyAdamD"   <- GE/REG_OP type name from OP_ADD(ApplyAdamD)
Both map to the one kernel spec class; the Kernel and GEIR pathways share it.

Benchmark: every golden funnels into the module-level ``_tf_apply_adam_d`` —
one shared numeric entry owned by no spec, so no spec class copies the maths.

Import rule: tensorflow / torch / ml_dtypes are imported lazily inside function
bodies only (TF via the ``_tf()`` singleton), so plugin discovery stays cheap.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np


__spec__ = {
    "apply_adam_d": "ApplyAdamDKernelSpec",
    "ApplyAdamD": "ApplyAdamDKernelSpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Both may coexist; ``__spec__`` wins.
__golden__ = {"kernel": {"apply_adam_d": "apply_adam_d_golden"}}

# Community default standard (stat_rel_err) for the three supported dtypes.
# cross_check is deliberately NOT pinned: it is a three-way comparison and TTK
# fails closed with GOLDEN_FAILURE when no third-party endpoint is alive. The
# CSV per-case precision_tolerances / absolute_precision are injected separately
# (legacy namespace) and are left untouched — no tolerance is relaxed here.
_TOLERANCE = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
}

_TF = None


def _tf():
    """Lazily import TensorFlow (CPU, quiet) so plugin discovery stays cheap."""
    global _TF
    if _TF is None:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        _TF = tf
    return _TF


def _is_low_precision(dtype):
    """Return True for float16 / bfloat16 (kernel up-casts these to FP32)."""
    if dtype == np.dtype(np.float16):
        return True
    try:
        from ml_dtypes import bfloat16 as _bf16

        if dtype == np.dtype(_bf16):
            return True
    except ImportError:
        pass
    return dtype.kind == "V" and dtype.itemsize == 2


def _scalar_f32(x):
    """Read an Adam scalar as float32.

    The scalars (beta1_power .. epsilon) are read at the op-def computational
    width, FP32 — they are never demoted to the tensor dtype even when the CSV
    declares them BF16/FP16, mirroring the kernel, which up-casts every scalar
    via ToFloat() before use (see @apply_adam_d_dag.h CalcVarT).

    The st CSV carries these scalars as shape (1,) in most cases and shape ()
    in others, so ``reshape(-1)[0]`` handles both without indexing a 0-d array.
    """
    arr = np.asarray(x)
    if arr.size == 0:
        return np.float32(0.0)
    return np.float32(np.asarray(arr, dtype=np.float32).reshape(-1)[0])


def _scalar_calc(x, calc_dtype):
    """Read a scalar at the reference compute dtype (float64 under Promote)."""
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array(0.0, dtype=calc_dtype)
    return np.asarray(arr, dtype=calc_dtype).reshape(-1)[0]


def _tf_apply_adam_d(var32, m32, v32, b1p, b2p, lr, b1, b2, eps, grad32, nesterov):
    """Reference update via tf.raw_ops.ApplyAdam, computed in FP32.

    Runs in an isolated tf.Graph + tf.compat.v1.Session with reference
    (non-resource) variables created via tf.compat.v1.Variable(...,
    use_resource=False). No global tf.compat.v1.disable_eager_execution() and
    no global resource-variable switch: the eager mode of the TTK process and
    of the sibling goldens (apply_adam / apply_ftrl both assert
    tf.executing_eagerly() in their self-smoke) stays untouched — the v1
    session only lives inside this function's `with` blocks.

    Three outputs: ApplyAdam has ref semantics — executing the node updates
    var / m / v in place. All three variables are therefore fetched in the SAME
    sess.run as the update node, so the values read back are the post-update
    state of the very run that executed it.

    State handling: var / m / v are initialized from the case's own inputs
    (independent per-case state), so repeated calls cannot leak state.

    Op-type proof: the built graph's update node type is asserted to be
    'ApplyAdam' — the computation path is exactly the specified raw op, not
    ResourceApplyAdam and not a numpy formula.
    """
    tf = _tf()
    tf_dtype = tf.as_dtype(np.asarray(var32).dtype)
    with tf.Graph().as_default() as graph:
        with tf.device("/CPU:0"):
            var = tf.compat.v1.Variable(var32, name="var", use_resource=False)
            m = tf.compat.v1.Variable(m32, name="m", use_resource=False)
            v = tf.compat.v1.Variable(v32, name="v", use_resource=False)
            upd = tf.raw_ops.ApplyAdam(
                var=var,
                m=m,
                v=v,
                beta1_power=tf.constant(b1p, dtype=tf_dtype),
                beta2_power=tf.constant(b2p, dtype=tf_dtype),
                lr=tf.constant(lr, dtype=tf_dtype),
                beta1=tf.constant(b1, dtype=tf_dtype),
                beta2=tf.constant(b2, dtype=tf_dtype),
                epsilon=tf.constant(eps, dtype=tf_dtype),
                grad=tf.constant(grad32, dtype=tf_dtype),
                use_locking=False,
                use_nesterov=bool(nesterov),
            )
            assert upd.op.type == "ApplyAdam", upd.op.type
            init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            var_val, m_val, v_val, upd_val = sess.run([var, m, v, upd])
        # the node output aliases the updated var buffer (ref semantics);
        # NaN-aware: out-of-domain inputs legitimately produce NaN (the st CSV
        # ranges let v go negative and 1-beta1_power reach 0), and NaN == NaN
        # here (the same buffer) must not read as "not executed".
        assert np.array_equal(var_val, upd_val, equal_nan=True), (
            "ApplyAdam update not executed"
        )
        return var_val, m_val, v_val


def _np_to_torch(x):
    """numpy ndarray -> torch tensor, bit-exact (bfloat16 via int16 bit view)."""
    import torch

    arr = np.asarray(x)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_to_np(t, dtype):
    """torch tensor -> numpy ndarray of ``dtype``, bit-exact for bfloat16."""
    import torch

    t = t.detach().cpu()
    if np.dtype(dtype).name == "bfloat16":
        import ml_dtypes

        return t.to(torch.bfloat16).view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy().astype(dtype, copy=False)


def _as_torch(x):
    """Normalize a carrier to torch, preserving device for torch inputs.

    torch inputs are returned as-is — a GPU third-party leg must stay on its
    device, with no silent .cpu() round-trip; numpy inputs are bridged
    bit-exactly.
    """
    import torch

    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return _np_to_torch(x)


def apply_adam_d_golden(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    use_locking: bool = False,
    use_nesterov: bool = False,
    **kwargs,
):
    """Kernel / GEIR golden for apply_adam_d (numpy carrier in and out).

    Parameter names and order follow @apply_adam_d_def.cpp inputs, then the two
    attributes. Benchmarked against tf.raw_ops.ApplyAdam (tensorflow==2.19.0)
    executed in an isolated graph/session with reference variables — NOT
    ResourceApplyAdam, NOT tf.train.AdamOptimizer, NOT a hand-written numpy
    formula.

    Dtypes:
        var / m / v / grad      : BF16 / FP16 / FP32 (state/computational)
        beta1_power ... epsilon : scalars, read at the FP32 computational width
    The kernel up-casts BF16/FP16 state to FP32, computes in FP32, then casts
    each result back (@apply_adam_d_dag.h Cast nodes). This golden mirrors that:
    var/m/v/grad are up-cast to FP32, the six scalars are read as FP32, the
    FP32 ApplyAdam runs, and each of the three results is cast back to the
    dtype of its corresponding input.

    Output convention:
        Returns a list of THREE tensors — [var, m, v] — matching the three
        Output() declarations of the op def and the CSV output_shapes /
        output_dtypes, which carry three entries. This is the key difference
        from apply_adam, whose op def declares a single explicit output.
    """
    out_dtypes = (np.asarray(var).dtype, np.asarray(m).dtype, np.asarray(v).dtype)

    if np.asarray(var).size == 0:
        return [
            np.asarray(src).astype(dt, copy=True)
            for src, dt in zip((var, m, v), out_dtypes)
        ]

    calc_dtype = (
        np.float64 if np.asarray(var).dtype == np.dtype(np.float64) else np.float32
    )
    var32 = np.asarray(var).astype(calc_dtype)
    m32 = np.asarray(m).astype(calc_dtype)
    v32 = np.asarray(v).astype(calc_dtype)
    grad32 = np.asarray(grad).astype(calc_dtype)

    outs = _tf_apply_adam_d(
        var32,
        m32,
        v32,
        _scalar_calc(beta1_power, calc_dtype),
        _scalar_calc(beta2_power, calc_dtype),
        _scalar_calc(lr, calc_dtype),
        _scalar_calc(beta1, calc_dtype),
        _scalar_calc(beta2, calc_dtype),
        _scalar_calc(epsilon, calc_dtype),
        grad32,
        use_nesterov,
    )
    return [res.astype(dt, copy=False) for res, dt in zip(outs, out_dtypes)]


class _ApplyAdamDTorchCompose:
    """torch third-party provider (spec-mode class form).

    Runs on the remote XPU/GPU endpoint, binding by parameter name against the
    operator's own input/attribute names. Pure torch basic-op composition in
    FP32 — an independent formulation from the TF golden, and it never calls the
    operator under test (no aclnn, no npu dispatch, no reuse of DUT outputs).

    Device discipline: torch inputs are kept exactly where they are. Nothing is
    detoured through .cpu() or numpy, so a GPU leg stays on the GPU.

    Returns three tensors (var, m, v) in op-def output order, each cast back to
    the dtype of its corresponding input.
    """

    def __call__(
        self,
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        beta1,
        beta2,
        epsilon,
        grad,
        use_locking: bool = False,
        use_nesterov: bool = False,
        **kwargs,
    ):
        import torch

        var_t, m_t, v_t, grad_t = (
            _as_torch(var),
            _as_torch(m),
            _as_torch(v),
            _as_torch(grad),
        )
        # Promote-aware: keep a promoted float64 reference exact.
        calc = torch.float64 if var_t.dtype == torch.float64 else torch.float32
        var_f = var_t.to(calc)
        m_f = m_t.to(calc)
        v_f = v_t.to(calc)
        grad_f = grad_t.to(calc)

        def sc(x):
            # scalars are read at the reference compute dtype (float64 under
            # Promote) and flattened so both () and (1,) work
            t = _as_torch(x).to(calc).reshape(-1)
            return t[0] if t.numel() else torch.zeros((), dtype=calc)

        b1p, b2p = sc(beta1_power), sc(beta2_power)
        lr_f, b1, b2, eps = sc(lr), sc(beta1), sc(beta2), sc(epsilon)

        m_new = b1 * m_f + (1.0 - b1) * grad_f
        v_new = b2 * v_f + (1.0 - b2) * grad_f * grad_f
        lr_hat = lr_f * torch.sqrt(1.0 - b2p) / (1.0 - b1p)
        num = (b1 * m_new + (1.0 - b1) * grad_f) if use_nesterov else m_new
        var_new = var_f - lr_hat * num / (eps + torch.sqrt(v_new))

        return [
            res.to(src.dtype)
            for res, src in zip((var_new, m_new, v_new), (var_t, m_t, v_t))
        ]


class ApplyAdamDKernelSpec:
    """Kernel / GEIR spec, registered by CSV ``op_name`` and GE type name.

    ``golden`` has no ``self``: TTK passes the CSV inputs positionally and the
    attributes by name, so it is bound as a staticmethod over the module-level
    golden rather than as an instance method.
    """

    golden = staticmethod(apply_adam_d_golden)
    third_party = {"torch": _ApplyAdamDTorchCompose}
    tolerance = _TOLERANCE


def _ref_all(var, m, v, b1p, b2p, lr, b1, b2, eps, grad, nesterov):
    """Independent fp64 element-wise reference returning (var, m, v).

    This is the corrected formula: the Nesterov branch divides by
    sqrt(v_new) — the previous golden referenced an undefined ``v_hat`` here
    and raised NameError for every use_nesterov=True case in the st CSV.
    """
    var = np.asarray(var).astype(np.float64)
    m = np.asarray(m).astype(np.float64)
    v = np.asarray(v).astype(np.float64)
    grad = np.asarray(grad).astype(np.float64)
    m_new = b1 * m + (1 - b1) * grad
    v_new = b2 * v + (1 - b2) * grad * grad
    lr_hat = lr * np.sqrt(1 - b2p) / (1 - b1p)
    num = (b1 * m_new + (1 - b1) * grad) if nesterov else m_new
    var_new = var - lr_hat * num / (eps + np.sqrt(v_new))
    return var_new, m_new, v_new


if __name__ == "__main__":
    import ml_dtypes

    rng = np.random.RandomState(0)
    n = 16
    tf = _tf()

    assert isinstance(__spec__, dict) and all(
        isinstance(k, str) and isinstance(val, str) for k, val in __spec__.items()
    ), "__spec__ must map str registry keys to str class names"

    # TF golden vs independent fp64 reference, over the three supported dtypes
    # and both Nesterov branches, with non-zero initial m/v state.
    for dt in ("float32", "float16", ml_dtypes.bfloat16):
        tol = 1e-5 if dt == "float32" else 3e-2
        for nesterov in (False, True):
            var = rng.uniform(-1, 1, n).astype(dt)
            m = rng.uniform(-0.5, 0.5, n).astype(dt)  # non-zero initial state
            v = rng.uniform(0.25, 2.0, n).astype(dt)  # non-zero initial state
            grad = rng.uniform(-0.5, 0.5, n).astype(dt)
            b1p, b2p, lr, b1, b2, eps = 0.6, 0.9, 0.01, 0.9, 0.999, 1e-8
            outs = apply_adam_d_golden(
                var,
                m,
                v,
                np.float32(b1p),
                np.float32(b2p),
                np.float32(lr),
                np.float32(b1),
                np.float32(b2),
                np.float32(eps),
                grad,
                use_nesterov=nesterov,
            )
            # Three outputs, in op-def order, each keeping its input dtype.
            assert isinstance(outs, list) and len(outs) == 3, "must return 3 tensors"
            for res, src in zip(outs, (var, m, v)):
                assert res.shape == src.shape, "shape mismatch"
                assert res.dtype == src.dtype, "dtype mismatch"
            refs = _ref_all(var, m, v, b1p, b2p, lr, b1, b2, eps, grad, nesterov)
            for res, ref in zip(outs, refs):
                assert np.allclose(res.astype(np.float64), ref, rtol=tol, atol=tol), (
                    f"TF-vs-fp64 mismatch dt={np.dtype(dt).name} nesterov={nesterov}"
                )

    def a(x):
        return np.float32(x)

    scal = (a(0.5), a(0.9), a(0.1), a(0.9), a(0.999), a(1e-8))
    var = np.array([0.5, -0.5], "float32")
    m = np.array([0.3, 0.2], "float32")
    v = np.array([1.0, 0.4], "float32")
    grad = np.array([0.2, -0.1], "float32")

    # Nesterov True/False must differ numerically: the branch is really taken,
    # and the fixed formula no longer raises NameError on the True path.
    o_f = apply_adam_d_golden(var, m, v, *scal, grad, use_nesterov=False)
    o_t = apply_adam_d_golden(var, m, v, *scal, grad, use_nesterov=True)
    assert not np.allclose(o_f[0], o_t[0]), "nesterov branch produced identical var"
    # m / v do not depend on the Nesterov flag — only the var numerator does.
    assert np.array_equal(o_f[1], o_t[1]) and np.array_equal(o_f[2], o_t[2]), (
        "m / v must be independent of use_nesterov"
    )

    # Scalar shape tolerance: the st CSV carries () in one case and (1,) in the
    # rest; both must give the same answer.
    o_s0 = apply_adam_d_golden(
        var, m, v, *[np.array(float(s), "float32") for s in scal], grad
    )
    o_s1 = apply_adam_d_golden(
        var, m, v, *[np.array([float(s)], "float32") for s in scal], grad
    )
    for r0, r1 in zip(o_s0, o_s1):
        assert np.array_equal(r0, r1), "scalar shape () vs (1,) disagree"

    # State isolation: repeated calls with the same inputs must agree (each
    # call builds a fresh graph with fresh variables initialized from inputs).
    o_rep = apply_adam_d_golden(var, m, v, *scal, grad, use_nesterov=False)
    for r0, r1 in zip(o_f, o_rep):
        assert np.array_equal(r0, r1), "state leaked across golden calls"

    # Low-precision path: fp16 tensors, FP32 scalars (scalars not demoted).
    o16 = apply_adam_d_golden(
        var.astype("float16"),
        m.astype("float16"),
        v.astype("float16"),
        *scal,
        grad.astype("float16"),
    )
    assert all(r.dtype == np.dtype("float16") for r in o16), "fp16 output dtype"

    # Empty input must not crash and must preserve shape/dtype per output.
    e = np.zeros((0,), "float32")
    o_e = apply_adam_d_golden(e, e, e, *scal, e)
    assert len(o_e) == 3 and all(r.shape == (0,) for r in o_e), "empty input"

    # torch third-party leg agrees with the TF golden on the same case.
    outs_t = _ApplyAdamDTorchCompose()(var, m, v, *scal, grad, use_nesterov=True)
    for res, ref in zip(outs_t, o_t):
        assert np.allclose(
            res.detach().cpu().numpy().astype(np.float64),
            ref.astype(np.float64),
            rtol=1e-5,
            atol=1e-6,
        ), "torch third-party leg disagrees with the TF golden"

    # Eager execution of the host process is untouched by the v1 sessions —
    # apply_adam / apply_ftrl assert the same thing in the same process.
    assert tf.executing_eagerly(), "global eager mode was disabled"

    print("apply_adam_d golden self-smoke OK")


# 通路交付情况
# 已注册: kernel + GEIR(复用同一 spec 类, 由 "apply_adam_d"/"ApplyAdamD" 两键指向)。
#   golden 口径 = tf.raw_ops.ApplyAdam 隔离图(tensorflow==2.19.0), 节点类型断言为
#   "ApplyAdam", 三个 ref 变量 var/m/v 与更新节点在同一次 sess.run 中取回。
# BUG 修复: 原 golden 的 use_nesterov 分支写成
#   var_update = alpha * m_hat / (np.sqrt(v_hat) + epsilon), 其中 v_hat 从未定义 ->
#   NameError。st CSV 99 条用例里有 46 条 use_nesterov=True, 该缺陷是活的(已实测复现)。
#   正确口径是除以 sqrt(v_new), 已与 @apply_adam_d_dag.h(CalcVarTNesterov 调用
#   CalcVarTWithLr(regMtAhead, regVt)) 和 TF ApplyAdam 对齐。
# aclnn: 不注册 —— ABI 实测 nm -D --defined-only libopapi.so 搜索
#   aclnnApplyAdamDGetWorkspaceSize 命中 0 次(cann-9.2.0-beta.1 / 9.0.0-beta.2 /
#   ascend-toolkit latest 三个安装包均为 0)。正式安装包未对外暴露该 aclnn 接口,
#   ops-nn/experimental/optim/apply_adam_d/op_api 下虽有手写实现, 但那需要自定义
#   算子包, 不能当作标准安装包的对外接口。
# e2e: 不注册, 列为待确认 —— torch_npu 侧确实有 npu_apply_adam(schema:
#   npu_apply_adam(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar beta1,
#   Scalar beta2, Scalar epsilon, Tensor grad, bool? use_locking,
#   bool? use_nesterov) -> (Tensor, Tensor, Tensor), var/m/v 走 .out 重载的 out 列表),
#   但它只在旧 acl_op 分支下沉 ApplyAdamD; 当前 op_api 分支下沉的是 ApplyAdam(非 D)。
#   实测证据: torch_npu 2.7.1.post6 的 libtorch_npu.so 中有 ApplyAdamKernelNpu /
#   ApplyAdamKernelNpuOpApi / aclnnApplyAdam / aclnnApplyAdamGetWorkspaceSize,
#   而字符串 "ApplyAdamD" 在该 .so 中 0 命中(仅 torchair 的
#   auto_generated_ge_raw_ops.py 有 IR 级 ApplyAdamD 绑定, 那是图模式 IR 直通,
#   不是 npu_apply_adam 的分发目标)。因此不能把 torch_npu.npu_apply_adam 算作本算子
#   (D 变体)的覆盖。要注册 E2E, 必须先钉住 torch_npu 版本/分支, 并证明该调用实际
#   下沉到 ApplyAdamD(例如用 dump/profiling 看到 ApplyAdamD 节点)。
# TensorFlow 三方腿: 不单独加 —— golden 自身已经走 TF raw op, 再加 tf 腿与 golden
#   同源, 不构成独立比对。三方腿只用 torch(独立的 torch 基础算子拼接口径)。
# GPU 三方实跑: 当前环境 ttk remote 未配置 endpoint, third_party 已声明但三方比对
#   结果记为未验证 —— 不等于不支持, 也不等于已通过。
