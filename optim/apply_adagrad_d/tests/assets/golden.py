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
"""TestSpec definitions for ApplyAdagradD: Kernel / GEIR path.

Update rule (per @apply_adagrad_d_dag.h, both template branches):
    accum_new = accum + grad * grad      if update_slots else accum
    var_new   = var - lr * grad / sqrt(accum_new)

Outputs: exactly two — var / accum — as declared by @apply_adagrad_d_def.cpp
(Output("var"), Output("accum")) and by the st CSV
(output_inplace_indexes="(0, 1)"). When update_slots=False the kernel does not
write accum_out at all; because the output aliases the accum input in place,
the observable accum output is the unchanged input, which is what this golden
returns.

Registered names:
  - "apply_adagrad_d"  <- CSV op_name (Kernel path, numpy carrier)
  - "ApplyAdagradD"    <- GE/REG_OP type name from OP_ADD(ApplyAdagradD)
                          (GEIR path, same spec class)
``tf.raw_ops.ApplyAdagrad`` is deliberately NOT registered as a front end here:
this is the D variant (it carries update_slots and declares two outputs), so its
output contract differs from stock ApplyAdagrad's single-output ref op. Only the
kernel / GEIR pair is claimed.

Benchmark: golden() funnels into the module-level ``_benchmark_apply_adagrad_d``
— one shared TensorFlow reference owned by no spec. It runs
``tf.raw_ops.ApplyAdagrad`` inside an isolated ``tf.Graph`` +
``tf.compat.v1.Session`` on reference (non-resource) variables, and BOTH outputs
are read back from TensorFlow: ApplyAdagrad has ref semantics, so after the node
executes the accum variable has been updated in place and is fetched from the
same session. No numpy arithmetic is left on the reference path — numpy is only
the carrier (and the bf16 bit bridge).

Eager hygiene: no global switch is ever flipped. The previous revision called
``tf.compat.v1.disable_v2_behavior()``, which is process-global and silently
broke sibling goldens in the same worker (optim/apply_adam and optim/apply_ftrl
both assert ``tf.executing_eagerly()``). The v1 session now lives only inside
the ``with`` blocks of one function, and the host process stays eager.

Precision: BF16/FP16 carriers are up-cast to FP32 once, the whole update runs in
FP32 (mirroring the kernel's Cast<float> DAG), then each output is cast back to
its input dtype.

Import rule: tensorflow / torch / ml_dtypes are imported lazily inside function
bodies only, so plugin discovery stays cheap and a TF import never happens for
specs that do not need it.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

__spec__ = {
    "apply_adagrad_d": "ApplyAdagradDKernelSpec",
    "ApplyAdagradD": "ApplyAdagradDKernelSpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Both may coexist; ``__spec__`` wins.
__golden__ = {"kernel": {"apply_adagrad_d": "apply_adagrad_d_golden"}}

# Community default standard (stat_rel_err). cross_check is deliberately NOT
# pinned: it is a three-way comparison and TTK fails closed when no third-party
# endpoint is alive. third_party is declared below, so
# `--compare cross_check --provider torch|tf` enables three-way comparison
# whenever a GPU endpoint exists. The CSV per-case precision_tolerances /
# absolute_precision are injected separately (legacy namespace) and are left
# untouched — no tolerance is relaxed here.
_TOLERANCE = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
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


def _bf16_dtype(dtype):
    """Return the ml_dtypes bfloat16 dtype when ``dtype`` is a bf16 carrier.

    bf16 normally lands as the ml_dtypes.bfloat16 extension dtype (itself of
    numpy kind 'V', itemsize 2); some carriers hand over an unnamed 2-byte void
    dtype instead, which must be re-viewed before any arithmetic. Anything else
    returns None.
    """
    dt = np.dtype(dtype)
    if dt.kind != "V" or dt.itemsize != 2:
        return None
    import ml_dtypes

    return np.dtype(ml_dtypes.bfloat16)


def _to_f32(x):
    """numpy carrier -> reference compute ndarray.

    Low-precision / float32 carriers compute in float32; a promoted float64
    reference (TTK Promote / cross_check) stays float64 so FP32 cancellation is
    not baked into the high-precision reference.
    """
    arr = np.asarray(x)
    bf16 = _bf16_dtype(arr.dtype)
    if bf16 is not None and arr.dtype != bf16:
        arr = arr.view(bf16)
    if arr.dtype == np.dtype(np.float64):
        return arr
    return arr.astype(np.float32)


def _from_f32(arr32, dtype):
    """float32 ndarray -> numpy carrier of ``dtype`` (bfloat16-safe)."""
    dt = np.dtype(dtype)
    bf16 = _bf16_dtype(dt)
    if bf16 is not None:
        out = arr32.astype(bf16)
        return out if dt == bf16 else out.view(dt)
    return arr32.astype(dt, copy=False)


def _scalar_f32(x):
    """Read lr as a python-level scalar at the reference compute dtype.

    The st CSV declares lr with shape (1,) and the kernel consumes it through
    Vec::Duplicate(ScalarAttr) — i.e. element 0 broadcast over the tile — so the
    scalar is index-extracted here instead of relying on numpy broadcasting.
    """
    flat = _to_f32(x).reshape(-1)
    if flat.size == 0:
        return np.float32(0.0)
    return flat[0]


def _benchmark_apply_adagrad_d(var32, accum32, lr32, grad32, update_slots):
    """Shared benchmark Adagrad-D — module-level, owned by no spec.

    Reference: ``tf.raw_ops.ApplyAdagrad`` (stock TensorFlow), evaluated in FP32
    inside an isolated ``tf.Graph`` plus ``tf.compat.v1.Session(graph=...)`` on
    reference variables created with ``use_resource=False``. No global switch is
    touched — no ``disable_v2_behavior``, no ``disable_eager_execution``, no
    global resource-variable flag — so the eager mode of the TTK worker and of
    every sibling golden in the same process stays intact. ``use_resource=False``
    is passed explicitly rather than inherited, so the graph is also immune to a
    global flag another plugin may already have flipped.

    Both outputs come from TensorFlow. ApplyAdagrad is a ref op: it mutates the
    var and accum variables in place, so the node is executed first and the two
    variables are then read back from the same session. Fetching the update node
    and reading the variables are two ordered ``sess.run`` calls on one session,
    which is deterministic — a single combined fetch would leave the read/write
    order unspecified.

    The node is evaluated in float64 regardless of the carrier. ApplyAdagrad's
    CPU kernel realises var - lr*grad/sqrt(accum') as a multiply by an
    approximate rsqrt, so an FP32 evaluation leaves up to 1 ULP(@1.0) of the
    kernel's own approximation in the reference (measured 1.25e-07 over 100
    random draws). That residue is harmless in the bulk but becomes an O(1)
    relative error wherever the exact result is zero, which is precisely what
    the deterministic near-zero case probes. In float64 the same stock node
    stays within 2.2e-16 of the exact value, so the reference no longer carries
    an error the operator under test does not have. A float64 carrier (TTK
    Promote / cross_check) was already float64 and is unaffected.

    Returns (var, accum) as float64 ndarrays in op-def output order; carrier /
    dtype restoration is the caller's job. Never touches the operator under test.
    """
    tf = _tf()
    calc = np.float64
    var_c = np.asarray(var32, dtype=calc)
    accum_c = np.asarray(accum32, dtype=calc)
    grad_c = np.asarray(grad32, dtype=calc)
    lr_c = calc(lr32)
    tf_dtype = tf.as_dtype(calc)
    with tf.Graph().as_default() as graph:
        with tf.device("/CPU:0"):
            var = tf.compat.v1.Variable(var_c, name="var", use_resource=False)
            accum = tf.compat.v1.Variable(accum_c, name="accum", use_resource=False)
            upd = tf.raw_ops.ApplyAdagrad(
                var=var,
                accum=accum,
                lr=tf.constant(lr_c, dtype=tf_dtype),
                grad=tf.constant(grad_c, dtype=tf_dtype),
                use_locking=False,
                update_slots=bool(update_slots),
            )
            # Op-type proof: the node on the computation path is exactly the
            # specified raw op — not ResourceApplyAdagrad, not a numpy formula.
            assert upd.op.type == "ApplyAdagrad", upd.op.type
            init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            upd_val = sess.run(upd)
            var_val, accum_val = sess.run([var, accum])
    # The node output aliases the updated var buffer (ref semantics), so an
    # equal read-back proves the update node really ran and committed — a stale
    # pre-update read would differ. NaN-aware: the CSV carries inf/nan cases
    # whose legitimate result is NaN, and NaN == NaN on the same buffer must not
    # be mistaken for "not executed".
    assert np.array_equal(var_val, upd_val, equal_nan=True), (
        "ApplyAdagrad update not executed"
    )
    return var_val, accum_val


def apply_adagrad_d_golden(
    var,
    accum,
    lr,
    grad,
    update_slots: bool = True,
    use_locking: bool = False,
    **kwargs,
):
    """Kernel / GEIR golden (numpy carrier in, numpy carrier out).

    Parameter names and order follow @apply_adagrad_d_def.cpp inputs, then the
    attributes. Returns [var, accum] — two tensors, matching the op def and the
    st CSV output_inplace_indexes="(0, 1)".

    use_locking only guards concurrent updates; with per-case private state it
    cannot change the numerics, and it is forwarded as False to keep the graph
    free of lock nodes.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name
    """
    del use_locking, kwargs
    var_dtype = np.asarray(var).dtype
    accum_dtype = np.asarray(accum).dtype

    if np.asarray(var).size == 0:
        return [np.asarray(var).copy(), np.asarray(accum).copy()]

    var_out32, accum_out32 = _benchmark_apply_adagrad_d(
        _to_f32(var), _to_f32(accum), _scalar_f32(lr), _to_f32(grad), update_slots
    )
    return [_from_f32(var_out32, var_dtype), _from_f32(accum_out32, accum_dtype)]


def _np_to_torch(x):
    """numpy ndarray -> torch tensor, bit-exact (bfloat16 via int16 bit view)."""
    import torch

    # np.ascontiguousarray would promote shape () to (1,), turning a scalar
    # carrier into a one-element vector; a 0-d array is contiguous already.
    arr = np.asarray(x)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy(order="C")
    if _bf16_dtype(arr.dtype) is not None:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


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


class _ApplyAdagradDTorchCompose:
    """torch third-party provider (spec-mode class form).

    Runs on the remote XPU endpoint, binding by parameter name against the
    operator's own input / attribute names. Composed from torch basic ops only —
    it never calls the operator under test and never calls TensorFlow, so it is
    independent of the golden's ApplyAdagrad path. Everything stays on the
    incoming tensors' device: no .cpu(), no numpy arithmetic.
    """

    def __call__(
        self,
        var,
        accum,
        lr,
        grad,
        update_slots: bool = True,
        use_locking: bool = False,
        **kwargs,
    ):
        import torch

        del use_locking, kwargs
        var_t, accum_t, grad_t = _as_torch(var), _as_torch(accum), _as_torch(grad)
        lr_t = _as_torch(lr).reshape(-1)[0]
        var_dtype, accum_dtype = var_t.dtype, accum_t.dtype

        calc = torch.float64 if var_t.dtype == torch.float64 else torch.float32
        var_f = var_t.to(calc)
        accum_f = accum_t.to(calc)
        grad_f = grad_t.to(calc)
        lr_f = lr_t.to(calc)

        accum_out = accum_f + grad_f * grad_f if update_slots else accum_f
        var_out = var_f - lr_f * grad_f / torch.sqrt(accum_out)
        return [var_out.to(var_dtype), accum_out.to(accum_dtype)]


class _ApplyAdagradDTfCompose:
    """TensorFlow third-party provider (spec-mode class form).

    Composed from TF element-wise ops (add / mul / sqrt / div) — deliberately
    NOT tf.raw_ops.ApplyAdagrad, which is the golden's own reference: reusing it
    here would make the third-party leg same-source and worthless as a cross
    check. Runs eagerly on whatever device the endpoint placed the inputs on;
    nothing is pulled back to host.
    """

    def __call__(
        self,
        var,
        accum,
        lr,
        grad,
        update_slots: bool = True,
        use_locking: bool = False,
        **kwargs,
    ):
        tf = _tf()
        del use_locking, kwargs
        var_t = tf.convert_to_tensor(var)
        accum_t = tf.convert_to_tensor(accum)
        grad_t = tf.convert_to_tensor(grad)
        var_dtype, accum_dtype = var_t.dtype, accum_t.dtype

        var_f = tf.cast(var_t, tf.float32)
        accum_f = tf.cast(accum_t, tf.float32)
        grad_f = tf.cast(grad_t, tf.float32)
        lr_f = tf.reshape(tf.cast(tf.convert_to_tensor(lr), tf.float32), [-1])[0]

        accum_out = (
            tf.add(accum_f, tf.multiply(grad_f, grad_f)) if update_slots else accum_f
        )
        var_out = tf.subtract(
            var_f, tf.divide(tf.multiply(lr_f, grad_f), tf.sqrt(accum_out))
        )
        return [tf.cast(var_out, var_dtype), tf.cast(accum_out, accum_dtype)]


class ApplyAdagradDKernelSpec:
    """Kernel / GEIR spec registered by CSV ``op_name`` and by GE type name."""

    golden = staticmethod(apply_adagrad_d_golden)
    third_party = {
        "torch": _ApplyAdagradDTorchCompose,
        "tf": _ApplyAdagradDTfCompose,
    }
    tolerance = _TOLERANCE


def _ref_fp64(var, accum, lr, grad, update_slots):
    """Independent fp64 element-wise reference — self-smoke only, not the golden.

    This is a test-side oracle used to cross-check the TensorFlow golden inside
    ``__main__``; the value actually returned to TTK never passes through here.
    """
    var = _to_f32(var).astype(np.float64)
    accum = _to_f32(accum).astype(np.float64)
    grad = _to_f32(grad).astype(np.float64)
    lr = np.float64(_scalar_f32(lr))
    accum_new = accum + grad * grad if update_slots else accum
    return var - lr * grad / np.sqrt(accum_new), accum_new


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    tf_mod = _tf()
    n = 32

    # TF golden vs independent fp64 oracle, both update_slots branches.
    for slots in (True, False):
        var = rng.uniform(-1, 1, n).astype("float32")
        accum = rng.uniform(0.25, 2.0, n).astype("float32")
        grad = rng.uniform(-0.5, 0.5, n).astype("float32")
        lr = np.array([0.01], "float32")
        outs = apply_adagrad_d_golden(var, accum, lr, grad, update_slots=slots)
        assert isinstance(outs, list) and len(outs) == 2, "must return 2 tensors"
        ref_var, ref_accum = _ref_fp64(var, accum, lr, grad, slots)
        assert outs[0].shape == var.shape and outs[0].dtype == var.dtype, "var meta"
        assert outs[1].shape == accum.shape and outs[1].dtype == accum.dtype, (
            "accum meta"
        )
        assert np.allclose(outs[0].astype(np.float64), ref_var, rtol=1e-6, atol=1e-7), (
            f"var mismatch update_slots={slots}"
        )
        assert np.allclose(
            outs[1].astype(np.float64), ref_accum, rtol=1e-6, atol=1e-7
        ), f"accum mismatch update_slots={slots}"

    # update_slots True/False must differ numerically (branch really exercised).
    var = np.array([0.5, -0.5, 2.0], "float32")
    accum = np.array([1.0, 0.4, 0.25], "float32")
    grad = np.array([0.2, -0.1, 0.5], "float32")
    lr = np.array([0.1], "float32")
    on = apply_adagrad_d_golden(var, accum, lr, grad, update_slots=True)
    off = apply_adagrad_d_golden(var, accum, lr, grad, update_slots=False)
    assert not np.allclose(on[0], off[0]), "update_slots branch gave identical var"
    assert not np.allclose(on[1], off[1]), "update_slots branch gave identical accum"

    # update_slots=False leaves accum untouched (kernel writes no accum_out; the
    # output aliases the input buffer in place).
    assert np.array_equal(off[1], accum), "accum must pass through when slots off"

    # Zero gradient: var unchanged, accum unchanged even with slots on.
    zero = apply_adagrad_d_golden(var, accum, lr, np.zeros_like(grad))
    assert np.allclose(zero[0], var) and np.allclose(zero[1], accum), "zero-grad"

    # State isolation: each call builds a fresh graph + fresh variables.
    again = apply_adagrad_d_golden(var, accum, lr, grad, update_slots=True)
    assert np.array_equal(again[0], on[0]) and np.array_equal(again[1], on[1]), (
        "state leaked across golden calls"
    )

    # inf / nan inputs must not crash (the st CSV carries such cases).
    wild_var = np.array([np.inf, -np.inf, np.nan, 1.0], "float32")
    wild_accum = np.array([0.0, 1.0, 2.0, 0.0], "float32")
    wild_grad = np.array([1.0, 0.0, 1.0, 1.0], "float32")
    wild = apply_adagrad_d_golden(wild_var, wild_accum, lr, wild_grad)
    assert len(wild) == 2 and wild[0].shape == wild_var.shape, "inf/nan shape"

    # Low-precision carriers: fp16 and bfloat16 round-trip to their own dtype.
    o16 = apply_adagrad_d_golden(
        var.astype("float16"),
        accum.astype("float16"),
        lr.astype("float16"),
        grad.astype("float16"),
    )
    assert o16[0].dtype == np.dtype("float16"), "fp16 output dtype"
    try:
        import ml_dtypes

        bf = ml_dtypes.bfloat16
        obf = apply_adagrad_d_golden(
            var.astype(bf), accum.astype(bf), lr.astype(bf), grad.astype(bf)
        )
        assert obf[0].dtype == np.dtype(bf), "bf16 output dtype"
    except ImportError:
        pass

    # third-party legs agree with the golden (independent compositions).
    tp_torch = _ApplyAdagradDTorchCompose()(var, accum, lr, grad, update_slots=True)
    assert np.allclose(
        tp_torch[0].numpy(), on[0], rtol=1e-6, atol=1e-7
    ) and np.allclose(tp_torch[1].numpy(), on[1], rtol=1e-6, atol=1e-7), "torch leg"
    tp_tf = _ApplyAdagradDTfCompose()(var, accum, lr, grad, update_slots=True)
    assert np.allclose(tp_tf[0].numpy(), on[0], rtol=1e-6, atol=1e-7) and np.allclose(
        tp_tf[1].numpy(), on[1], rtol=1e-6, atol=1e-7
    ), "tf leg"

    # Eager execution of the host process is untouched by the v1 sessions — this
    # is the regression the previous disable_v2_behavior() revision caused.
    assert tf_mod.executing_eagerly(), "global eager mode was disabled"

    # Rank regression (review): a 0-d carrier must stay 0-d on every leg.
    # np.ascontiguousarray promotes shape () to (1,), which would silently turn
    # a scalar state into a one-element vector in the numpy->torch bridge.
    _z = [np.array(_x, "float32") for _x in (1.0, 2.0, 0.5, 0.25)]
    _o0 = apply_adagrad_d_golden(*_z)
    assert [_o.shape for _o in _o0] == [(), ()], (
        f"0-d carrier promoted to {[_o.shape for _o in _o0]}"
    )
    assert [tuple(_t.shape) for _t in _ApplyAdagradDTorchCompose()(*_z)] == [(), ()], (
        "torch leg promoted the 0-d carrier"
    )
    _v = [np.array([_x], "float32") for _x in (1.0, 2.0, 0.5, 0.25)]
    _o1 = apply_adagrad_d_golden(*_v)
    assert [_o.shape for _o in _o1] == [(1,), (1,)], "(1,) carrier lost its rank"
    assert all(np.array_equal(_a.reshape(1), _b) for _a, _b in zip(_o0, _o1)), (
        "0-d and (1,) carriers disagree"
    )

    print("apply_adagrad_d golden self-smoke OK")


# 通路交付情况
# 已注册: kernel("apply_adagrad_d") + GEIR("ApplyAdagradD", 复用同一 spec 类)。
#   两键取自 CSV op_name 与 op_host/apply_adagrad_d_def.cpp 的 OP_ADD(ApplyAdagradD)。
# golden 口径: 两个输出全部取自 TensorFlow —— tf.raw_ops.ApplyAdagrad 在隔离
#   tf.Graph()+tf.compat.v1.Session(graph=...) 内以 use_resource=False 的 reference
#   variable 执行, 先 run 更新节点, 再在同一 session 读回 var/accum(ref 语义下 accum
#   已被原地更新)。上一版"var 走 TF、accum 走 numpy"的双口径已消除, 返回链上不再有
#   numpy 数学(numpy 仅作载体与 bf16 位桥)。节点类型断言 upd.op.type=="ApplyAdagrad"。
# eager 污染修复: 不再调用 tf.compat.v1.disable_v2_behavior() 等任何进程级全局开关。
#   该调用会污染同进程其它 golden —— 同仓 optim/apply_adam 与 optim/apply_ftrl 的
#   golden 末尾正好断言 tf.executing_eagerly()。本文件的 v1 session 只存在于单个函数
#   的 with 作用域内, 且显式传 use_resource=False(不继承可能已被别人翻掉的全局标志)。
# aclnn: 标准安装包不注册，但接口并非不存在 —— ABI 实测 libopapi.so 中
#   aclnnApplyAdagradDGetWorkspaceSize 命中 0（对照 aclnnApplyAdamWGetWorkspaceSize
#   命中 1，探测手法有效）；而 experimental/optim/apply_adagrad_d/op_host/
#   CMakeLists.txt 为 ACLNNTYPE aclnn，且 examples/test_aclnn_apply_adagrad_d.cpp
#   调用该接口 —— 接口由 experimental 包编译生成，需先构建并安装实验包。本环境
#   未构建/部署，状态记为待确认（缺项：实验包构建与部署），不能写成“接口不存在”。
# e2e: 不注册 —— 无独立前端证据。本算子是 D 变体(带 update_slots 属性、双输出),
#   与 stock tf.raw_ops.ApplyAdagrad 的输出契约不同, 不以同名前端冒充; 算子目录下
#   也没有 framework/ 插件。
# GPU 三方实跑: third_party 已声明 torch + tf 两条独立拼接腿(tf 腿用 add/mul/sqrt/div
#   组合, 刻意不复用 raw_ops.ApplyAdagrad 以免与 golden 同源), 但当前环境 ttk remote
#   未配置 endpoint, 三方比对结果记为未验证 —— 两条腿已在本文件自测块内用 CPU 载体
#   与 golden 对齐, 但"远端 GPU 上真跑过"未验证。
# tolerance: 统一 stat_rel_err(float32/float16/bfloat16)。不写 cross_check ——
#   无 GPU endpoint 时 TTK fail-closed; 需要三方交叉时由 CLI
#   --compare cross_check --provider torch|tf 开启。CSV 内每用例的
#   precision_tolerances / absolute_precision 走旧命名空间单独注入, 未被放宽。
