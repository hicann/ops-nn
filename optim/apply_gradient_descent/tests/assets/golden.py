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
"""TestSpec definitions for ApplyGradientDescent: Kernel / GEIR path.

Update rule (per @apply_gradient_descent_dag.h):
    var_out = var - delta * alpha

Outputs: exactly one — var — as declared by @apply_gradient_descent_def.cpp
(a single Output("var")) and by the st CSV (output_inplace_indexes="(0,)").
The operator has no attributes.

Registered names:
  - "apply_gradient_descent"  <- CSV op_name (Kernel path, numpy carrier)
  - "ApplyGradientDescent"    <- GE/REG_OP type name from
                                 OP_ADD(ApplyGradientDescent) (GEIR path, same
                                 spec class)

Benchmark: golden() funnels into the module-level
``_benchmark_apply_gradient_descent`` — one shared torch basic-op composition
owned by no spec. The previous revision computed ``var - alpha * delta`` with
NumPy operators, i.e. the reference arithmetic lived in the carrier itself; the
arithmetic now runs in an independent framework (torch) and NumPy is only the
input/output carrier bridge (bf16 via int16 bit view). The operator under test is
never called: no aclnn, no npu dispatch, no device kernel, no reuse of DUT
outputs.

Third party: torch (basic-op composition) and TensorFlow
(``tf.raw_ops.ResourceApplyGradientDescent`` — the real stock TF kernel for this
update, legitimately independent here because the golden runs on torch). The
operator ships framework/apply_gradient_descent_tf_plugin.cpp, so the TF front
end is a genuine competitor rather than a re-implementation.

Precision: BF16/FP16 carriers are up-cast to FP32 once, the update runs in FP32
(mirroring the kernel's Cast<float> DAG), then the output is cast back to the
input dtype. alpha is index-extracted as a scalar because the kernel consumes it
through Vec::Duplicate(ScalarAttr) — element 0 broadcast over the tile — and the
st CSV declares it with shape (1,) against var ranks from 1 to 8.

Import rule: torch / tensorflow / ml_dtypes are imported lazily inside function
bodies only, so plugin discovery stays cheap and no TF import happens for specs
that do not need it.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

__spec__ = {
    "apply_gradient_descent": "ApplyGradientDescentKernelSpec",
    "ApplyGradientDescent": "ApplyGradientDescentKernelSpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Both may coexist; ``__spec__`` wins.
__golden__ = {"kernel": {"apply_gradient_descent": "apply_gradient_descent_golden"}}

# Community default standard (stat_rel_err). cross_check is deliberately NOT
# pinned: it is a three-way comparison and TTK fails closed when no third-party
# endpoint is alive. third_party is declared below, so
# `--compare cross_check --provider torch|tf` enables three-way comparison
# whenever a GPU endpoint exists. The CSV per-case absolute_precision (1e-08) is
# injected separately (legacy namespace) and is left untouched — no tolerance is
# relaxed here.
_TOLERANCE = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
}

_TF = None


def _tf():
    """Lazily import TensorFlow (quiet) — only the TF third-party leg needs it."""
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


def _np_to_torch(x):
    """numpy ndarray -> torch tensor, bit-exact (bfloat16 via int16 bit view)."""
    import torch

    # np.ascontiguousarray would promote shape () to (1,), turning a scalar
    # carrier into a one-element vector; a 0-d array is contiguous already.
    arr = np.asarray(x)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy(order="C")
    bf16 = _bf16_dtype(arr.dtype)
    if bf16 is not None:
        if arr.dtype != bf16:
            arr = arr.view(bf16)
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_to_np(t, dtype):
    """torch tensor -> numpy ndarray of ``dtype``, bit-exact for bfloat16."""
    import torch

    t = t.detach().cpu()
    dt = np.dtype(dtype)
    bf16 = _bf16_dtype(dt)
    if bf16 is not None:
        out = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(bf16)
        return out if dt == bf16 else out.view(dt)
    return t.numpy().astype(dt, copy=False)


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


def _benchmark_apply_gradient_descent(var, alpha, delta):
    """Shared benchmark gradient descent — module-level, owned by no spec.

    Pure torch basic-op composition evaluated in FP32 (``torch.sub`` /
    ``torch.mul``), mirroring the kernel's Cast<float> DAG. Returns one torch
    FP32 tensor; carrier / dtype restoration is the caller's job. alpha is
    reduced to a true scalar first so the (1,)-shaped CSV scalar can never be
    mistaken for an element-wise operand of a (1,)-shaped var. Device is
    preserved — nothing is detoured through host or NumPy. Never touches the
    operator under test.
    """
    import torch

    var_t, delta_t = _as_torch(var), _as_torch(delta)
    alpha_t = _as_torch(alpha).reshape(-1)[0]

    # Promote-aware: a float64 reference (TTK Promote / cross_check) must stay
    # float64, otherwise FP32 cancellation is baked into the "high precision"
    # reference. Low-precision and float32 carriers still compute in float32,
    # mirroring the kernel's Cast<float> DAG.
    calc = torch.float64 if var_t.dtype == torch.float64 else torch.float32
    var_f = var_t.to(calc)
    delta_f = delta_t.to(calc)
    alpha_f = alpha_t.to(calc)
    return torch.sub(var_f, torch.mul(delta_f, alpha_f))


def apply_gradient_descent_golden(var, alpha, delta, **kwargs):
    """Kernel / GEIR golden (numpy carrier in, numpy carrier out).

    Parameter names and order follow @apply_gradient_descent_def.cpp inputs
    (var, alpha, delta) without the output; the operator declares no attributes.
    Returns [var_out] — one tensor, matching the op def's single explicit output
    and the st CSV output_inplace_indexes="(0,)".

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name
    """
    del kwargs
    var_arr = np.asarray(var)
    if var_arr.size == 0:
        return [var_arr.copy()]

    out = _benchmark_apply_gradient_descent(var, alpha, delta)
    return [_torch_to_np(out, var_arr.dtype)]


class _ApplyGradientDescentTorchCompose:
    """torch third-party provider (spec-mode class form).

    Runs on the remote XPU endpoint, binding by parameter name against the
    operator's own input names. Shares the benchmark with the golden, so the GPU
    leg answers the same composition the host answered — the comparison it adds
    is hardware (XPU vs host), not formula. Everything stays on the incoming
    tensors' device: no .cpu(), no numpy arithmetic.
    """

    def __call__(self, var, alpha, delta, **kwargs):
        del kwargs
        out = _benchmark_apply_gradient_descent(var, alpha, delta)
        return [out.to(_as_torch(var).dtype)]


class _ApplyGradientDescentTfCompose:
    """TensorFlow third-party provider (spec-mode class form).

    Reference: ``tf.raw_ops.ResourceApplyGradientDescent`` — the genuine stock
    TF kernel for this update, and an independent competitor because the golden
    runs on torch. The resource variant is used rather than
    ``tf.raw_ops.ApplyGradientDescent`` for a measured reason: the ref-variable
    op rejects eager execution outright ("op does not support eager execution.
    Arg 'out' is a ref"), and the only way to run it would be a v1
    Graph+Session, whose ``session.run`` always returns host arrays — that would
    move the third-party arithmetic off the endpoint's device, which this leg
    must not do. ResourceApplyGradientDescent runs eagerly, in place, on the
    the endpoint placed the inputs on, and ``read_value()`` keeps the result
    device-resident.

    Computed in FP32 then cast back, mirroring the kernel's own up-cast, so the
    leg is comparable against the golden instead of drifting on bf16 rounding.
    """

    def __call__(self, var, alpha, delta, **kwargs):
        tf = _tf()
        del kwargs
        var_t = tf.convert_to_tensor(var)
        out_dtype = var_t.dtype
        var_ref = tf.Variable(tf.cast(var_t, tf.float32), trainable=False)
        alpha_s = tf.reshape(tf.cast(tf.convert_to_tensor(alpha), tf.float32), [-1])[0]
        delta_f = tf.cast(tf.convert_to_tensor(delta), tf.float32)
        upd = tf.raw_ops.ResourceApplyGradientDescent(
            var=var_ref.handle, alpha=alpha_s, delta=delta_f, use_locking=False
        )
        del upd  # ref op: the result is the in-place mutation of var_ref
        return [tf.cast(var_ref.read_value(), out_dtype)]


class ApplyGradientDescentKernelSpec:
    """Kernel / GEIR spec registered by CSV ``op_name`` and by GE type name."""

    golden = staticmethod(apply_gradient_descent_golden)
    third_party = {
        "torch": _ApplyGradientDescentTorchCompose,
        "tf": _ApplyGradientDescentTfCompose,
    }
    tolerance = _TOLERANCE


if __name__ == "__main__":
    # ---- self-smoke: value / shape / dtype / boundary ----
    # The golden path is torch; NumPy below is only the test-side oracle and the
    # carrier, never part of the value returned to TTK.

    # value / shape / dtype across the three declared dtypes and several ranks
    for dt in ("float32", "float16"):
        for shape in ((4,), (37, 31), (1, 1, 1, 7)):
            v = np.arange(int(np.prod(shape)), dtype=dt).reshape(shape) - 3.0
            a = np.array([0.1], dtype=dt)
            d = (np.arange(int(np.prod(shape)), dtype=dt).reshape(shape) * 0.5) - 1.0
            outs = apply_gradient_descent_golden(v, a, d)
            assert isinstance(outs, list) and len(outs) == 1, "must return 1 tensor"
            out = outs[0]
            ref = v.astype(np.float32) - np.float32(a[0]) * d.astype(np.float32)
            assert out.shape == v.shape, "shape mismatch"
            assert out.dtype == v.dtype, "dtype mismatch"
            assert np.allclose(
                out.astype(np.float32),
                ref.astype(dt).astype(np.float32),
                rtol=1e-3,
                atol=1e-3,
            ), f"value mismatch dt={dt} shape={shape}"

    # bfloat16 carrier: bit-exact numpy<->torch bridge, output keeps its dtype
    try:
        import ml_dtypes

        bf = ml_dtypes.bfloat16
        vb = np.array([1.0, -2.0, 0.5], dtype=bf)
        ab = np.array([0.25], dtype=bf)
        db = np.array([2.0, -1.0, 4.0], dtype=bf)
        ob = apply_gradient_descent_golden(vb, ab, db)[0]
        assert ob.dtype == np.dtype(bf), "bf16 output dtype"
        assert np.allclose(
            ob.astype(np.float32),
            vb.astype(np.float32) - 0.25 * db.astype(np.float32),
            rtol=1e-2,
            atol=1e-2,
        ), "bf16 value mismatch"
    except ImportError:
        pass

    # boundary: zero gradient -> var unchanged (bit-exact, not just close)
    v = np.array([1.0, 2.0, 3.0], dtype="float32")
    d = np.zeros_like(v)
    out = apply_gradient_descent_golden(v, np.array([0.9], "float32"), d)[0]
    assert np.array_equal(out, v), "zero-grad invariant violated"

    # boundary: extreme learning rate
    out = apply_gradient_descent_golden(
        np.array([1.0], "float32"),
        np.array([1e4], "float32"),
        np.array([1.0], "float32"),
    )[0]
    assert np.isclose(float(out[0]), 1.0 - 1e4), "extreme alpha failed"

    # boundary: alpha=0 is a no-op; alpha shape (1,) must broadcast as a scalar
    # even when var itself is (1,)-shaped (never element-wise by accident)
    out = apply_gradient_descent_golden(
        np.array([5.0], "float32"),
        np.array([0.0], "float32"),
        np.array([7.0], "float32"),
    )[0]
    assert np.array_equal(out, np.array([5.0], "float32")), "alpha=0 must be a no-op"

    # inf / nan inputs must not crash
    wild = apply_gradient_descent_golden(
        np.array([np.inf, -np.inf, np.nan, 1.0], "float32"),
        np.array([0.5], "float32"),
        np.array([1.0, 1.0, 1.0, np.inf], "float32"),
    )[0]
    assert wild.shape == (4,), "inf/nan shape"

    # third-party legs agree with the golden (torch leg shares the benchmark;
    # the TF leg is the independent stock ResourceApplyGradientDescent kernel)
    v = np.array([1.0, -2.0, 0.5, 3.0], "float32")
    a = np.array([0.1], "float32")
    d = np.array([2.0, -1.0, 0.0, 4.0], "float32")
    base = apply_gradient_descent_golden(v, a, d)[0]
    tp_torch = _ApplyGradientDescentTorchCompose()(v, a, d)
    assert len(tp_torch) == 1 and np.allclose(
        tp_torch[0].numpy(), base, rtol=1e-6, atol=1e-7
    ), "torch leg"
    tf_mod = _tf()
    tp_tf = _ApplyGradientDescentTfCompose()(v, a, d)
    assert len(tp_tf) == 1 and np.allclose(
        tp_tf[0].numpy(), base, rtol=1e-6, atol=1e-7
    ), "tf leg"

    # the TF leg must not have flipped any process-global execution mode
    assert tf_mod.executing_eagerly(), "global eager mode was disabled"

    # Promote regression (review): a float64 reference must stay float64.
    # FP32 cancellation would make 1e8 - (1+2^-23)*99999992 = 0 instead of
    # -3.9209280014038086; the promoted golden must not downcast.
    v64 = np.array([1e8], np.float64)
    a64 = np.array([1.0 + 2.0**-23], np.float64)
    d64 = np.array([99999992.0], np.float64)
    out64 = apply_gradient_descent_golden(v64, a64, d64)[0]
    assert out64.dtype == np.float64, "promoted golden lost float64"
    ref64 = v64 - a64 * d64
    assert np.array_equal(out64, ref64), (
        f"promoted golden downcast to FP32: {out64} != {ref64}"
    )

    # Rank regression (review): a 0-d carrier must stay 0-d on every leg.
    # np.ascontiguousarray promotes shape () to (1,), which would silently turn
    # a scalar var into a one-element vector in the numpy->torch bridge.
    _v0 = np.array(1.0, "float32")
    _a0 = np.array(0.5, "float32")
    _d0 = np.array(2.0, "float32")
    _o0 = apply_gradient_descent_golden(_v0, _a0, _d0)[0]
    assert _o0.shape == (), f"0-d carrier promoted to {_o0.shape}"
    assert tuple(_ApplyGradientDescentTorchCompose()(_v0, _a0, _d0)[0].shape) == (), (
        "torch leg promoted the 0-d carrier"
    )
    assert tuple(_ApplyGradientDescentTfCompose()(_v0, _a0, _d0)[0].shape) == (), (
        "tf leg promoted the 0-d carrier"
    )
    _o1 = apply_gradient_descent_golden(np.array([1.0], "float32"), _a0, _d0)[0]
    assert _o1.shape == (1,), "(1,) carrier lost its rank"
    assert np.array_equal(_o0.reshape(1), _o1), "0-d and (1,) carriers disagree"

    # Subnormal semantics (review): the gd_subnormal* ST cases must be able to
    # fail. ApplyGradientDescent's arch35 DAG is Vec::Mul + Vec::Sub on the
    # vector pipe and selects no FTZ algorithm, so FP32 subnormals survive both
    # as operands and as results; --cce-ftz only governs the algorithm-selected
    # paths (INTRINSIC / DIFF_COMPENSATION, e.g. Reg::Div) that this operator
    # never takes. Measured on Ascend950PR_9589 / CANN 9.3.0: device output is
    # bit-identical to the references asserted below.
    def _ttk_close_pass(device, golden_v, rtol, atol, ptol):
        """Mirror of TTK CloseComparison: rtol==atol==0 is the exact branch."""
        dv = np.asarray(device, "float32").reshape(-1)
        gv = np.asarray(golden_v, "float32").reshape(-1)
        if rtol == 0 and atol == 0:
            ok = dv == gv
        else:
            ok = np.abs(dv - gv) <= atol + rtol * np.abs(gv)
        return (1.0 - ok.sum() / ok.size) <= ptol

    # name, var, alpha, delta, (rtol, atol, ptol), expected bits, flushed value
    _subnormal_cases = (
        ("gd_subnormal", 1e-40, 1e-40, 1e-40, (0.0, 0.0, 0.0), 0x000116C2, 0.0),
        (
            "gd_subnormal_in_amplified",
            1.0,
            1e38,
            1e-40,
            (1e-4, 1e-4, 1e-4),
            0x3F7D70A5,
            1.0,
        ),
        (
            "gd_subnormal_out_exact",
            2e-38,
            1.0,
            1.9e-38,
            (0.0, 0.0, 0.0),
            0x000AE398,
            0.0,
        ),
    )
    for _nm, _v, _a, _d, (_rt, _at, _pt), _bits, _flushed_val in _subnormal_cases:
        _ref = apply_gradient_descent_golden(
            np.full((8,), _v, "float32"),
            np.array([_a], "float32"),
            np.full((8,), _d, "float32"),
        )[0]
        _got = int(_ref.view(np.uint32)[0])
        # Guards the host too: a reference path that flushed subnormals would
        # make these ST cases vacuous instead of failing loudly here.
        assert _got == _bits, (
            f"{_nm}: reference is 0x{_got:08x}, expected 0x{_bits:08x} — the host "
            "FP32 path no longer reproduces the recorded device result"
        )
        assert _ttk_close_pass(_ref, _ref, _rt, _at, _pt), f"{_nm}: self-consistency"
        # Counter-example: a device that flushed the subnormal must not pass.
        _flushed = np.full_like(_ref, _flushed_val)
        assert not _ttk_close_pass(_flushed, _ref, _rt, _at, _pt), (
            f"{_nm}: a flushed device result ({_flushed_val}) still passes — "
            "the case has no discriminating power"
        )

    # The pre-review gd_subnormal tolerance (atol=1e-8) could not fail: that is
    # why the case is now exact. Kept as a regression on the criterion itself.
    _vac = apply_gradient_descent_golden(
        np.full((8,), 1e-40, "float32"),
        np.array([1e-40], "float32"),
        np.full((8,), 1e-40, "float32"),
    )[0]
    assert _ttk_close_pass(np.zeros_like(_vac), _vac, 1e-4, 1e-8, 1e-4), (
        "atol=1e-8 is expected to be blind to a zeroed subnormal output"
    )

    print("apply_gradient_descent golden self-smoke OK")


# 通路交付情况
# 已注册: kernel("apply_gradient_descent") + GEIR("ApplyGradientDescent", 复用同一
#   spec 类)。两键取自 CSV op_name 与 op_host/apply_gradient_descent_def.cpp 的
#   OP_ADD(ApplyGradientDescent)。
# golden 口径: 改为 torch 基础算子(torch.sub/torch.mul)在 FP32 下组合, NumPy 只做
#   载体与 bf16 位桥(int16 位视图, bit-exact), 返回链上不再有 NumPy 数学。上一版
#   直接用 NumPy 算 var-alpha*delta, 等价于"真值与载体同源", 已消除。等价性由文件内
#   自测块逐 dtype/逐 rank 与 NumPy 口径对齐验证(自测块里的 NumPy 只是测试侧 oracle,
#   不在正式返回链上)。
# aclnn: 待确认/受阻, 不注册 —— ABI 实测 nm -D libopapi.so 中
#   aclnnApplyGradientDescentGetWorkspaceSize 命中数为 0(同一条命令对
#   aclnnApplyAdamWGetWorkspaceSize 命中 1, 证明探测手法有效)。
#   experimental/optim/apply_gradient_descent 下确实有 op_api/ 与
#   docs/aclnnApplyGradientDescent.md, 但那是实验自定义包, 标准安装包未部署(以上 ABI
#   实测为证); 源码存在不等于标准包已部署, 故不注册。解除阻塞的条件: ①确认实验包是否
#   纳入标准部署; ②在部署后的 libopapi.so 上复测符号命中; ③核对接口签名与 ABI(参数
#   名/可选参数/原位语义)与 docs 一致。三条都满足再补 aclnn spec。
# e2e: 不注册(前端命中未验证) —— 算子目录下有
#   framework/apply_gradient_descent_tf_plugin.cpp, 且 tf.raw_ops.
#   ApplyGradientDescent 是真实 stock TF 前端, 所以 TF 前端本身是成立的; 但要把 TF 图
#   下沉到本算子需要 Ascend TF adapter(npu_device / tfplugin), 当前环境未安装, 无法
#   验证"TF 图真的落到本算子内核"。按不伪造原则, 不写 E2E 注册。
# GPU 三方实跑: third_party 已声明 torch + tf 两条腿(tf 腿用
#   tf.raw_ops.ResourceApplyGradientDescent —— 用 resource 变体是实测结论: ref 变体
#   tf.raw_ops.ApplyGradientDescent 在 eager 下直接抛
#   "op does not support eager execution. Arg 'out' is a ref", 只能靠 v1
#   Graph+Session 跑, 而 session.run 必然把结果取回 host, 等于把三方计算搬离 endpoint
#   设备, 违反"third_party 不下设备"的要求)。当前环境 ttk remote 未配置 endpoint, 三方
#   比对结果记为未验证 —— 两条腿已在本文件自测块内用 CPU 载体与 golden 对齐, 但
#   "远端 GPU 上真跑过"未验证。
# 次正规(subnormal)语义: 本算子 arch35 DAG 只用 Vec::Mul + Vec::Sub(无 FTZ 选择型
#   算法), --cce-ftz 只约束算法选择型路径(INTRINSIC / DIFF_COMPENSATION, 如
#   Reg::Div), 故 FP32 次正规数在算子入参与结果上都保留。Ascend950PR_9589 /
#   CANN 9.3.0 实测: gd_subnormal(0x000116c2)、gd_subnormal_in_amplified
#   (0x3f7d70a5)、gd_subnormal_out_exact(0x000ae398) 三条设备输出与参考位级一致,
#   且两条次正规输出用例设备侧 8/8 元素仍为次正规(未被清零)。原 gd_subnormal 的
#   atol=1e-8 对"输出被清零"无区分力(差值仅 1e-40), 已改为零容忍精确比对;
#   in_amplified 隔离"入参被清零"(放大到正规区, 清零则相对差 1e-2 >> 容差),
#   out_exact 隔离"结果被清零"(入参均为正规, Sterbenz 差值精确落在次正规区)。
#   自测块内含反例校验: 三条用例一旦被清零, 判据必须失败。
# tolerance: 统一 stat_rel_err(float32/float16/bfloat16)。不写 cross_check ——
#   无 GPU endpoint 时 TTK fail-closed; 需要三方交叉时由 CLI
#   --compare cross_check --provider torch|tf 开启。CSV 内每用例的
#   absolute_precision(1e-08) 走旧命名空间单独注入, 未被放宽。
