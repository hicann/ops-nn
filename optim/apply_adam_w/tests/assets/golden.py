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
"""TestSpec definitions for ApplyAdamW: Kernel/GEIR, ACLNN and Torch E2E.

Update rule (per docs/aclnnApplyAdamW.md; the previous golden's operation order
is preserved verbatim, only the executing framework changed):
    g_t   = -grad if maximize else grad
    m_t   = beta1 * m + (1 - beta1) * g_t
    v_t   = beta2 * v + (1 - beta2) * g_t * g_t
    b1_t  = beta1_power * beta1     (beta-power protocol, NOT a step counter)
    b2_t  = beta2_power * beta2
    v_eff = max(max_grad_norm, v_t) if amsgrad else v_t
    denom = sqrt(v_eff / (1 - b2_t)) + epsilon
    var   = var * (1 - lr * weight_decay) - lr * m_t / (1 - b1_t) / denom

Outputs: exactly three — var / m / v — as declared by @apply_adam_w_def.cpp
(Output("var"), Output("m"), Output("v")) and by the st CSV
(output_inplace_indexes="(0, 1, 2)"). The amsgrad running max is internal state
of the update, not an output of this operator, so it is never returned.

Registered names:
  - "apply_adam_w" / "ApplyAdamW"  <- CSV op_name (Kernel / GEIR path, numpy
                                      carrier; snake_case is the CSV dialect,
                                      CamelCase the GE/REG_OP type name)
  - "aclnnApplyAdamW"             <- CSV api_name (ACLNN path, torch carrier).
                                      Deployment verified by ABI probe:
                                      aclnnApplyAdamWGetWorkspaceSize is defined
                                      in the installed libopapi.so, and this
                                      operator ships op_api/aclnn_apply_adam_w.
  - "torch_npu.npu_apply_adam_w"  <- CSV api_name (Torch E2E path). Verified
                                      present in torch_npu 2.7.1.post6; the
                                      schema passes var/m/v through the ``out``
                                      list of the ``.out`` overload.

Benchmark: every spec golden() funnels into the module-level
``_benchmark_apply_adam_w`` — one shared torch basic-op composition owned by no
spec. Independence: torch-only, never calls the operator under test (no aclnn,
no npu dispatch, no device kernel, no reuse of DUT outputs). NumPy appears only
as an input/output carrier bridge (bf16 via int16 bit view) and never carries
the reference arithmetic.

Precision: BF16/FP16 carriers are up-cast to FP32 once, the whole update runs in
FP32 (mirroring the kernel), then each output is cast back to its input dtype.
Scalars keep their declared dtype and broadcast naturally, so the (1,)-shaped
scalars of the st CSV are not index-extracted.

Import rule: torch / ml_dtypes are imported lazily inside function bodies only,
so plugin discovery stays cheap.
"""

__spec__ = {
    "apply_adam_w": "ApplyAdamWKernelSpec",
    "ApplyAdamW": "ApplyAdamWKernelSpec",
    "aclnnApplyAdamW": "ApplyAdamWAclnnSpec",
    "torch_npu.npu_apply_adam_w": "ApplyAdamWTorchE2ESpec",
    # 真实可执行入口：torch_npu 仅实现 .out 重载（default 报 not implemented）。
    "torch.ops.npu.npu_apply_adam_w.out": "ApplyAdamWTorchE2ESpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Only the kernel entry is mirrored here —
# the aclnn / E2E goldens take torch carriers and different parameter names, so
# mapping them into the legacy dict would misbind.
__golden__ = {"kernel": {"apply_adam_w": "apply_adam_w_golden"}}

# Community default standard (stat_rel_err). cross_check is deliberately NOT
# pinned: it is a three-way comparison and TTK fails closed with GOLDEN_FAILURE
# when no third-party endpoint is alive. third_party is declared below, so
# `--compare cross_check --provider torch` enables three-way comparison whenever
# a GPU endpoint exists. The CSV per-case precision_tolerances /
# absolute_precision are injected separately (legacy namespace) and are left
# untouched — no tolerance is relaxed here.
_TOLERANCE = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
}


def _np_to_torch(x):
    """numpy ndarray -> torch tensor, bit-exact (bfloat16 via int16 bit view)."""
    import numpy as np
    import torch

    arr = np.asarray(x)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_to_np(t, dtype):
    """torch tensor -> numpy ndarray of ``dtype``, bit-exact for bfloat16."""
    import numpy as np
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


def _benchmark_apply_adam_w(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    grad,
    max_grad_norm=None,
    amsgrad=False,
    maximize=False,
):
    """Shared benchmark AdamW — module-level, owned by no spec.

    Pure torch basic-op composition evaluated in FP32. Returns the three torch
    FP32 tensors (var, m, v) in op-def output order; carrier/dtype restoration
    is the caller's job. Never touches the operator under test.
    """
    import torch

    def f32(x):
        t = _as_torch(x)
        if t is None:
            return None
        # Promote-aware: keep a promoted float64 reference exact.
        return t if t.dtype == torch.float64 else t.to(torch.float32)

    def scalar32(x):
        """Read one of the seven coefficients as a 0-d tensor.

        ApplyAdamWTiling::CheckIsScalar accepts any shape holding a single
        element and the kernel reads one value, so a (1, 1) carrier must not
        broadcast the rank of var/m/v. max_grad_norm is not in this set: it is
        shape-checked against var and stays element-wise.
        """
        t = f32(x)
        if t is None or t.dim() == 0:
            return t
        if t.numel() != 1:
            raise ValueError(
                "ApplyAdamW scalar input must hold one element, got shape "
                f"{tuple(t.shape)}"
            )
        return t.reshape(())

    var_f, m_f, v_f, grad_f, mgn_f = (
        f32(var),
        f32(m),
        f32(v),
        f32(grad),
        f32(max_grad_norm),
    )
    b1p, b2p, lr_f, wd_f, b1_f, b2_f, eps_f = (
        scalar32(beta1_power),
        scalar32(beta2_power),
        scalar32(lr),
        scalar32(weight_decay),
        scalar32(beta1),
        scalar32(beta2),
        scalar32(epsilon),
    )

    g = -grad_f if maximize else grad_f
    m_out = m_f * b1_f + (1.0 - b1_f) * g
    v_out = v_f * b2_f + (1.0 - b2_f) * g * g

    # beta-power protocol: the powers are multiplied by beta, never derived from
    # a step counter (do not swap in the D / W-V2 step conventions).
    b1p_out = b1p * b1_f
    b2p_out = b2p * b2_f

    if amsgrad and mgn_f is not None:
        v_eff = torch.maximum(mgn_f, v_out)
    else:
        v_eff = v_out
    denom = torch.sqrt(v_eff / (1.0 - b2p_out)) + eps_f

    # Operation order is kept byte-for-byte equivalent to the previous golden:
    # var_t = var * (1 + (-lr * wd)); m_t = m_out / (b1_t - 1); then
    # var_t + lr * m_t / denom. Note (b1_t - 1) < 0, so the trailing "+" is the
    # subtraction of lr * m_hat / denom. Re-associating this (e.g. to
    # var * (1 - lr*wd) - lr * m_out / (1 - b1_t) / denom) is algebraically
    # identical but shifts FP32 rounding by ~1e-6 relative, so it is avoided.
    var_t = var_f * (1.0 + (-lr_f * wd_f))
    m_t = m_out / (b1p_out - 1.0)
    var_out = var_t + lr_f * m_t / denom
    return var_out, m_out, v_out


def apply_adam_w_golden(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    grad,
    max_grad_norm=None,
    amsgrad: bool = False,
    maximize: bool = False,
    **kwargs,
):
    """Kernel / GEIR golden (numpy carrier in, numpy carrier out).

    Parameter names and order follow @apply_adam_w_def.cpp inputs, then the
    attributes. Returns [var, m, v] — three tensors, matching the op def and the
    st CSV output_inplace_indexes="(0, 1, 2)".
    """
    import numpy as np

    outs = _benchmark_apply_adam_w(
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm,
        amsgrad,
        maximize,
    )
    return [
        _torch_to_np(res, np.asarray(src).dtype) for res, src in zip(outs, (var, m, v))
    ]


class _ApplyAdamWTorchCompose:
    """torch third-party provider (spec-mode class form).

    Runs on the remote XPU endpoint, binding by parameter name against the
    operator's own input/attribute names. The GPU leg stays on its device — the
    benchmark keeps torch tensors where they are, so nothing is detoured through
    CPU or NumPy — and the operator under test is never called.
    """

    def __call__(
        self,
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm=None,
        amsgrad: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        outs = _benchmark_apply_adam_w(
            var,
            m,
            v,
            beta1_power,
            beta2_power,
            lr,
            weight_decay,
            beta1,
            beta2,
            epsilon,
            grad,
            max_grad_norm,
            amsgrad,
            maximize,
        )
        srcs = [_as_torch(x) for x in (var, m, v)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]


def _scalar_like_state(value, state):
    """Round a scalar to the state tensor's dtype, as the real entry does.

    @apply_adam_w_def.cpp declares the seven scalar inputs with the same
    ``dataType`` as var/m/v, so a bf16/fp16 entry consumes rounded scalars. The
    E2E schema hands Python floats, which would otherwise stay FP32 and drift
    by ~1 ULP from the device. FP32 state is a no-op.
    """
    import torch

    t = _as_torch(state)
    if t.dtype == torch.float64:
        # Already-promoted golden/carrier: keep the scalar at float64.
        s = _as_torch(value).to(torch.float64).reshape(-1)
        return s[0] if s.numel() else torch.zeros((), dtype=torch.float64)
    s = _as_torch(value).to(torch.float32).reshape(-1)
    s = s[0] if s.numel() else torch.zeros((), dtype=torch.float32)
    if t.dtype in (torch.bfloat16, torch.float16):
        s = s.to(t.dtype).to(torch.float32)
    return s


class _ApplyAdamWAclnnTorchCompose:
    """ACLNN torch third-party provider (aclnnApplyAdamWGetWorkspaceSize names).

    Binds varRef/mRef/vRef, beta1Power/beta2Power, weightDecay, ``eps`` and
    ``maxGradNormOptional`` (not the kernel spellings). Returns the three
    in-place states in op-def order.
    """

    def __call__(
        self,
        varRef,
        mRef,
        vRef,
        beta1Power,
        beta2Power,
        lr,
        weightDecay,
        beta1,
        beta2,
        eps,
        grad,
        maxGradNormOptional=None,
        amsgrad: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        outs = _benchmark_apply_adam_w(
            varRef,
            mRef,
            vRef,
            beta1Power,
            beta2Power,
            lr,
            weightDecay,
            beta1,
            beta2,
            eps,
            grad,
            maxGradNormOptional,
            amsgrad,
            maximize,
        )
        srcs = [_as_torch(x) for x in (varRef, mRef, vRef)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]


class _ApplyAdamWE2ETorchCompose:
    """Torch E2E third-party provider (real ``.out`` schema).

    var/m/v are not positional inputs: they arrive through ``out``. Scalars are
    rounded to the state dtype first, matching @apply_adam_w_def.cpp. Returns
    the three updated states.
    """

    def __call__(
        self,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm=None,
        amsgrad: bool = False,
        maximize: bool = False,
        out=None,
        **kwargs,
    ):
        if out is None or len(out) < 3:
            raise ValueError(
                "npu_apply_adam_w third_party needs out=(var, m, v) from the .out schema"
            )
        var, m, v = out[0], out[1], out[2]
        outs = _benchmark_apply_adam_w(
            var,
            m,
            v,
            _scalar_like_state(beta1_power, var),
            _scalar_like_state(beta2_power, var),
            _scalar_like_state(lr, var),
            _scalar_like_state(weight_decay, var),
            _scalar_like_state(beta1, var),
            _scalar_like_state(beta2, var),
            _scalar_like_state(epsilon, var),
            grad,
            max_grad_norm,
            amsgrad,
            maximize,
        )
        srcs = [_as_torch(x) for x in (var, m, v)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]


class ApplyAdamWKernelSpec:
    """Kernel / GEIR spec registered by CSV ``op_name`` (numpy carrier)."""

    golden = staticmethod(apply_adam_w_golden)
    third_party = {"torch": _ApplyAdamWTorchCompose}
    tolerance = _TOLERANCE


class ApplyAdamWAclnnSpec:
    """ACLNN spec registered by CSV ``api_name`` "aclnnApplyAdamW".

    Parameter names follow aclnnApplyAdamWGetWorkspaceSize exactly: varRef /
    mRef / vRef are the in-place state triple, ``eps`` (not ``epsilon``) is the
    aclnn name, and ``maxGradNormOptional`` is the optional amsgrad state.
    Returns the same three updated states in op-def order.
    """

    def golden(
        varRef,
        mRef,
        vRef,
        beta1Power,
        beta2Power,
        lr,
        weightDecay,
        beta1,
        beta2,
        eps,
        grad,
        maxGradNormOptional=None,
        amsgrad: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        outs = _benchmark_apply_adam_w(
            varRef,
            mRef,
            vRef,
            beta1Power,
            beta2Power,
            lr,
            weightDecay,
            beta1,
            beta2,
            eps,
            grad,
            maxGradNormOptional,
            amsgrad,
            maximize,
        )
        srcs = [_as_torch(x) for x in (varRef, mRef, vRef)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]

    third_party = {"torch": _ApplyAdamWAclnnTorchCompose}
    tolerance = _TOLERANCE


class ApplyAdamWTorchE2ESpec:
    """Torch E2E spec registered by CSV ``api_name`` "torch_npu.npu_apply_adam_w".

    Bound to the real torch_npu schema (read back from torch_npu 2.7.1.post6)::

        npu_apply_adam_w(Scalar beta1_power, Scalar beta2_power, Scalar lr,
                         Scalar weight_decay, Scalar beta1, Scalar beta2,
                         Scalar epsilon, Tensor grad, Tensor? max_grad_norm,
                         bool? amsgrad, bool? maximize) -> (Tensor, Tensor, Tensor)

    var / m / v are not positional inputs of this schema: the in-place state
    triple is supplied through the ``out`` list of the ``.out`` overload, which
    is why ``out`` carries var/m/v here. The three returned tensors are the
    updated var / m / v — all three in-place states participate in comparison,
    not just the main weight.
    """

    def golden(
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm=None,
        amsgrad: bool = False,
        maximize: bool = False,
        out=None,
        **kwargs,
    ):
        if out is None or len(out) < 3:
            raise ValueError(
                "npu_apply_adam_w golden needs out=(var, m, v): the schema "
                "passes the in-place state triple through the out list"
            )
        var, m, v = out[0], out[1], out[2]
        # @apply_adam_w_def.cpp: scalars follow var/m/v dataType; the E2E schema
        # hands Python floats, so round them exactly like the device entry does.
        outs = _benchmark_apply_adam_w(
            var,
            m,
            v,
            _scalar_like_state(beta1_power, var),
            _scalar_like_state(beta2_power, var),
            _scalar_like_state(lr, var),
            _scalar_like_state(weight_decay, var),
            _scalar_like_state(beta1, var),
            _scalar_like_state(beta2, var),
            _scalar_like_state(epsilon, var),
            grad,
            max_grad_norm,
            amsgrad,
            maximize,
        )
        srcs = [_as_torch(x) for x in (var, m, v)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]

    third_party = {"torch": _ApplyAdamWE2ETorchCompose}
    tolerance = _TOLERANCE


# 通路交付情况
# 已注册: kernel + GEIR + aclnn + e2e(torch_npu .out 重载)。
# aclnn: aclnnApplyAdamWGetWorkspaceSize 标准包有符号；provider 按
#   varRef/mRef/vRef/beta1Power/beta2Power/weightDecay/eps/maxGradNormOptional
#   绑定，返回三态。
# e2e: 用例钉 torch.ops.npu.npu_apply_adam_w.out；out 承载 var/m/v。def.cpp
#   的七个标量与状态同 dataType，E2E golden/provider 用 _scalar_like_state
#   复现按张量 dtype 的舍入(FP32 无变化)。
# third_party: kernel/ACLNN/E2E 各用独立 provider；E2E 从 out 取状态并返回
#   三输出；torch 张量保持 device，不绕 CPU/numpy。
# 验证: 本地 CPU xpu-server ACLNN 2/2、E2E 2/2 cross_check PASS；E2E 默认
#   多轮在 TTK out 复位修复后 PASS。GPU/远端 XPU 未配置，设备三方未验证。
# TensorFlow: 不注册(stock TF 无 ApplyAdamW 单算子 raw op)。


def _provider_binding_selftest():
    """Real TTK bind_params smoke for the kernel/aclnn/e2e provider entries."""
    try:
        from ttk.core_modules.common.xpu import bind_params
    except ImportError:
        print("provider binding: SKIP (ttk not importable)")
        return
    v = object()
    kernel = {
        n: v
        for n in (
            "var",
            "m",
            "v",
            "beta1_power",
            "beta2_power",
            "lr",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "grad",
            "max_grad_norm",
            "amsgrad",
            "maximize",
        )
    }
    aclnn = {
        n: v
        for n in (
            "varRef",
            "mRef",
            "vRef",
            "beta1Power",
            "beta2Power",
            "lr",
            "weightDecay",
            "beta1",
            "beta2",
            "eps",
            "grad",
            "maxGradNormOptional",
            "amsgrad",
            "maximize",
        )
    }
    e2e = {
        n: v
        for n in (
            "beta1_power",
            "beta2_power",
            "lr",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "grad",
            "max_grad_norm",
            "amsgrad",
            "maximize",
            "out",
        )
    }
    for label, cls, pool in (
        ("kernel/torch", _ApplyAdamWTorchCompose, kernel),
        ("aclnn/torch", _ApplyAdamWAclnnTorchCompose, aclnn),
        ("e2e/torch", _ApplyAdamWE2ETorchCompose, e2e),
    ):
        bind_params(cls().__call__, pool, device=None)
        print(f"provider binding OK: {label}")


if __name__ == "__main__":
    import numpy as np

    _provider_binding_selftest()

    # Scalar-contract regression (review): ApplyAdamWTiling::CheckIsScalar accepts
    # any shape holding one element, so (), (1,) and (1, 1) coefficients must all
    # produce var-shaped states on the golden and on every third-party leg — a
    # (1, 1) coefficient broadcasting var (2,) into (1, 2) is a shape regression a
    # flattened comparison would miss.
    _cs = (0.9, 0.999, 0.01, 0.0, 0.9, 0.999, 1e-8)
    _sv = np.array([1.0, 2.0], np.float32)
    _sm = np.array([0.1, 0.2], np.float32)
    _svv = np.array([0.3, 0.4], np.float32)
    _sg = np.array([0.5, 0.6], np.float32)
    _ref = apply_adam_w_golden(_sv, _sm, _svv, *[np.float32(_c) for _c in _cs], _sg)
    for _shape in ((), (1,), (1, 1)):
        _co = [np.full(_shape, _c, np.float32) for _c in _cs]
        _g = apply_adam_w_golden(_sv, _sm, _svv, *_co, _sg)
        assert [_o.shape for _o in _g] == [_sv.shape] * 3, (
            f"golden shapes {[_o.shape for _o in _g]} for coefficient {_shape}"
        )
        assert all(np.array_equal(_a, _b) for _a, _b in zip(_g, _ref)), (
            f"golden value mismatch for coefficient {_shape}"
        )
        _legs = (
            ("kernel/torch", _ApplyAdamWTorchCompose()(_sv, _sm, _svv, *_co, _sg)),
            ("aclnn/torch", _ApplyAdamWAclnnTorchCompose()(_sv, _sm, _svv, *_co, _sg)),
            (
                "e2e/torch",
                _ApplyAdamWE2ETorchCompose()(*_co, _sg, out=(_sv, _sm, _svv)),
            ),
        )
        for _name, _outs in _legs:
            assert [tuple(_o.shape) for _o in _outs] == [_sv.shape] * 3, (
                f"{_name}: coefficient {_shape} broadcast var into "
                f"{[tuple(_o.shape) for _o in _outs]}"
            )
            assert all(
                np.allclose(_o.numpy(), _b, rtol=1e-6, atol=1e-7)
                for _o, _b in zip(_outs, _ref)
            ), f"{_name}: value mismatch for coefficient {_shape}"

    # a coefficient that is not single-element is rejected, never broadcast
    try:
        apply_adam_w_golden(
            _sv, _sm, _svv, *[np.full((2,), _c, np.float32) for _c in _cs], _sg
        )
        raise AssertionError("multi-element coefficient must be rejected")
    except ValueError:
        pass

    # 0-d state keeps rank 0 even when the coefficients arrive as (1, 1)
    _z = np.array(1.0, np.float32)
    _c11 = [np.full((1, 1), _c, np.float32) for _c in _cs]
    assert [_o.shape for _o in apply_adam_w_golden(_z, _z, _z, *_c11, _z)] == [
        ()
    ] * 3, "0-d state promoted by a (1, 1) coefficient"

    print("apply_adam_w golden self-smoke OK")
