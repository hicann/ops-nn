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
"""TestSpec definitions for AdamApplyOneWithDecay: Kernel / GEIR path.

Update rule (per @adam_apply_one_with_decay_def.cpp and
adam_apply_one_with_decay_proto.h; the previous golden's operation order is
preserved verbatim, only the executing framework changed):
    output0 = square(input0) * mul3_x + input1 * mul2_x
    output1 = input2 * mul0_x + input0 * mul1_x
    tmp     = output1 / (sqrt(output0) + add2_y)
    output2 = input3 - (tmp + input3 * mul4_x) * input4

This is one fused Adam step with decoupled weight decay. The positional operand
names carry Adam meaning (from adam_apply_one_with_decay_proto.h):
    input0 = grad          input1 = v (second moment)
    input2 = m (first moment)
    input3 = var (params to update)
    input4 = lr
    mul0_x = beta1         mul1_x = 1 - beta1
    mul2_x = beta2         mul3_x = 1 - beta2
    mul4_x = weight_decay
    add2_y = epsilon
    output0 = v_new        output1 = m_new        output2 = var_new

The decay term sits *inside* the lr-scaled bracket — var is reduced by
``lr * (m_hat/denom + weight_decay * var)``, not by a separate
``var * (1 - lr*wd)`` factor. Do not swap in the ApplyAdamW form.

Note the proto header's inline formula for output2 parenthesizes as
``output1 / sqrt(output0) + add2_y + input3 * mul4_x``; README.md and the kernel
DAG both agree on ``(output1 / (sqrt(output0) + add2_y) + input3 * mul4_x)``,
the epsilon-in-denominator Adam form implemented here. The proto comment is a
typo, not a second semantic.

Outputs: exactly three — output0 / output1 / output2 — as declared by
@adam_apply_one_with_decay_def.cpp (Output("output0"), Output("output1"),
Output("output2")).

Registered names:
  - "adam_apply_one_with_decay" / "AdamApplyOneWithDecay"
        <- CSV op_name (Kernel / GEIR path, numpy carrier; snake_case is the CSV
           dialect, CamelCase the GE/REG_OP type name from
           OP_ADD(AdamApplyOneWithDecay)). GEIR reuses the kernel spec.

Benchmark: the spec golden() funnels into the module-level
``_benchmark_adam_apply_one_with_decay`` — one shared torch basic-op
composition owned by no spec. Independence: torch-only, never calls the operator
under test (no aclnn, no npu dispatch, no device kernel, no reuse of DUT
outputs). NumPy appears only as an input/output carrier bridge (bf16 via int16
bit view) and never carries the reference arithmetic.

Precision: BF16/FP16 carriers are up-cast to FP32 once, the whole update runs in
FP32, then each output is cast back to its input dtype. This mirrors the kernel,
which pins U=float for every dtype
(adam_apply_one_with_decay.cpp: ``AdamApplyOneWithDecayCompute<DTYPE_INPUT0,
float>``), so golden and device share one compute precision. The previous
golden already up-cast both bf16 and fp16; that behaviour is kept.

One op inside that FP32 body is widened further: sqrt is evaluated in FP64 and
narrowed back (see ``_sqrt_correctly_rounded``). ``torch.sqrt`` on FP32 CPU
tensors is 1 ULP off the correctly-rounded result on ~17% of inputs, and output2
cancels, which amplifies that wobble by ~4 orders of magnitude in relative
terms. With the widening, this golden is bit-exact against the previous numpy
reference on all three dtypes.

Scalars keep their declared dtype and broadcast naturally, so the ()-shaped
scalars of the st CSV are not index-extracted.

Import rule: torch / ml_dtypes are imported lazily inside function bodies only,
so plugin discovery stays cheap.
"""

import numpy as np


__spec__ = {
    "adam_apply_one_with_decay": "AdamApplyOneWithDecayKernelSpec",
    "AdamApplyOneWithDecay": "AdamApplyOneWithDecayKernelSpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Only the kernel entry is mirrored here —
# this operator has no aclnn / E2E leg to map (see the delivery note below).
__golden__ = {
    "kernel": {"adam_apply_one_with_decay": "adam_apply_one_with_decay_golden"}
}

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

    # np.ascontiguousarray would promote shape () to (1,), turning a scalar
    # carrier into a one-element vector; a 0-d array is contiguous already.
    arr = np.asarray(x)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy(order="C")
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
    import numpy as np
    import torch

    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return _np_to_torch(np.asarray(x))


def _sqrt_correctly_rounded(x):
    """FP32 sqrt, correctly rounded — evaluated in FP64 then narrowed.

    ``torch.sqrt`` on a float32 CPU tensor goes through a vectorized path that
    is 1 ULP off the correctly-rounded result on roughly 17% of inputs (measured
    against a float64 oracle on torch 2.7.1+cpu); ``numpy.sqrt`` and the device's
    ``Reg::Sqrt`` are both correctly rounded. A 1-ULP wobble here is not
    cosmetic: output2 cancels against input3, so the relative error on small
    output2 values gets amplified by ~4 orders of magnitude.

    Widening to FP64 for this one op makes the golden exactly agree with the
    previous numpy reference and with the device's own rounding, at no
    meaningful cost. Everything else stays in FP32, so the kernel's compute
    precision is still what is being mirrored.
    """
    import torch

    if x.dtype is torch.float32:
        return torch.sqrt(x.to(torch.float64)).to(torch.float32)
    return torch.sqrt(x)


def _benchmark_adam_apply_one_with_decay(
    input0,
    input1,
    input2,
    input3,
    input4,
    mul0_x,
    mul1_x,
    mul2_x,
    mul3_x,
    mul4_x,
    add2_y,
):
    """Shared benchmark fused-Adam-with-decay step — module-level, owned by no
    spec.

    Pure torch basic-op composition evaluated in FP32. Returns the three torch
    FP32 tensors (output0, output1, output2) in op-def output order;
    carrier/dtype restoration is the caller's job. Never touches the operator
    under test.
    """
    import torch

    def f32(x):
        # Promote-aware: keep a promoted float64 reference exact; low precision
        # and float32 carriers still compute in float32 (mirrors the kernel).
        t = _as_torch(x)
        return t if t.dtype == torch.float64 else t.to(torch.float32)

    g, v, m, var, lr = (f32(input0), f32(input1), f32(input2), f32(input3), f32(input4))
    beta1, one_m_beta1, beta2, one_m_beta2 = (
        f32(mul0_x),
        f32(mul1_x),
        f32(mul2_x),
        f32(mul3_x),
    )
    weight_decay, eps = f32(mul4_x), f32(add2_y)

    # Operation order mirrors the previous numpy golden and the kernel DAG
    # (…CalcOutput0 / …CalcOutput1 / …CalcOutput2) term for term. Re-associating
    # these is algebraically identical but shifts FP32 rounding, so it is
    # avoided.
    output0 = g * g * one_m_beta2 + v * beta2
    output1 = m * beta1 + g * one_m_beta1

    tmp = output1 / (_sqrt_correctly_rounded(output0) + eps)
    output2 = var - (tmp + var * weight_decay) * lr
    return output0, output1, output2


def adam_apply_one_with_decay_golden(
    input0,
    input1,
    input2,
    input3,
    input4,
    mul0_x,
    mul1_x,
    mul2_x,
    mul3_x,
    mul4_x,
    add2_y,
    **kwargs,
):
    """Kernel / GEIR golden (numpy carrier in, numpy carrier out).

    Parameter names and order follow @adam_apply_one_with_decay_def.cpp inputs;
    this operator declares no attributes. Returns [output0, output1, output2] —
    three tensors, matching the op def.

    Each output takes the dtype of the state it updates (output0<-input1,
    output1<-input2, output2<-input3); the op def constrains all inputs and
    outputs to one dtype, so this only makes the correspondence explicit.
    """
    import numpy as np

    outs = _benchmark_adam_apply_one_with_decay(
        input0,
        input1,
        input2,
        input3,
        input4,
        mul0_x,
        mul1_x,
        mul2_x,
        mul3_x,
        mul4_x,
        add2_y,
    )
    return [
        _torch_to_np(res, np.asarray(src).dtype)
        for res, src in zip(outs, (input1, input2, input3))
    ]


class _AdamApplyOneWithDecayTorchCompose:
    """torch third-party provider (spec-mode class form).

    Runs on the remote XPU endpoint, binding by parameter name against the
    operator's own input names (TTK's remote executor binds a third_party class
    through ``__init__``/``__call__`` signature inspection, so the operand names
    are spelled out rather than swallowed by ``*args``). The GPU leg stays on
    its device — the benchmark keeps torch tensors where they are, so nothing is
    detoured through CPU or NumPy — and the operator under test is never called.
    """

    def __call__(
        self,
        input0,
        input1,
        input2,
        input3,
        input4,
        mul0_x,
        mul1_x,
        mul2_x,
        mul3_x,
        mul4_x,
        add2_y,
        **kwargs,
    ):
        outs = _benchmark_adam_apply_one_with_decay(
            input0,
            input1,
            input2,
            input3,
            input4,
            mul0_x,
            mul1_x,
            mul2_x,
            mul3_x,
            mul4_x,
            add2_y,
        )
        srcs = [_as_torch(x) for x in (input1, input2, input3)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]


class AdamApplyOneWithDecayKernelSpec:
    """Kernel / GEIR spec registered by CSV ``op_name`` (numpy carrier)."""

    golden = staticmethod(adam_apply_one_with_decay_golden)
    third_party = {"torch": _AdamApplyOneWithDecayTorchCompose}
    tolerance = _TOLERANCE


# 通路交付情况
# 已注册: kernel + GEIR(复用同一个 kernel spec)。注册键取并集: CSV 的 op_name
#   "adam_apply_one_with_decay"(snake_case) + OP_ADD(AdamApplyOneWithDecay) 的 GE
#   类型名 "AdamApplyOneWithDecay"(CamelCase), 两个键指向同一个
#   AdamApplyOneWithDecayKernelSpec。
# aclnn: 标准安装包不注册，但接口并非不存在 —— ABI 实测 nm -D --defined-only
#   libopapi.so 搜 aclnnAdamApplyOneWithDecayGetWorkspaceSize 为 0 命中（标准包
#   未部署）；而 experimental/optim/adam_apply_one_with_decay/op_host/CMakeLists.txt
#   为 ACLNNTYPE aclnn，且 examples/test_aclnn_adam_apply_one_with_decay.cpp 调用该
#   接口 —— 接口由 experimental 包编译生成，需先构建并安装实验包。本环境未构建/
#   部署，状态记为待确认（缺项：实验包构建与部署），不能写成“接口不存在”。
# e2e: 不注册 —— 全仓 grep 无独立前端证据(无 torch_npu 侧 API, 无任何 .py 侧调用)。
#   本算子是 bert/lamb 的融合算子, 由图融合 pass 产生, 不是前端直接暴露的 API。
# TensorFlow: 不注册 —— 本算子目录下无 framework/ 插件(同族的 with_decay_assign
#   才有), stock TF 也没有 AdamApplyOneWithDecay raw op, 三方腿只用 torch。
# GPU 三方实跑: 当前环境 ttk remote 未配置 endpoint, third_party 已声明但三方比对
#   结果记为未验证 —— 不等于不支持, 也不等于已通过。
#
# 精度口径: 本算子 kernel 恒为 U=float(adam_apply_one_with_decay.cpp 无 dtype 分支),
#   golden 的"bf16/fp16 升 FP32 再回投"与设备同口径, 旧 golden 的升精度行为原样保留,
#   本轮无数学语义变更(仅把参考实现从 NumPy 换成 torch 基础算子组合)。


if __name__ == "__main__":
    # Promote regression (review): a float64 reference must stay float64.
    # g=0, v=1, beta1=beta2=1, m=1+2^-23, lr=99999992, var=1e8, wd=eps=0
    # gives var - m*lr = -3.9209280014038086. FP32 cancellation would give 0.
    g = np.array([0.0], np.float64)
    v = np.array([1.0], np.float64)
    m = np.array([1.0 + 2.0**-23], np.float64)
    var = np.array([1e8], np.float64)
    lr = np.array([99999992.0], np.float64)
    one = np.array([1.0], np.float64)
    zero = np.array([0.0], np.float64)
    out = adam_apply_one_with_decay_golden(
        g, v, m, var, lr, one, zero, one, zero, zero, zero
    )
    assert all(x.dtype == np.float64 for x in out), "promoted golden lost float64"
    ref = var - m * lr
    assert np.array_equal(out[2], ref), (
        f"promoted golden downcast to FP32: {out[2]} != {ref}"
    )
    # Rank regression (review): a 0-d carrier must stay 0-d on every leg.
    # np.ascontiguousarray promotes shape () to (1,), which would silently turn
    # a scalar state into a one-element vector in the numpy->torch bridge.
    _s0 = [np.array(0.5, np.float32) for _ in range(11)]
    _o0 = adam_apply_one_with_decay_golden(*_s0)
    assert all(_o.shape == () for _o in _o0), (
        f"0-d carrier promoted to {[_o.shape for _o in _o0]}"
    )
    _t0 = _AdamApplyOneWithDecayTorchCompose()(*_s0)
    assert all(tuple(_t.shape) == () for _t in _t0), (
        "torch leg promoted the 0-d carrier"
    )
    _s1 = [np.array([0.5], np.float32) for _ in range(11)]
    _o1 = adam_apply_one_with_decay_golden(*_s1)
    assert all(_o.shape == (1,) for _o in _o1), "(1,) carrier lost its rank"
    assert all(np.array_equal(_a.reshape(1), _b) for _a, _b in zip(_o0, _o1)), (
        "0-d and (1,) carriers disagree"
    )

    print("adam_apply_one_with_decay_golden promote self-smoke OK")
