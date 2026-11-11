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
"""TestSpec definitions for AdamApplyOne: Kernel / GEIR path.

Update rule (per @adam_apply_one_def.cpp and adam_apply_one_proto.h; the
previous golden's operation order is preserved verbatim, only the executing
framework changed):
    output0 = square(input0) * mul3_x + input1 * mul2_x
    output1 = input2 * mul0_x + input0 * mul1_x
    tmp     = output1 / (sqrt(output0) + add2_y)
    output2 = input3 - tmp * input4

This is one fused Adam step. The positional operand names carry Adam meaning
(from adam_apply_one_proto.h):
    input0 = grad          input1 = v (second moment)
    input2 = m (first moment)
    input3 = var (params to update)
    input4 = lr
    mul0_x = beta1         mul1_x = 1 - beta1
    mul2_x = beta2         mul3_x = 1 - beta2
    add2_y = epsilon
    output0 = v_new        output1 = m_new        output2 = var_new

Note the proto header's inline formula for output2 parenthesizes as
``output1 / sqrt(output0) + add2_y``; README.md and the kernel DAG both agree on
``output1 / (sqrt(output0) + add2_y)``, which is the epsilon-in-denominator Adam
form implemented here. The proto comment is a typo, not a second semantic.

Outputs: exactly three — output0 / output1 / output2 — as declared by
@adam_apply_one_def.cpp (Output("output0"), Output("output1"),
Output("output2")).

Registered names:
  - "adam_apply_one" / "AdamApplyOne"  <- CSV op_name (Kernel / GEIR path,
                                         numpy carrier; snake_case is the CSV
                                         dialect, CamelCase the GE/REG_OP type
                                         name from OP_ADD(AdamApplyOne)). GEIR
                                         reuses the kernel spec.

Benchmark: the spec golden() funnels into the module-level
``_benchmark_adam_apply_one`` — one shared torch basic-op composition owned by
no spec. Independence: torch-only, never calls the operator under test (no
aclnn, no npu dispatch, no device kernel, no reuse of DUT outputs). NumPy
appears only as an input/output carrier bridge (bf16 via int16 bit view) and
never carries the reference arithmetic.

Precision: the compute dtype is chosen per input dtype, matching the kernel's
own ``if constexpr`` branch one-for-one (adam_apply_one.cpp):

    carrier   kernel compute type U          golden compute dtype
    bfloat16  float  (…Compute<bfloat16_t, float>)   float32
    float16   half   (…Compute<DTYPE_INPUT0, DTYPE_INPUT0>)   float16
    float32   float  (…Compute<DTYPE_INPUT0, DTYPE_INPUT0>)   float32

So bfloat16 is up-cast to FP32 and cast back, while **float16 computes natively
in float16** — the kernel takes U=float only for bfloat16, so an FP32 golden on
the FP16 leg would be a different oracle than the device. In particular
``square(input0)`` overflows FP16 to ``inf`` for |input0| > 256, and the golden
must reproduce that ``inf`` rather than a finite FP32 value; an FP32 golden would
report a nan/inf mismatch that no tolerance can absorb. See ``_compute_dtype``.

This differs from the with_decay / with_decay_assign siblings, whose goldens do
up-cast FP16 — because *their kernels* pin U=float unconditionally. The
divergence lives in the kernel sources, not in these goldens; see the operator-
side question recorded in the delivery note at the end of this file.

Within the FP32 compute path one op is widened further: sqrt is evaluated in
FP64 and narrowed back (see ``_sqrt_correctly_rounded``). ``torch.sqrt`` on FP32
CPU tensors is 1 ULP off the correctly-rounded result on ~17% of inputs, and
output2 cancels, which amplifies that wobble by ~4 orders of magnitude in
relative terms. The FP16 path is deliberately left out of that widening: lifting
FP16 intermediates to FP64 would wash out exactly the overflow-to-``inf``
behaviour the FP16 leg is supposed to mirror. With this split, the golden is
bit-exact against the previous numpy reference on all three dtypes.

Scalars keep their declared dtype and broadcast naturally, so the ()- and
(1,)-shaped scalars of the st CSV are not index-extracted.

Import rule: torch / ml_dtypes are imported lazily inside function bodies only,
so plugin discovery stays cheap.
"""

__spec__ = {
    "adam_apply_one": "AdamApplyOneKernelSpec",
    "AdamApplyOne": "AdamApplyOneKernelSpec",
}

# Keep the legacy TTK loader working: newer runners resolve ``__spec__``, older
# probes only inspect ``__golden__``. Only the kernel entry is mirrored here —
# this operator has no aclnn / E2E leg to map (see the delivery note below).
__golden__ = {"kernel": {"adam_apply_one": "adam_apply_one_golden"}}

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


def _compute_dtype(carrier_dtype):
    """Pick the compute dtype for a carrier dtype, mirroring the kernel.

    adam_apply_one.cpp dispatches on the input dtype::

        if constexpr (std::is_same<DTYPE_INPUT0, bfloat16_t>::value)
            AdamApplyOneCompute<bfloat16_t, float>      // U = float
        else
            AdamApplyOneCompute<DTYPE_INPUT0, DTYPE_INPUT0>  // U = T

    so only bfloat16 is promoted; float16 computes in float16 and float32 in
    float32. The golden follows the same branch, which keeps FP16 overflow
    (``square(input0)`` -> ``inf`` for |input0| > 256) faithful to the device.

    Note this is NOT what the with_decay / with_decay_assign siblings do — their
    kernels pin U=float for every dtype, so their goldens up-cast FP16 too.
    """
    import torch

    return torch.float32 if carrier_dtype is torch.bfloat16 else carrier_dtype


def _sqrt_correctly_rounded(x):
    """sqrt mirroring the device's rounding, per compute dtype.

    FP32: ``torch.sqrt`` on a float32 CPU tensor goes through a vectorized path
    that is 1 ULP off the correctly-rounded result on roughly 17% of inputs
    (measured against a float64 oracle on torch 2.7.1+cpu); ``numpy.sqrt`` and
    the device's ``Reg::Sqrt`` are both correctly rounded. A 1-ULP wobble here is
    not cosmetic: output2 = var - tmp * lr cancels, so the relative error on
    small output2 values gets amplified by ~4 orders of magnitude. Widening to
    FP64 for this one op makes the golden exactly agree with the previous numpy
    reference and with the device's own rounding, at no meaningful cost.

    FP16: left on the native ``torch.sqrt`` path — measured bit-exact against
    ``numpy.sqrt`` on float16. It is deliberately NOT widened: lifting FP16
    intermediates into FP64 would turn the overflowed ``inf`` of the FP16 leg
    back into a finite value and stop mirroring the kernel.
    """
    import torch

    if x.dtype is torch.float32:
        return torch.sqrt(x.to(torch.float64)).to(torch.float32)
    return torch.sqrt(x)


def _benchmark_adam_apply_one(
    input0,
    input1,
    input2,
    input3,
    input4,
    mul0_x,
    mul1_x,
    mul2_x,
    mul3_x,
    add2_y,
):
    """Shared benchmark fused-Adam step — module-level, owned by no spec.

    Pure torch basic-op composition. The compute dtype follows the kernel's
    ``if constexpr`` branch via ``_compute_dtype``: bfloat16 carriers compute in
    FP32, float16 computes natively in FP16, float32 in FP32. Returns the three
    torch tensors (output0, output1, output2) in op-def output order, in that
    compute dtype; carrier/dtype restoration is the caller's job. Never touches
    the operator under test.
    """
    cdt = _compute_dtype(_as_torch(input0).dtype)

    def cast(x):
        return _as_torch(x).to(cdt)

    g, v, m, var, lr = (
        cast(input0),
        cast(input1),
        cast(input2),
        cast(input3),
        cast(input4),
    )
    beta1, one_m_beta1, beta2, one_m_beta2, eps = (
        cast(mul0_x),
        cast(mul1_x),
        cast(mul2_x),
        cast(mul3_x),
        cast(add2_y),
    )

    # Operation order mirrors the previous numpy golden and the kernel DAG
    # (CalcOutput0Op / CalcOutput1Op / CalcOutput2Op) term for term. Re-
    # associating these is algebraically identical but shifts FP32 rounding, so
    # it is avoided.
    output0 = g * g * one_m_beta2 + v * beta2
    output1 = m * beta1 + g * one_m_beta1

    tmp = output1 / (_sqrt_correctly_rounded(output0) + eps)
    output2 = var - tmp * lr
    return output0, output1, output2


def adam_apply_one_golden(
    input0,
    input1,
    input2,
    input3,
    input4,
    mul0_x,
    mul1_x,
    mul2_x,
    mul3_x,
    add2_y,
    **kwargs,
):
    """Kernel / GEIR golden (numpy carrier in, numpy carrier out).

    Parameter names and order follow @adam_apply_one_def.cpp inputs; this
    operator declares no attributes. Returns [output0, output1, output2] —
    three tensors, matching the op def.

    Each output takes the dtype of the state it updates (output0<-input1,
    output1<-input2, output2<-input3); the op def constrains all inputs and
    outputs to one dtype, so this only makes the correspondence explicit.

    The benchmark already computes in the kernel's own compute dtype
    (``_compute_dtype``), so for float16 and float32 carriers this cast back is a
    no-op and only bfloat16 is actually narrowed from FP32.
    """
    import numpy as np

    outs = _benchmark_adam_apply_one(
        input0,
        input1,
        input2,
        input3,
        input4,
        mul0_x,
        mul1_x,
        mul2_x,
        mul3_x,
        add2_y,
    )
    return [
        _torch_to_np(res, np.asarray(src).dtype)
        for res, src in zip(outs, (input1, input2, input3))
    ]


class _AdamApplyOneTorchCompose:
    """torch third-party provider (spec-mode class form).

    Runs on the remote XPU endpoint, binding by parameter name against the
    operator's own input names (TTK's remote executor binds a third_party class
    through ``__init__``/``__call__`` signature inspection, so the operand names
    are spelled out rather than swallowed by ``*args``). The GPU leg stays on
    its device — the benchmark keeps torch tensors where they are, so nothing is
    detoured through CPU or NumPy — and the operator under test is never called.

    Shares the one benchmark with golden(), so it inherits the same per-dtype
    compute precision: FP16 inputs are computed natively in FP16 (not promoted),
    matching this kernel's ``if constexpr`` branch.
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
        add2_y,
        **kwargs,
    ):
        outs = _benchmark_adam_apply_one(
            input0,
            input1,
            input2,
            input3,
            input4,
            mul0_x,
            mul1_x,
            mul2_x,
            mul3_x,
            add2_y,
        )
        srcs = [_as_torch(x) for x in (input1, input2, input3)]
        return [res.to(src.dtype) for res, src in zip(outs, srcs)]


class AdamApplyOneKernelSpec:
    """Kernel / GEIR spec registered by CSV ``op_name`` (numpy carrier)."""

    golden = staticmethod(adam_apply_one_golden)
    third_party = {"torch": _AdamApplyOneTorchCompose}
    tolerance = _TOLERANCE


if __name__ == "__main__":
    import numpy as np

    # Rank regression (review): a 0-d carrier must stay 0-d on every leg.
    # np.ascontiguousarray promotes shape () to (1,), which would silently turn
    # a scalar state into a one-element vector in the numpy->torch bridge.
    _VALS = (0.5, 1.0, 2.0, 3.0, 0.1, 0.9, 0.1, 0.999, 0.001, 1e-8)
    for _dt in ("float32", "float16"):
        _s0 = [np.array(_x, _dt) for _x in _VALS]
        _o0 = adam_apply_one_golden(*_s0)
        assert [_o.shape for _o in _o0] == [(), (), ()], (
            f"{_dt} 0-d carrier promoted to {[_o.shape for _o in _o0]}"
        )
        assert all(_o.dtype == np.dtype(_dt) for _o in _o0), f"{_dt} 0-d dtype"
        _t0 = _AdamApplyOneTorchCompose()(*_s0)
        assert [tuple(_t.shape) for _t in _t0] == [(), (), ()], (
            f"{_dt} torch leg promoted the 0-d carrier"
        )
        _s1 = [np.array([_x], _dt) for _x in _VALS]
        _o1 = adam_apply_one_golden(*_s1)
        assert [_o.shape for _o in _o1] == [(1,), (1,), (1,)], f"{_dt} (1,) carrier"
        assert all(np.array_equal(_a.reshape(1), _b) for _a, _b in zip(_o0, _o1)), (
            f"{_dt} 0-d and (1,) carriers disagree"
        )

    try:
        import ml_dtypes

        _sb = [np.array(_x, ml_dtypes.bfloat16) for _x in _VALS]
        _ob = adam_apply_one_golden(*_sb)
        assert [_o.shape for _o in _ob] == [(), (), ()], "bf16 0-d carrier promoted"
        assert all(_o.dtype == np.dtype(ml_dtypes.bfloat16) for _o in _ob), (
            "bf16 0-d dtype"
        )
    except ImportError:
        pass

    print("adam_apply_one golden self-smoke OK")


# 通路交付情况
# 已注册: kernel + GEIR(复用同一个 kernel spec)。注册键取并集: CSV 的 op_name
#   "adam_apply_one"(snake_case) + OP_ADD(AdamApplyOne) 的 GE 类型名
#   "AdamApplyOne"(CamelCase), 两个键指向同一个 AdamApplyOneKernelSpec。
# aclnn: 标准安装包不注册，但接口并非不存在 —— ABI 实测 nm -D --defined-only
#   libopapi.so 搜 aclnnAdamApplyOneGetWorkspaceSize 为 0 命中（标准包未部署）；
#   而 experimental/optim/adam_apply_one/op_host/CMakeLists.txt 为 ACLNNTYPE aclnn，
#   且 examples/test_aclnn_adam_apply_one.cpp 调用该接口 —— aclnn 接口由
#   experimental 包编译生成，需先构建并安装实验包才会产生符号。本环境未构建/
#   部署该实验包，故不登记为可执行通路，状态记为待确认（缺项：实验包构建与部署），
#   不能写成“接口不存在”。
# e2e: 不注册 —— 全仓 grep 无独立前端证据(无 torch_npu.npu_adam_apply_one, 无
#   任何 .py 侧调用)。本算子是 bert/lamb 的融合算子, 由图融合 pass 产生, 不是前端
#   直接暴露的 API。
# TensorFlow: 不注册 —— 本算子目录下无 framework/ 插件, stock TF 也没有
#   AdamApplyOne raw op, 三方腿只用 torch。
# GPU 三方实跑: 当前环境 ttk remote 未配置 endpoint, third_party 已声明但三方比对
#   结果记为未验证 —— 不等于不支持, 也不等于已通过。
#
# 精度口径(按 dtype 分支对齐 kernel, 本轮无数学语义变更):
#   golden 的计算 dtype 与 adam_apply_one.cpp 的 `if constexpr` 分支逐一对应:
#     bfloat16 -> AdamApplyOneCompute<bfloat16_t, float>          U=float  -> golden FP32
#     float16  -> AdamApplyOneCompute<DTYPE_INPUT0, DTYPE_INPUT0> U=half   -> golden FP16
#     float32  -> AdamApplyOneCompute<DTYPE_INPUT0, DTYPE_INPUT0> U=float  -> golden FP32
#   即只有 bf16 升 FP32, **fp16 原生在 FP16 里算**, 与旧 numpy golden 的口径一致
#   (旧 golden 也只给 bfloat16 升精度), 本轮只把参考实现从 NumPy 换成 torch 基础算子
#   组合, 没有改低精度提升行为。torch 的 FP16 CPU 算子(mul/add/sub/div/sqrt/square)
#   实测与 numpy FP16 逐 bit 相同, 含溢出到 inf 的行为, 所以 FP16 腿换成 torch 后仍
#   与原版 bit-exact。
#   注意 _sqrt_correctly_rounded 只对 FP32 走 FP64 再回落(torch FP32 sqrt 比
#   正确舍入差 1 ULP, 而 output2 存在相减抵消会放大该误差); FP16 分支**故意不抬精度**
#   —— 把 FP16 中间量抬到 FP64 会把本该溢出的 inf 洗成有限值, 反而不忠于 kernel。
#
# 算子侧疑点(如实记录, 本轮未改 kernel 代码, 也未验证应如何改):
#   本算子 kernel 的 FP16 腿不升精度(U=half), 而同族的 adam_apply_one_with_decay.cpp
#   与 adam_apply_one_with_decay_assign.cpp 都是无条件 U=float(FP16 也升 FP32)。
#   三者数学语义同源(前两行完全相同), 这个不一致看起来不像有意设计:
#   后果是 FP16 下 square(input0) 在 |input0|>256 时就溢出到 inf(FP16 上限 65504),
#   而同族算子在 FP32 中间量里不会。本算子 st CSV(84 行, 其中 33 行 fp16)里有 4 个
#   用例的 input0 取值域会触发该溢出:
#     AdamApplyOne_float16_ND_fuzz_005126            input0 ∈ (-65504, 65504), 平方 ~4.29e9
#     adam_apply_one_whitebox_fp16_broadcast_00001   input0 ∈ (100, 1000)
#     adam_apply_one_fuzz_ID0000_00019               input0 ∈ (-1000, -100)
#     adam_apply_one_whitebox_fp16_datarange_00006   input0 ∈ (100, 1000)
#   这些用例上 golden 与 kernel 现在同口径(都得 inf), 所以比对不会因此失败 ——
#   但"FP16 中间量溢出"本身是否是期望行为, 需要算子负责人确认: 若期望与同族一致,
#   应改的是 kernel 的 U 选型(并同步回退本文件的 FP16 分支), 而不是单独调 golden。
