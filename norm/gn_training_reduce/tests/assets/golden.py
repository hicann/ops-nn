#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK TestSpec golden for gn_training_reduce (GroupNorm training statistical reduction).

Fact sources — spec/spec.yaml ONLY.  Nothing under op/ (kernel artifacts) is
imported, executed or reused: the reference math below is derived from
math_semantics.formula / math_semantics.format_variants, the tolerance from
cross_check + numerical_tolerance, and the competitor from reference_oracle.

Semantics (spec math_semantics.formula; REQUIREMENTS §2.1 / §5.3.1):
    x is rank-4, layout NCHW (canonical) or NHWC, dtype fp16/fp32,
    attr num_groups = G, D = C/G, M = D*H*W.
        NCHW: xg = x[N,C,H,W] -> [N,G,D,H,W], reduce axes (2,3,4)
        NHWC: xg = x[N,H,W,C] -> [N,H,W,G,D], reduce axes (1,2,4)
        sum        = reduce_sum(xg)        keepdims -> fp32, shape (N,G,1,1,1) / (N,1,1,G,1)
        square_sum = reduce_sum(xg * xg)  keepdims -> fp32, shape (N,G,1,1,1) / (N,1,1,G,1)

Backend priority ladder (spec reference_oracle = {framework: torch, absent: true}):
  * tier 1 — single PyTorch API: SKIPPED.  No torch API returns BOTH (sum,
    square_sum) for a per-(n,g) grouped reduction.  torch.nn.functional.group_norm /
    torch.ops.aten.native_group_norm / torch.native_group_norm return the normalized
    output (plus Welford mean/rstd), never the raw Σx / Σx²; torch.var_mean returns the
    Welford mean/var, not the raw first/second moments; torch.sum yields only one of
    the two outputs.  (REQUIREMENTS §4.5 / §4.6)
  * tier 2 — single TensorFlow API: SKIPPED.  TF has no operator producing grouped
    [N,G] Σx / Σx².  tf.nn.moments returns mean/variance; tf.math.reduce_sum yields a
    single reduction; FusedBatchNormV3 reduces per-channel over N·H·W, a different
    grouping.  (REQUIREMENTS §3.2–§3.6)
  * tier 3 — PyTorch API composition: TAKEN.  torch.reshape (regroup C into the grouped
    view) + torch.square (xg²) + torch.sum (reduce the group axes).  Exact APIs used:
    torch.reshape, torch.square, torch.sum.
  * tiers 4 (TensorFlow composition) / 5 (numpy) — not reached: tier 3 holds.

Float64 accumulation rule (workflow-mandated; overrides the spec accumulator_dtype
float32): every input is lifted with .to(torch.float64), the reduction accumulates
entirely in float64, and each output is cast back once to the target output dtype
(float32).  No fp16/fp32 accumulation happens anywhere in the reduction.

Tolerance derivation (spec.yaml, never the operator name):
  * default by op.paradigms [Reduction] (float output) = stat_rel_err;
  * the spec top-level cross_check block {standard: cross_check, level: L1} overrides
    every fp16/fp32 output dtype -> {standard: cross_check, level: L1}.  This op's
    outputs are always fp32 (with an fp16 input path), so those are the keys.

third_party: spec reference_oracle is absent (no single framework API), so the competitor
is declared here as the torch reduction composition — an independent computation form
(ones-vector multiply-accumulate, not torch.sum) reached only through this declaration.
The competitor runs at the operator's fp32 contract precision (mirroring the arch35
kernel), NOT at the golden's fp64 precision: cross_check compares peer error levels, so
an fp64 competitor would be exact and collapse the ratio denominator.

cross_check applicability (both outputs are raw moments): the `rmse` leg of cross_check
is an *absolute* RMS ratio against a floor-clamped denominator, so it is only stable when
the raw sum is far from zero and not of extreme magnitude. Here Σx may cancel to ~0 and
Σx² scales with the input range squared, so extreme-range / cancellation cases make the
ratio blow up even when the NPU is strictly more accurate than the competitor (its
relative-error ratios are then ~1e-3). Convention (per bn3d_training_reduce/tests/assets/
golden.py): keep the random cross_check regression on a non-zero-mean range, and accept
cancellation / near-zero / extreme-value cases with the formal `--compare close`
(FLOAT32 rtol/atol/ptol) tolerance instead.
"""

__spec__ = {
    "gn_training_reduce": "GNTrainingReduceTestSpec",  # legacy official IR op name
}

import numpy as np
import torch

_NCHW = "NCHW"
_NHWC = "NHWC"


def _flatten_tokens(value):
    """Yield every scalar token inside a (possibly nested) format structure."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tokens(item)
    elif hasattr(value, "tolist"):
        yield from _flatten_tokens(value.tolist())
    else:
        yield value


def _resolve_format(kwargs):
    """Layout of x -- purely format-driven, never guessed.

    TTK always supplies the input format (``input_formats`` is carried inside the
    X-Input-Schema and handed to the golden/compose), and for this rank-4 op an
    ambiguous shape (space dim equal to C) is resolved by the format alone.

    Only NCHW / NHWC are meaningful here, so the first such token wins.  When no
    format is supplied the golden must fail loudly instead of assuming NCHW:
    assuming NCHW for an NHWC input reads the channel axis off the wrong dim,
    which either crashes the competitor outright or -- worse -- lands on a
    different-but-legal grouping and silently reports a bogus comparison.
    """
    for key in (
        "input_formats",
        "input_ori_formats",
        "tensor_formats",
        "tensor_ori_formats",
    ):
        for token in _flatten_tokens(kwargs.get(key)):
            token = str(token).upper()
            if token in (_NCHW, _NHWC):
                return token
    raise ValueError(
        "no input format for layout resolution (need input_formats in kwargs): "
        "format-ish keys present=%s" % sorted(k for k in kwargs if "format" in k)
    )


def _grouped_view(shape, num_groups, fmt):
    """(grouped_shape, reduction_axes) per spec math_semantics.format_variants."""
    g = int(num_groups)
    if fmt == _NHWC:
        n, h, w, c = (int(s) for s in shape)
        d = c // g
        return (n, h, w, g, d), (1, 2, 4)
    n, c, h, w = (int(s) for s in shape)
    d = c // g
    return (n, g, d, h, w), (2, 3, 4)


def _torch_group_reduce(x, num_groups, fmt):
    """tier-3 golden body: torch.reshape + torch.square + torch.sum, all in float64."""
    view, axes = _grouped_view(tuple(x.shape), num_groups, fmt)
    xg = torch.reshape(x.to(torch.float64), view)
    sum_out = torch.sum(xg, dim=axes, keepdim=True).to(torch.float32)
    square_sum_out = torch.sum(torch.square(xg), dim=axes, keepdim=True).to(
        torch.float32
    )
    return sum_out, square_sum_out


def _torch_oracle_group_reduce(x, num_groups, fmt):
    """Independent competitor (third_party): fp32 ones-vector multiply-accumulate.

    The competitor mirrors the arch35 kernel's float32 arithmetic, which is the
    operator contract (``spec.yaml accumulator_dtype: float32``): fp16/bf16 lanes
    are promoted to fp32, the reduction accumulates in fp32, and both outputs are
    stored as fp32.  An fp64 competitor would be effectively exact, collapsing the
    cross_check ratio denominator onto the ``small_value`` floor and turning a
    peer-to-peer comparison into an absolute-error-vs-floor test.

    Same value semantics as the formula but a different computation form from the
    golden's torch.sum: the group axis is flattened and reduced with torch.matmul
    against a ones vector, so the two references do not share a summation code path.
    """
    g = int(num_groups)
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    if fmt == _NHWC:
        n, h, w, c = (int(s) for s in x.shape)
        d = c // g
        xg = torch.reshape(x, (n, h, w, g, d))
        xg = torch.reshape(torch.permute(xg, (0, 3, 1, 2, 4)), (n, g, h * w * d))
        out_shape = (n, g, 1, 1, 1)
        permute_to_nhwc = True
    else:
        n, c, h, w = (int(s) for s in x.shape)
        d = c // g
        xg = torch.reshape(x, (n, g, d * h * w))
        out_shape = (n, g, 1, 1, 1)
        permute_to_nhwc = False

    ones = torch.ones((xg.shape[2], 1), dtype=xg.dtype, device=xg.device)
    sum_out = torch.reshape(torch.matmul(xg, ones), out_shape)
    square_sum_out = torch.reshape(torch.matmul(torch.square(xg), ones), out_shape)
    if permute_to_nhwc:  # (N,G,1,1,1) -> (N,1,1,G,1)
        sum_out = torch.permute(sum_out, (0, 2, 3, 1, 4))
        square_sum_out = torch.permute(square_sum_out, (0, 2, 3, 1, 4))
    return sum_out.to(torch.float32), square_sum_out.to(torch.float32)


# The competitor is a small-operator composition (reshape/permute/matmul); for a
# fair XPU performance baseline it must be compiled (kernel fusion) rather than
# paid as eager launch overhead.  Fail-open: if torch.compile is unavailable the
# eager implementation is kept, so golden/cross_check semantics are unaffected.
try:
    _torch_oracle_group_reduce = torch.compile(
        _torch_oracle_group_reduce, dynamic=False
    )
except Exception:  # pragma: no cover - torch.compile optional
    pass


def _group_reduce(x, num_groups, fmt):
    """Dispatch on the input container: torch (ACLNN) vs numpy (Kernel/GEIR)."""
    if isinstance(x, torch.Tensor):
        return list(_torch_group_reduce(x, num_groups, fmt))
    x_t = torch.from_numpy(np.asarray(x))
    return [out.numpy() for out in _torch_group_reduce(x_t, num_groups, fmt)]


class GNTrainingReduceTorchOracle:
    """reference_oracle competitor, bound by name to the op's x / num_groups.

    The xpu-server binds parameters by name, so ``num_groups`` must stay verbatim
    (spec attributes[].name); a renamed parameter would silently fall back to the
    default and the competitor would run the wrong grouping.
    """

    def __init__(self, num_groups=2, **kwargs):
        self.num_groups = int(num_groups)

    def __call__(self, x, **kwargs):
        fmt = _resolve_format(
            kwargs
        )  # no layout attr exists; defaults to canonical NCHW
        return list(_torch_oracle_group_reduce(x, self.num_groups, fmt))


class GNTrainingReduceTestSpec:
    """gn_training_reduce TestSpec: golden + third_party + tolerance.

    The same class serves both flows: Kernel/GEIR pass numpy.ndarray (converted to
    torch inside and back), ACLNN/E2E pass torch.Tensor.  sumOut / squareSumOut are
    the ACLNN output slots (positional in the C header); the reference math returns
    freshly computed tensors regardless.
    """

    def golden(x, num_groups=2, sumOut=None, squareSumOut=None, **kwargs):
        fmt = _resolve_format(kwargs)
        return _group_reduce(x, num_groups, fmt)

    third_party = {"torch": GNTrainingReduceTorchOracle}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
