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
"""TestSpec golden for gn_training_update (TTK ``--plugin``).

Semantics source: spec/spec.yaml ``math_semantics.formula`` ONLY.
This file never imports or reuses any kernel artifacts (the op/ tree).

Backend priority-ladder trace (dispatch-mandated search):
- Tier 1 (single PyTorch API) — SKIPPED. The spec's reference_oracle is
  torch.native_group_norm, but it is NOT equivalent to math_semantics.formula:
  native_group_norm re-reduces statistics from x itself and has no parameters
  to consume the op's ``sum`` / ``square_sum`` inputs, whereas the formula
  defines batch_mean = sum/M and batch_variance = square_sum/M - mean^2
  (statistics arrive as inputs produced upstream by GNTrainingReduce; e.g. a
  NaN in x must NOT flow back into batch_mean/batch_variance). The search
  lacked a torch API that consumes pre-computed group sum/square_sum.
- Tier 2 (single TensorFlow API) — SKIPPED. No single TF API consumes
  pre-computed per-group sum/square_sum tensors either (tf.nn.moments /
  fused batch-norm variants all reduce from the activation itself).
- Tier 3 (PyTorch API composition) — SKIPPED. With the statistics already
  given as inputs, no part of the formula is covered by any higher-level
  torch API; only elementary tensor ops (reshape/div/sqrt/sub/mul/add)
  would remain, and re-implementing an API's internals with elementary
  tensor ops is a tier-5 hand implementation, not a composition.
- Tier 4 (TensorFlow API composition) — SKIPPED for the same reason: no TF
  API covers any part of the formula once sum/square_sum are inputs.
- Tier 5 (NumPy hand implementation) — TAKEN. The formula is implemented
  directly with numpy primitives: reshape to the 5D group view, divide by
  M = (C/G)*H*W, sqrt(var + eps), broadcast normalize, optional affine.

Float64 accumulation rule: the golden performs NO reduction/accumulation
itself (the group sums are inputs produced upstream, not recomputed here).
Regardless, the ENTIRE main chain is lifted to float64
(x/sum/square_sum/scale/offset -> numpy.float64 -> formula -> cast back:
y -> x.dtype, batch_mean/batch_variance -> float32), which subsumes the
mandated astype(float64) -> accumulate -> astype(out_dtype) pattern; no
float16/float32 arithmetic appears anywhere in the computation.

Layout: canonical NCHW (x [N,C,H,W], stats [N,G,1,1,1], affine
[1,G,1,1,1]); the NHWC variant (x [N,H,W,C], stats [N,1,1,G,1], affine
[1,1,1,G,1]) is detected from the ``input_formats`` framework metadata
when present, otherwise from the group-axis position in ``sum``'s shape
(the G == 1 ambiguous case is layout-invariant by construction).

Tolerance: spec.yaml declares a top-level cross_check block
(standard: cross_check, level: L1), so both float dtypes use
{standard: cross_check, level: L1}; the reference_oracle competitor API
# NOTE（支持域限制，cross_check 竞品参照）：third_party 的 torch.native_group_norm
# 从 x 自行重约统计量，与 spec 公式（统计量由 sum/square_sum 输入给定）不等价；
# 且其 fp32 归约在 x 含 NaN/Inf 或满幅（±3.4e38）输入域不可用（溢出/NaN）。
# 因此该参照仅适用于常规数值域，NaN/Inf/满幅输入类用例不参与三方裁定。
(torch.native_group_norm) is declared as third_party (cross_check requires
third_party). third_party __init__/__call__ parameter names match the op's
attribute/input names verbatim (num_groups, epsilon) so the xpu-server
binds by name.
"""

import numpy as np

__spec__ = {"gn_training_update": "GnTrainingUpdateTestSpec"}


def _is_nchw(x_shape, sum_shape, num_groups, kwargs):
    """Detect canonical NCHW vs NHWC variant.

    Prefer framework metadata (input_formats[0] is x's format); fall back to
    the group-axis position in the 5D statistics shape: NCHW [N,G,1,1,1]
    vs NHWC [N,1,1,G,1]. Ambiguity is only possible when G == 1, where both
    layouts are semantically identical (a single group spans all channels).
    """
    formats = kwargs.get("input_formats") or []
    if formats:
        fmt0 = str(formats[0]).upper()
        if "NHWC" in fmt0:
            return False
        if "NCHW" in fmt0:
            return True
    g_axis1 = len(sum_shape) >= 2 and sum_shape[1] == num_groups
    g_axis3 = len(sum_shape) >= 4 and sum_shape[3] == num_groups
    if g_axis3 and not g_axis1:
        return False
    return True  # default: canonical NCHW (also the G == 1 ambiguous case)


def _golden_impl(x, sum_arr, square_sum, scale, offset, num_groups, epsilon, kwargs):
    """Full-float64 evaluation of spec math_semantics.formula.

    Returns (y, batch_mean, batch_variance) as numpy arrays; y keeps x.dtype,
    batch_mean/batch_variance are float32, both with sum's shape.
    """
    g = int(num_groups)
    nchw = _is_nchw(x.shape, sum_arr.shape, g, kwargs)
    # float64 main chain (subsumes the float64 accumulation rule)
    x64 = np.asarray(x, dtype=np.float64)
    sum64 = np.asarray(sum_arr, dtype=np.float64)
    sq64 = np.asarray(square_sum, dtype=np.float64)

    if nchw:
        n, c, h, w = x.shape
        m = (c // g) * h * w
        x5 = x64.reshape(n, g, c // g, h, w)
        y_shape = (n, c, h, w)
    else:
        n, h, w, c = x.shape
        m = (c // g) * h * w
        x5 = x64.reshape(n, h, w, g, c // g)
        y_shape = (n, h, w, c)

    # mean = sum/M ; var = square_sum/M - mean^2 (biased, eps NOT included)
    batch_mean = sum64 / m
    batch_variance = sq64 / m - batch_mean * batch_mean
    multiplier = np.sqrt(batch_variance + float(epsilon))
    # stats [N,G,1,1,1] / [N,1,1,G,1] broadcast along their size-1 axes
    y5 = (x5 - batch_mean) / multiplier
    if scale is not None:
        scale64 = np.asarray(scale, dtype=np.float64)
        offset64 = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.float64(0.0)
        )
        y5 = y5 * scale64 + offset64

    y = y5.reshape(y_shape).astype(x.dtype)
    return (
        y,
        batch_mean.astype(np.float32),
        batch_variance.astype(np.float32),
    )


class GnTrainingUpdateTestSpec:
    """gn_training_update — kernel-flow TestSpec (numpy in, numpy out).

    Inputs (positional, op-def order): x, sum, square_sum, then optional
    scale, offset, mean, variance. ``mean``/``variance`` are official-IR
    reserved optional inputs: per spec they never participate in the
    computation (statistics are always recomputed from sum/square_sum), so
    they are accepted and ignored. Attributes: num_groups (int64, default 2),
    epsilon (float32, default 1e-4) — keyword-only after ``*``.
    """

    def golden(
        x,
        sum,
        square_sum,
        scale=None,
        offset=None,
        mean=None,
        variance=None,
        *,
        num_groups=2,
        epsilon=0.0001,
        **kwargs,
    ):
        del mean, variance  # IR-reserved inputs: excluded by spec semantics
        y, batch_mean, batch_variance = _golden_impl(
            x, sum, square_sum, scale, offset, num_groups, epsilon, kwargs
        )
        return [y, batch_mean, batch_variance]

    # -- third_party: reference_oracle competitor API (torch.native_group_norm) --
    # Runs in the framework-isolated third-party process (xpu-server); never
    # imported by the golden itself. native_group_norm re-reduces statistics
    # from x, so it is the *competitor baseline* for cross_check, not the
    # golden. Per spec reference_oracle notes: its 3rd output is rstd, so
    # batch_variance = 1/rstd^2 - eps (biased variance, correction=0); the
    # eps default gap (torch 1e-5 vs IR 1e-4) is aligned by forwarding the
    # op's epsilon attribute verbatim (xpu-server binds attrs by name).
    class TorchNativeGroupNormRef:
        def __init__(self, *, num_groups=2, epsilon=0.0001, **kwargs):
            self.num_groups = int(num_groups)
            self.epsilon = float(epsilon)

        def __call__(
            self,
            x,
            sum,
            square_sum,
            scale=None,
            offset=None,
            mean=None,
            variance=None,
            **kwargs,
        ):
            import torch

            del mean, variance  # IR-reserved; competitor re-reduces from x
            g = self.num_groups
            nchw = _is_nchw(tuple(x.shape), tuple(sum.shape), g, kwargs)
            del sum, square_sum

            x_t = x if nchw else x.permute(0, 3, 1, 2).contiguous()
            n, c = x_t.shape[0], x_t.shape[1]
            hw = x_t.shape[2] * x_t.shape[3]
            x32 = x_t.to(torch.float32)
            weight = bias = None
            if scale is not None:
                # affine params are per-group [.,G,1,1,.]; expand to per-channel
                s = scale.to(torch.float32).reshape(g)
                weight = s.repeat_interleave(c // g)
                if offset is not None:
                    b = offset.to(torch.float32).reshape(g)
                    bias = b.repeat_interleave(c // g)
            y32, mean_t, rstd = torch.native_group_norm(
                x32, weight, bias, n, c, hw, g, self.epsilon
            )
            var_t = 1.0 / (rstd * rstd) - self.epsilon
            y_t = y32.to(x.dtype)
            if not nchw:
                y_t = y_t.permute(0, 2, 3, 1).contiguous()
            stat_shape = (n, g, 1, 1, 1) if nchw else (n, 1, 1, g, 1)
            batch_mean = mean_t.reshape(stat_shape).to(torch.float32)
            batch_var = var_t.reshape(stat_shape).to(torch.float32)
            return [y_t, batch_mean, batch_var]

    third_party = {"torch": TorchNativeGroupNormRef}

    # cross_check override from spec.yaml top-level cross_check block:
    # standard=cross_check, level=L1 for every float dtype (fp16/fp32 here;
    # the op has no bf16 combination). cross_check requires third_party above.
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
