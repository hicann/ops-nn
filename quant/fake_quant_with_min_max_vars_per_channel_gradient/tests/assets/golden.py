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

"""Golden reference for FakeQuantWithMinMaxVarsPerChannelGradient (per-channel QAT backward).

The reference math is expressed with torch tensor-operator composition (not bare NumPy), and the
per-channel nudge is computed in float32 so it matches the Kernel and the TensorFlow reference
implementation exactly (`round(qMin - min/scale)` at the fp32 precision used on device). A
TensorFlow `third_party` competitor is provided for cross-checking.

Contract (README):
  - q_min = 1 if narrow_range else 0, q_max = 2^num_bits - 1,
    scale = (max - min) / (q_max - q_min), nudged zero point = round(q_min - min/scale)
    (half away from zero, as TF std::round) clipped to [q_min, q_max],
    nudged_min = (q_min - zp) * scale, nudged_max = (q_max - zp) * scale.
  - only a both-zero channel (min == 0 and max == 0) passes gradients through with
    bp_min/bp_max = 0 (TF special case); every other max <= min channel keeps the raw
    nudge formula above: min == max != 0 collapses the nudged interval to [0, 0]
    (mask keeps only x == 0) and max < min reverses it (mask empty), so the out-of-range
    gradients still land in bp_min/bp_max.
  - backprops_wrt_x = gradients * mask, mask = (nudged_min <= x <= nudged_max); a NaN element
    yields mask = 0.
  - backprops_wrt_min = sum_{x < nudged_min} gradients, backprops_wrt_max = sum_{x > nudged_max}
    gradients over all leading axes, accumulated in float64 to stay close to the Kahan-compensated
    device accumulation.
"""

import numpy as np

__golden__ = {
    "kernel": {
        "FakeQuantWithMinMaxVarsPerChannelGradient": (
            "fake_quant_with_min_max_vars_per_channel_gradient_golden"
        ),
        # Compatibility alias: some prepared TTK case CSVs bind op_name to the installed
        # opInterface value (snake_case) instead of the GE op_type (CamelCase).
        "fake_quant_with_min_max_vars_per_channel_gradient": (
            "fake_quant_with_min_max_vars_per_channel_gradient_golden"
        ),
    }
}

# Explicit module-level TestSpec registration (newer TTK spec convention) alongside the legacy
# __golden__ mapping above, so both discovery mechanisms resolve this operator.
__spec__ = {
    "FakeQuantWithMinMaxVarsPerChannelGradient": (
        "FakeQuantWithMinMaxVarsPerChannelGradientTestSpec"
    ),
    # snake_case / tf python API keys so the ttk e2e route (which resolves the
    # golden by the CSV api_name) picks this golden up, aligned with the OA
    # master __spec__ registration shape.
    "fake_quant_with_min_max_vars_per_channel_gradient": (
        "FakeQuantWithMinMaxVarsPerChannelGradientTestSpec"
    ),
    "tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient": (
        "FakeQuantWithMinMaxVarsPerChannelGradientTestSpec"
    ),
}


def _as_tensor(value):
    import torch

    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def fake_quant_with_min_max_vars_per_channel_gradient_golden(
    gradients, x, min, max, num_bits=8, narrow_range=False, **kwargs
):
    # ttk e2e calls golden with 6 positional args; attributes absent from the case
    # CSV arrive as the TF python-signature default None (the real op defaults live
    # in the op registration: num_bits=8, narrow_range=False). Accept positional or
    # keyword attrs and normalize None to the op defaults (issue #14).
    if num_bits is None:
        num_bits = 8
    if narrow_range is None:
        narrow_range = False
    import torch

    g = _as_tensor(gradients)
    x = _as_tensor(x)
    mn = _as_tensor(min).reshape(-1)
    mx = _as_tensor(max).reshape(-1)
    if x.dim() == 0:
        x = x.reshape(1)

    q_min = 1 if narrow_range else 0
    q_max = 2 ** int(num_bits) - 1
    channel = mn.shape[-1]
    q_range = float(q_max - q_min)

    # Nudge precision: float32, scale = (max - min) / qRange via a single Sub then a single
    # Div, matching the forward FakeQuantWithMinMaxVarsPerChannel and the TF reference
    # (README「约束说明」). Extreme/special min/max (+/-FLT_MAX, +/-inf) are outside the
    # documented domain; this reference keeps the plain fp32 evaluation for them.
    # No degeneracy short-circuit: TF's Nudge() applies the raw formula even when
    # max <= min (scale <= 0 reverses/collapses the nudged interval and the masks follow);
    # only min == 0 and max == 0 keeps the TF pass-through special case below.
    scale = (mx - mn) / q_range
    zp = q_min - mn / scale
    # Round half away from zero (TF std::round), not torch.round (round-half-to-even):
    # a zp_raw == n + 0.5 channel would otherwise shift the nudged interval by one scale.
    rounded = torch.sign(zp) * torch.floor(torch.abs(zp) + 0.5)
    nudged_zp = torch.clamp(rounded, float(q_min), float(q_max))
    nudged_min = (q_min - nudged_zp) * scale
    nudged_max = (q_max - nudged_zp) * scale

    shape = [1] * x.dim()
    shape[-1] = channel
    n_min = nudged_min.reshape(shape)
    n_max = nudged_max.reshape(shape)
    zero_both = ((mn == 0) & (mx == 0)).reshape(shape)

    between = ((x >= n_min) & (x <= n_max)).to(x.dtype)
    # Known benign divergence from the kernel's sentinel implementation for
    # zero_both channels: TF (and this reference) propagate bpx = g even when x
    # is NaN, while the kernel's [-inf, +inf] nudged-interval sentinel yields 0
    # for NaN comparisons (documented in the regbase kernel comment). Test data
    # does not cover NaN in this channel.
    bpx = torch.where(zero_both, g, g * between)

    min_mask = ((x < n_min) & ~zero_both).to(x.dtype)
    max_mask = ((x > n_max) & ~zero_both).to(x.dtype)
    # Both-zero channels contribute exactly zero (TF short-circuit): a plain g*mask would
    # give inf*0=NaN for +/-inf gradients and pollute the reductions.
    zero_g = torch.zeros_like(g)
    min_contrib = torch.where(zero_both, zero_g, g * min_mask)
    max_contrib = torch.where(zero_both, zero_g, g * max_mask)
    if x.dim() > 1:
        dims = tuple(range(x.dim() - 1))
        bpm = min_contrib.to(torch.float64).sum(dim=dims).to(x.dtype)
        bpmx = max_contrib.to(torch.float64).sum(dim=dims).to(x.dtype)
    else:
        bpm = min_contrib
        bpmx = max_contrib

    return [
        (bpx * 1.0).numpy(),
        bpm.numpy(),
        bpmx.numpy(),
    ]


class _TfCompetitor:
    def __init__(self, num_bits=None, narrow_range=None, **kwargs):
        self.num_bits = 8 if num_bits is None else int(num_bits)
        self.narrow_range = False if narrow_range is None else bool(narrow_range)

    def __call__(self, gradients, x, min, max, **kwargs):
        import tensorflow as tf
        import torch

        def arr(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        return tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient(
            arr(gradients),
            arr(x),
            arr(min),
            arr(max),
            num_bits=self.num_bits,
            narrow_range=self.narrow_range,
        )


class FakeQuantWithMinMaxVarsPerChannelGradientTestSpec:
    """TestSpec for the newer TTK spec convention (Kernel/GEIR route)."""

    def golden(gradients, x, min, max, num_bits=8, narrow_range=False, **kwargs):
        return fake_quant_with_min_max_vars_per_channel_gradient_golden(
            gradients, x, min, max, num_bits=num_bits, narrow_range=narrow_range
        )

    third_party = {"tf": _TfCompetitor}
    # Tolerance per spec.yaml numerical_tolerance (rtol/atol 1e-5, max_relative):
    # tightens the mix_tolerance table values for this quantization compute operator.
    tolerance = {
        "float32": {"standard": "mix_tolerance", "rtol": 1.0e-5, "atol": 1.0e-5}
    }
