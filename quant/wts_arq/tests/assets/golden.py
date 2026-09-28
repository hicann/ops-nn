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
"""WtsARQ test assets: golden + third_party + tolerance (TTK TestSpec).

Framework choice (TTK three-leg golden spec): the operator ships a TensorFlow
frontend plugin (framework/wts_arq_tf_plugin.cpp maps OriginOpType "WtsARQ"),
so the CPU golden is implemented with TensorFlow; numpy is used only for I/O
conversion and the torch third_party stays the independent GPU-leg reference.

Semantics (README.md formula, kernel.h and DESIGN.md §3.5/§3.6):
    range fix:  w_min = min(w_min, 0), w_max = max(w_max, 0)
    scale:      offset_flag=true  -> w_max/255 - w_min/255
                offset_flag=false -> max(|w_min|/128, w_max/127)
    eps guard:  scale < float32 eps -> scale = 1.0
    quant:      q = clamp(rint(w/scale) [+ offset], -128, 127) [- offset],
                offset = -rint(w_min/scale) - 128
    dequant:    y = q * scale

Cast rule (float only, no integer outputs in @wts_arq_def.cpp):
    - tolerance standard is mix_tolerance, i.e. two-leg comparison; TTK does
      not Promote, so the golden reproduces the NPU kernel width itself:
      LoadWiden fp16 -> fp32 for the whole computation, CAST_RINT (half to
      even) narrow back to the dispatched input dtype at the exit.
    - if a caller switches the judge to cross_check, TTK runs Promote and the
      golden receives lifted arrays/output_dtypes; golden_mode==Promote is
      therefore detected and the golden then does zero cast (no entry or exit
      narrowing), letting TTK own the promotion.
"""

import numpy as np
import torch

__golden__ = {
    "kernel": {"wts_arq": "wts_arq_golden"},
    "e2e": {"tf.compat.v1.WtsARQ": "wts_arq_e2e_golden"},
}

__spec__ = {
    "wts_arq": "WtsArqTestSpec",
    "WtsARQ": "WtsArqTestSpec",
    "tf.compat.v1.WtsARQ": "WtsArqE2eSpec",
}

SCALE_EPS = 1.1920929e-07

# Precision standard: explicit ecosystem default. torch third_party below is the
# independent reference for `--compare cross_check` (cross_check needs an
# XPU/torch provider; without one the run reports GOLDEN_FAILURE, so cross_check
# is not set as the default judge here). Keys must cover every dtype registered
# in op_host/wts_arq_def.cpp: DT_FLOAT16, DT_FLOAT.
_TOL_KERNEL = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
}

# e2e(TF frontend) route judge: same scale/format of the operator as kernel.
_TOL_E2E = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
}


def wts_arq_golden(w, w_min, w_max, num_bits=8, offset_flag=False, **kwargs):
    """CPU golden for wts_arq (kernel/GEIR route, numpy.ndarray in/out).

    Inputs follow @wts_arq_def.cpp without outputs: w, w_min, w_max; attrs:
    num_bits (only 8 supported), offset_flag. w_min/w_max use restricted
    broadcast (each dim == 1 or == the w dim).

    All arithmetic runs on TensorFlow tensors; numpy only converts the inbound
    arrays and hands the outbound array back.
    """
    import tensorflow as tf

    del (
        num_bits
    )  # the operator only supports 8 bits; other values are rejected upstream

    promoted = kwargs.get("golden_mode") == "Promote"
    out_dtype = np.asarray(w).dtype

    def _work(a):
        t = tf.convert_to_tensor(np.asarray(a))
        # non-Promote: reproduce the kernel LoadWiden (fp16 -> fp32). Promote
        # keeps TTK's lifted dtype untouched (zero cast at the entry).
        if not promoted and t.dtype in (tf.float16, tf.bfloat16):
            t = tf.cast(t, tf.float32)
        return t

    w_t = _work(w)
    mn_t = tf.broadcast_to(_work(w_min), tf.shape(w_t))
    mx_t = tf.broadcast_to(_work(w_max), tf.shape(w_t))

    # range fix must run before the scale formula
    mn_t = tf.minimum(mn_t, tf.zeros_like(mn_t))
    mx_t = tf.maximum(mx_t, tf.zeros_like(mx_t))

    dtype = w_t.dtype
    eps = tf.cast(SCALE_EPS, dtype)
    if offset_flag:
        scale = mx_t / tf.cast(255.0, dtype) - mn_t / tf.cast(255.0, dtype)
    else:
        scale = tf.maximum(
            tf.abs(mn_t) / tf.cast(128.0, dtype), mx_t / tf.cast(127.0, dtype)
        )
    # scale < eps -> 1.0; NaN keeps NaN (comparison is False)
    scale = tf.where(scale < eps, tf.ones_like(scale), scale)

    q = tf.round(w_t / scale)
    if offset_flag:
        offset = -tf.round(mn_t / scale) - tf.cast(128.0, dtype)
        q = (
            tf.clip_by_value(q + offset, tf.cast(-128.0, dtype), tf.cast(127.0, dtype))
            - offset
        )
    else:
        q = tf.clip_by_value(q, tf.cast(-128.0, dtype), tf.cast(127.0, dtype))

    y = q * scale
    y_np = y.numpy()
    if not promoted:
        # Reproduce NarrowStore / CAST_RINT; promoted runs return the lifted
        # dtype unchanged (TTK restores the declared dtypes itself).
        y_np = y_np.astype(out_dtype, copy=False)
    return [y_np]


def wts_arq_e2e_golden(w, w_min, w_max, num_bits=8, offset_flag=False, **kwargs):
    """e2e(TF frontend) golden: TF tensors in, numpy arrays out.

    The e2e route feeds framework tensors through the same TestSpec dispatch;
    converting at the boundary keeps one computation source of truth.
    """

    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    return wts_arq_golden(
        _np(w),
        _np(w_min),
        _np(w_max),
        num_bits=num_bits,
        offset_flag=offset_flag,
        **kwargs,
    )


def _torch_compute(w, w_min, w_max, offset_flag):
    """Independent torch implementation used as the cross_check third_party."""
    w32 = torch.as_tensor(w).detach().cpu().to(torch.float32)
    mn32 = torch.as_tensor(w_min).detach().cpu().to(torch.float32).expand_as(w32)
    mx32 = torch.as_tensor(w_max).detach().cpu().to(torch.float32).expand_as(w32)

    mn32 = torch.minimum(mn32, torch.zeros_like(mn32))
    mx32 = torch.maximum(mx32, torch.zeros_like(mx32))

    if offset_flag:
        scale = mx32 / 255.0 - mn32 / 255.0
    else:
        scale = torch.maximum(torch.abs(mn32) / 128.0, mx32 / 127.0)
    scale = torch.where(scale < float(SCALE_EPS), torch.ones_like(scale), scale)

    q = torch.round(w32 / scale)
    if offset_flag:
        offset = -torch.round(mn32 / scale) - 128.0
        q = torch.clamp(q + offset, -128.0, 127.0) - offset
    else:
        q = torch.clamp(q, -128.0, 127.0)

    y32 = (q.to(torch.float64) * scale.to(torch.float64)).to(torch.float32)
    return y32


class _WtsArqCompose:
    """third_party entry: framework passes inputs positionally, attrs as kwargs."""

    def __init__(self, num_bits=8, offset_flag=False, **kwargs):
        del num_bits
        self.offset_flag = bool(offset_flag)

    def __call__(self, w, w_min, w_max, **kwargs):
        out_dtype = torch.as_tensor(w).dtype
        return [_torch_compute(w, w_min, w_max, self.offset_flag).to(out_dtype)]


class WtsArqTestSpec:
    """wts_arq kernel/GEIR TestSpec (GEIR reuses the kernel registration)."""

    golden = wts_arq_golden
    third_party = {"torch": _WtsArqCompose}
    tolerance = _TOL_KERNEL


class WtsArqE2eSpec:
    """wts_arq e2e(TF frontend) TestSpec: same golden compute, own judge."""

    third_party = {"torch": _WtsArqCompose}
    tolerance = _TOL_E2E


# Delivery scope:
#   registered: kernel + GEIR (GEIR reuses the kernel TestSpec) + e2e(TF frontend)
#   not registered: aclnn/torch — the operator declares GE IR graph mode only
#   (README.md "仅 GE IR 图模式"); the TensorFlow frontend is mapped through
#   framework/wts_arq_tf_plugin.cpp (OriginOpType "WtsARQ").
