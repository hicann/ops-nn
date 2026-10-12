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
"""
TTK kernel input generator for swiglu_mx_quant_with_dual_axis.

The op's 2nd input ``group_index`` is a cumulative row-boundary vector (int64,
non-decreasing, last element == M = x.shape[-2]). TTK fills raw input tensors
with random values from ``input_data_ranges``; a random vector is NOT a valid
cumsum, so this module rewrites ``group_index`` IN PLACE into a deterministic,
contract-valid partition BEFORE both the golden and the NPU kernel consume it
(TTK feeds the SAME sanitized tensor to both, so the comparison stays fair).

``GROUP_PLANS`` is the single source of truth for the grouped partitions and is
imported by the CSV generator so the declared output shapes can never drift from
the boundaries actually fed at run time. Each plan is a cumsum list ending at M
and exercises one boundary flavour:

    aligned     : every boundary a multiple of 64 (block-span aligned)
    unaligned   : boundaries not aligned to 32/64, plus a non-aligned tail
    emptygrp    : a zero-row (empty) group in the middle (repeated cumsum value)
    tail        : a short, non-64-aligned final group
    smallrange  : quant-boundary data range with a non-aligned tail
"""

import numpy as np

__input__ = {
    "kernel": {"swiglu_mx_quant_with_dual_axis": "swiglu_mx_quant_with_dual_axis_input"}
}

# suffix tag (contained in testcase_name) -> cumsum boundaries (last == M)
GROUP_PLANS = {
    "grp_fp16_e5m2_rint_alg0_aligned": [64, 128],
    "grp_fp16_e4m3_rint_alg0_unaligned": [50, 90, 130],
    "grp_fp16_e2m1_floor_alg0_emptygrp": [64, 64, 128],
    "grp_bf16_e1m2_rint_alg0_tail": [70, 100],
    "grp_fp16_e5m2_rint_alg0_smallrange": [32, 96],
}


def resolve_group_index(testcase_name, m, g_len):
    """Return the deterministic cumsum boundaries for a testcase.

    Falls back to an even ``g_len``-way split ending at ``m`` when the testcase
    name matches no registered plan (keeps any future case runnable & valid).
    """
    plan = None
    if testcase_name:
        best = ""
        for tag, boundaries in GROUP_PLANS.items():
            if tag in testcase_name and len(tag) > len(best):
                best, plan = tag, boundaries
    if plan is not None:
        if len(plan) != g_len or plan[-1] != m:
            raise ValueError(
                f"group plan {plan} inconsistent with declared shape "
                f"(G={g_len}) / M={m} for testcase {testcase_name}"
            )
        return list(plan)
    # fallback: even split
    step = max(1, m // g_len)
    boundaries = [min(m, step * (i + 1)) for i in range(g_len)]
    boundaries[-1] = m
    for i in range(1, g_len):  # enforce non-decreasing
        boundaries[i] = max(boundaries[i], boundaries[i - 1])
    return boundaries


def swiglu_mx_quant_with_dual_axis_input(x, group_index=None, **kwargs):
    """Sanitize inputs. Non-grouped cases (group_index None) pass through.

    Parameter names/order follow swiglu_mx_quant_with_dual_axis_def.cpp (inputs
    only). TTK (>=2.8) validates that the returned list has the same number of
    flat arrays as the declared inputs, and feeds the returned tensors to BOTH
    the golden and the NPU kernel. The op declares TWO inputs (x + optional
    group_index), so both cases MUST return two slots:

      * non-grouped: ``[x, None]`` — the ``None`` slot is preserved by
        ``deep_flatten`` and keeps the returned count == declared count (2);
      * grouped: ``[x, group_index]`` with ``group_index`` rewritten in place
        to a contract-valid cumsum before it reaches the golden/kernel.
    """
    if group_index is None:
        return [x, None]
    gi = np.asarray(group_index)
    m = int(x.shape[-2])
    g_len = int(gi.reshape(-1).shape[0])
    boundaries = resolve_group_index(kwargs.get("testcase_name"), m, g_len)
    gi.reshape(-1)[:] = np.asarray(boundaries, dtype=gi.dtype)
    return [x, group_index]
