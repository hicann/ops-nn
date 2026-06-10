#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0. See LICENSE for details.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {
    "kernel": {
        "lamb_next_mv": "lamb_next_mv_golden"
    }
}


def _scalars(*xs):
    return tuple(float(np.asarray(x, 'float32').reshape(-1)[0]) for x in xs)


def lamb_next_mv_golden(input_mul3, input_mul2, input_realdiv1, input_mul1, input_mul0, input_realdiv0, input_mul4, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x, add2_y, **kwargs):
    """Golden for LambNextMV. Params follow lamb_next_m_v_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = input_mul3.dtype
    g2, v, g, m, param = [x.astype('float32') for x in (input_mul3, input_mul2, input_mul1, input_mul0, input_mul4)]
    rd1, rd0, b1, omb1, b2, omb2, wd, eps = _scalars(input_realdiv1, input_realdiv0, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x, add2_y)
    next_v = v * b2 + g2 * omb2
    next_m = m * b1 + g * omb1
    v_unb, m_unb = next_v / rd1, next_m / rd0
    y1 = param * wd + m_unb / np.sqrt(v_unb + eps)
    y4 = m_unb / (np.sqrt(v_unb) + eps)
    return [y1.astype(dt), next_m.astype(dt), next_v.astype(dt), y4.astype(dt)]
