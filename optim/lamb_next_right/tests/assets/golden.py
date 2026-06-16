#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0. See LICENSE for details.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {
    "kernel": {
        "lamb_next_right": "lamb_next_right_golden"
    }
}


def _scalars(*xs):
    return tuple(float(np.asarray(x, 'float32').reshape(-1)[0]) for x in xs)


def lamb_next_right_golden(input_square, input_mul2, mul2_x, mul3_x, truediv1_recip, add2_y, **kwargs):
    """Golden for LambNextRight. Params follow lamb_next_right_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = input_square.dtype
    g, v = input_square.astype('float32'), input_mul2.astype('float32')
    b2, omb2, recip, eps = _scalars(mul2_x, mul3_x, truediv1_recip, add2_y)
    next_v = v * b2 + g * g * omb2
    y2 = np.sqrt(next_v * recip) + eps
    return [next_v.astype(dt), y2.astype(dt)]
