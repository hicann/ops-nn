#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0. See LICENSE for details.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {
    "kernel": {
        "lamb_apply_optimizer_assign": "lamb_apply_optimizer_assign_golden"
    }
}


def _scalars(*xs):
    return tuple(float(np.asarray(x, 'float32').reshape(-1)[0]) for x in xs)


def lamb_apply_optimizer_assign_golden(grad, inputv, inputm, input3, mul0_x, mul1_x, mul2_x, mul3_x, add2_y, steps, do_use_weight, weight_decay_rate, **kwargs):
    """Golden for LambApplyOptimizerAssign. Params follow lamb_apply_optimizer_assign_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = grad.dtype
    g, v, m, w = [x.astype('float32') for x in (grad, inputv, inputm, input3)]
    b1, omb1, b2, omb2, eps, t, du, wd = _scalars(mul0_x, mul1_x, mul2_x, mul3_x, add2_y, steps, do_use_weight, weight_decay_rate)
    next_v = v * b2 + g * g * omb2
    next_m = m * b1 + g * omb1
    update = (next_m / (1 - b1 ** t)) / (np.sqrt(next_v / (1 - b2 ** t)) + eps) + w * wd * du
    return [update.astype(dt), next_v.astype(dt), next_m.astype(dt)]
