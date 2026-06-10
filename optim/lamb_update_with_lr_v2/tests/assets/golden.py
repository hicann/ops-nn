#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0. See LICENSE for details.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {
    "kernel": {
        "lamb_update_with_lr_v2": "lamb_update_with_lr_v2_golden"
    }
}


def _scalars(*xs):
    return tuple(float(np.asarray(x, 'float32').reshape(-1)[0]) for x in xs)


def lamb_update_with_lr_v2_golden(x1, x2, x3, x4, x5, greater_y, select_e, **kwargs):
    """Golden for LambUpdateWithLrV2. Params follow lamb_update_with_lr_v2_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = x4.dtype
    a, b, lr, gy, se = _scalars(x1, x2, x3, greater_y, select_e)
    upd, param = x4.astype('float32'), x5.astype('float32')
    inner = (a / b) if b > gy else se
    ratio = inner if a > gy else se
    return [(param - lr * ratio * upd).astype(dt)]
