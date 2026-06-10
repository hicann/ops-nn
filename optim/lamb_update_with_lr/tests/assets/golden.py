#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0. See LICENSE for details.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {
    "kernel": {
        "lamb_update_with_lr": "lamb_update_with_lr_golden"
    }
}


def _scalars(*xs):
    return tuple(float(np.asarray(x, 'float32').reshape(-1)[0]) for x in xs)


def lamb_update_with_lr_golden(input_greater1, input_greater_realdiv, input_realdiv, input_mul0, input_mul1, input_sub, greater_y, select_e, minimum_y, **kwargs):
    """Golden for LambUpdateWithLr. Params follow lamb_update_with_lr_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = input_mul1.dtype
    g1, grd, rd, lr, gy, se, miny = _scalars(input_greater1, input_greater_realdiv, input_realdiv, input_mul0, greater_y, select_e, minimum_y)
    upd, param = input_mul1.astype('float32'), input_sub.astype('float32')
    realdiv0 = grd / rd
    select0 = realdiv0 if g1 > gy else se
    select1 = select0 if grd > gy else se
    clip = max(min(select1, miny), gy)
    return [(param - clip * lr * upd).astype(dt)]
