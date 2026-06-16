#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0. See LICENSE for details.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {
    "kernel": {
        "lamb_apply_weight_assign": "lamb_apply_weight_assign_golden"
    }
}


def _scalars(*xs):
    return tuple(float(np.asarray(x, 'float32').reshape(-1)[0]) for x in xs)


def lamb_apply_weight_assign_golden(input0, input1, input2, input3, input_param, **kwargs):
    """Golden for LambApplyWeightAssign. Params follow lamb_apply_weight_assign_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = input3.dtype
    wn, gn, lr = _scalars(input0, input1, input2)
    upd, param = input3.astype('float32'), input_param.astype('float32')
    inner = (wn / gn) if gn > 0 else 1.0
    ratio = inner if wn > 0 else 1.0
    return [(param - lr * ratio * upd).astype(dt)]
