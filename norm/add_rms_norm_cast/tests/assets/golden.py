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

import numpy as np
import os


__golden__ = {
    "kernel": {
        "add_rms_norm_cast": "add_rms_norm_cast_golden"
    }
}


def add_rms_norm_cast_golden(x1, x2, gamma, # inputs
                             epsilon: float=1e-6, # attributes
                             **kwargs):
    '''
    Golden function for add_rms_norm_cast.
    All the parameters (names and order) follow @add_rms_norm_cast_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    from pathlib import Path
    current_file = Path(__file__).resolve()
    target_file = Path(os.path.join(current_file.parent, '..', '..', '..',
        'add_rms_norm', 'tests', 'assets', 'golden.py')).resolve()

    add_rms_norm = import_from_path('golden', target_file)

    kwargs.update({'_post_action': 'cast'})
    return add_rms_norm.add_rms_norm_golden(x1, x2, gamma, epsilon, **kwargs)


def import_from_path(module_name, file_path):
    '''import module dynamicly'''
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module