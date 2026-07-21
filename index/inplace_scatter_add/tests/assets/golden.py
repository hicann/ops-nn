#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch

__golden__ = {"kernel": {"inplace_scatter_add": "inplace_scatter_add_golden"}}


def inplace_scatter_add_golden(var, indices, updates, **kwargs):
    var_t = torch.from_numpy(var).clone()
    indices_t = torch.from_numpy(indices)
    updates_t = torch.from_numpy(updates)
    var_t.index_add_(0, indices_t, updates_t)

    return var_t.numpy()
