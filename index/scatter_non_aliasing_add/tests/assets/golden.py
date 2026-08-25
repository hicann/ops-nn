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

import tensorflow as tf

__spec__ = {
    "scatter_non_aliasing_add": "ScatterNonAliasingAddKernelSpec",
}


def _validate_indices_rank(indices):
    indices = tf.convert_to_tensor(indices)
    tf.debugging.assert_rank_at_least(
        indices,
        2,
        message="The rank of indices must be >= 2",
    )
    return indices


class ScatterNonAliasingAddKernelSpec:
    @staticmethod
    def golden(x, indices, updates, **kwargs):
        indices = _validate_indices_rank(indices)
        out = tf.raw_ops.ScatterNdNonAliasingAdd(
            input=x,
            indices=indices,
            updates=updates,
        )
        out = out.numpy()
        return out

    class ThirdPartyImpl:
        """TensorFlow GPU golden。"""

        def __init__(self, x, indices, updates, **kwargs):
            self.x = tf.convert_to_tensor(x)
            self.indices = _validate_indices_rank(indices)
            self.updates = tf.convert_to_tensor(updates)

        def __call__(self):
            out = tf.raw_ops.ScatterNdNonAliasingAdd(
                input=self.x,
                indices=self.indices,
                updates=self.updates,
            )
            return out

    third_party = {"tf": ThirdPartyImpl}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "int32": {"standard": "binary_equal", "level": "L1"},
    }
