#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Golden TestSpec for npu_clear_float_status operator.

NPUClearFloatStatus is an Ascend NPU hardware status management operator.
It clears the float overflow status register on each AI core.
Output is always 8 float32 zeros, independent of input data.
"""

import tensorflow as tf

__spec__ = {
    "npu_clear_float_status": "NpuClearFloatStatusTestSpec",
}


class NpuClearFloatStatusTestSpec:
    """
    TestSpec shared by Kernel and GEIR for npu_clear_float_status.

    Input:
        addr: numpy.ndarray (Kernel) / tf.Tensor (third_party)
              float32, shape (8,) - address placeholder, data not used

    Output:
        data: float32, shape (8,) - always 8 zeros

    Precision:
        binary_equal - output is bit-exact zeros (non-computational operator)
    """

    @staticmethod
    def golden(addr, **kwargs):
        """
        Kernel/GEIR Golden function.

        Parameters:
            addr: numpy.ndarray, float32, shape (8,)
                  Address placeholder tensor. Data content does not affect output.

        Returns:
            list: [numpy.ndarray of 8 float32 zeros]
        """
        result = tf.zeros([8], dtype=tf.float32)
        return [result.numpy()]

    class ThirdPartyImpl:
        """
        Provider third-party implementation for GEIR cross-check.

        Receives original dtype tf.Tensor, independently computes output.
        Since output is always zeros, the result is deterministic
        and bit-exact regardless of input data.
        """

        def __init__(self, **kwargs):
            pass

        def __call__(self, addr, **kwargs):
            return [tf.zeros([8], dtype=tf.float32)]

    # Maps framework key to its third-party implementation class
    third_party = {"tf": ThirdPartyImpl}

    # Bit-exact float32 output: use binary_equal (output is always zeros)
    tolerance = {
        "float32": {"standard": "binary_equal"},
    }
