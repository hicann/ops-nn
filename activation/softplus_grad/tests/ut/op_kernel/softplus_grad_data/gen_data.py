#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np

try:
    import ml_dtypes

    BFLOAT16 = ml_dtypes.bfloat16
except ImportError:
    BFLOAT16 = None


def gen_golden_data_simple(shape_str, dtype, data_case):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    if BFLOAT16 is not None:
        dtype_dict["bfloat16_t"] = BFLOAT16
    np_dtype = dtype_dict[dtype]

    shape_list = [int(x) for x in shape_str.strip("(").strip(")").split(",")]
    shape = tuple(shape_list)

    if data_case == "special":
        gradients = np.array(
            [
                3e38,
                -3e38,
                np.inf,
                -np.inf,
                np.inf,
                -np.inf,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.nan,
                0.0,
                -0.0,
                42.0,
                -42.0,
            ],
            dtype=np.float32,
        )
        features = np.array(
            [
                -104.0,
                -104.0,
                0.0,
                0.0,
                -np.inf,
                -np.inf,
                -89.0,
                -103.0,
                -104.0,
                np.inf,
                np.nan,
                1.0,
                np.nan,
                -0.0,
                -120.0,
                120.0,
            ],
            dtype=np.float32,
        )
        data_gradients = np.resize(gradients, shape).astype(np_dtype)
        data_features = np.resize(features, shape).astype(np_dtype)
    else:
        data_gradients = np.random.uniform(-2, 2, shape).astype(np_dtype)
        data_features = np.random.uniform(-2, 2, shape).astype(np_dtype)

    # softplus(x) = log(1 + exp(x)), 导数 = sigmoid(x) = 1 / (1 + exp(-x))
    # softplus_grad(gradients, features) = gradients * sigmoid(features)
    x_dtype = data_features.dtype
    feat_f32 = data_features.astype(np.float32)
    grad_f32 = data_gradients.astype(np.float32)

    sigmoid_feat = 1.0 / (1.0 + np.exp(-feat_f32))
    golden_f32 = grad_f32 * sigmoid_feat
    golden = golden_f32.astype(x_dtype, copy=False)

    data_gradients.tofile(f"{dtype}_gradients.bin")
    data_features.tofile(f"{dtype}_features.bin")
    golden.tofile(f"{dtype}_golden.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python gen_data.py <shape> <dtype> <random|special>")
        print("Example: python gen_data.py '(128, 64)' float32 special")
        exit(1)
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3])
