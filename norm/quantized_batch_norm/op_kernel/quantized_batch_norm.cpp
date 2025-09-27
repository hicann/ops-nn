/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantized_batch_norm.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "quantized_batch_norm_welford.h"
using namespace AscendC;
using namespace QuantizedBatchNormOps;

namespace {
constexpr static int R0_SPLIT_NOT_ALIGN_MODE = 0;
constexpr static int R0_SPLIT_ALIGN_MODE = 1;
constexpr static int R1_SPLIT_NOT_ALIGN_MODE = 2;
constexpr static int R1_SPLIT_ALIGN_MODE = 3;

constexpr static int R0_NOT_ALIGN = 0;
constexpr static int R0_ALIGN = 1;
}; // namespace

extern "C" __global__ __aicore__ void quantized_batch_norm(
    GM_ADDR x, GM_ADDR mean, GM_ADDR var, GM_ADDR input_scale, GM_ADDR input_zero_point, GM_ADDR output_scale,
    GM_ADDR output_zero_point, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(QuantizedBatchNormWelfordTilingData, tiling_data_in, tiling);
    const QuantizedBatchNormWelfordTilingData* __restrict tilingData = &tiling_data_in;
    if (TILING_KEY_IS(1000)) {
        QuantizedBatchNormWelford<DTYPE_X, DTYPE_WEIGHT, R0_SPLIT_NOT_ALIGN_MODE, R0_NOT_ALIGN> op(&pipe);
        op.Init(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        QuantizedBatchNormWelford<DTYPE_X, DTYPE_WEIGHT, R0_SPLIT_ALIGN_MODE, R0_NOT_ALIGN> op(&pipe);
        op.Init(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        QuantizedBatchNormWelford<DTYPE_X, DTYPE_WEIGHT, R1_SPLIT_NOT_ALIGN_MODE, R0_NOT_ALIGN> op(&pipe);
        op.Init(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1003)) {
        QuantizedBatchNormWelford<DTYPE_X, DTYPE_WEIGHT, R1_SPLIT_ALIGN_MODE, R0_NOT_ALIGN> op(&pipe);
        op.Init(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1012)) {
        QuantizedBatchNormWelford<DTYPE_X, DTYPE_WEIGHT, R1_SPLIT_NOT_ALIGN_MODE, R0_ALIGN> op(&pipe);
        op.Init(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1013)) {
        QuantizedBatchNormWelford<DTYPE_X, DTYPE_WEIGHT, R1_SPLIT_ALIGN_MODE, R0_ALIGN> op(&pipe);
        op.Init(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias, y, tilingData);
        op.Process();
    }
}