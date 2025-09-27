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
 * \file apply_top_k_top_p_with_sorted.cpp
 * \brief
 */

#include "apply_top_k_top_p_with_sorted.h"

using namespace AscendC;
using namespace ApplyTopKTopPWithSortedOp;

extern "C" __global__ __aicore__ void apply_top_k_top_p_with_sorted(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ApplyTopKTopPWithSortedOp::ApplyTopKTopPWithSorted<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ApplyTopKTopPWithSortedOp::ApplyTopKTopPWithSorted<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.ProcessTopK();
    } else if (TILING_KEY_IS(2)) {
        ApplyTopKTopPWithSortedOp::ApplyTopKTopPWithSorted<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.ProcessTopP();
    }
}