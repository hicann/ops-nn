/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_slice.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/sparse_slice_empty.h"
#include "arch35/sparse_slice_dimension_base.h"
#include "arch35/sparse_slice_simt_local.h"

#define TILING_KEY_DIMENSION_BASE 10000
#define TILING_KEY_EMPTY_OUTPUT 20000
#define TILING_KEY_SIMT_OUTPUT 40000

using namespace SparseSlice;

extern "C" __global__ __aicore__ void sparse_slice(
    GM_ADDR indices, GM_ADDR values, GM_ADDR shape, GM_ADDR start, GM_ADDR size, GM_ADDR yIndices, GM_ADDR yValues,
    GM_ADDR yShape, GM_ADDR outputShape1, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(TILING_KEY_DIMENSION_BASE)) {
        TPipe pipe;
        SparseSliceDimension<DTYPE_VALUES> op;
        op.Init(indices, values, shape, start, size, yIndices, yValues, yShape, outputShape1, userWS, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_EMPTY_OUTPUT)) {
        TPipe pipe;
        SparseSliceEmpty<DTYPE_VALUES> op;
        op.Init(indices, values, shape, start, size, yIndices, yValues, yShape, outputShape1, userWS, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_OUTPUT)) {
        TPipe pipe;
        SparseSliceSimt<int64_t, DTYPE_VALUES> op;
        op.Init(indices, values, shape, start, size, yIndices, yValues, yShape, outputShape1, userWS, tilingData, &pipe);
        op.Process();
    }
}
