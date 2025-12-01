/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file max_pool3d_with_argmax_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "max_pool3d_with_argmax_v2_base.h"
#include "max_pool3d_with_argmax_v2_no_expand_indices.h"
#include "max_pool3d_with_argmax_big_kernel.h"

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(300001UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict__ tilingData = &tilingDataIn;
#if ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_DISABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_DISABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(300002UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict__ tilingData = &tilingDataIn;
#if ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_ENABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_ENABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(311110)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict__ tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<float, float, false> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 0);
        op.Process();
    } else if (TILING_KEY_IS(311111)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict__ tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<half, half, false> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 1);
        op.Process();
    } else if (TILING_KEY_IS(311112)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict__ tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<bfloat16_t, float, false> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 2);
        op.Process();
    }

    return;
}