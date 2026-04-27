/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/bucketize_v2_full_load.h"
#include "arch35/bucketize_v2_cascade.h"
#include "arch35/bucketize_v2_tiling_key.h"
#include "arch35/bucketize_v2_struct.h"
using namespace AscendC;

template <uint64_t TEMPLATE_MODE, bool BOUNDARY_MODE>
__global__ __aicore__ void bucketize_v2(GM_ADDR x, GM_ADDR boundaries, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) 
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
    REGISTER_TILING_DEFAULT(BucketizeV2TilingData);
    if constexpr (TEMPLATE_MODE == TPL_MODE_FULL_LOAD && BOUNDARY_MODE == TPL_MODE_NO_RIGHT) {
        GET_TILING_DATA_WITH_STRUCT(BucketizeV2FullLoadTilingData, tilingDataIn, tiling);
        BucketizeV2::BucketizeV2FullLoad<DTYPE_X, DTYPE_BOUNDARIES, DTYPE_Y, false> op(&pipe);
        op.Init(x, boundaries, y, &tilingDataIn);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_FULL_LOAD && BOUNDARY_MODE == TPL_MODE_RIGHT) {
        GET_TILING_DATA_WITH_STRUCT(BucketizeV2FullLoadTilingData, tilingDataIn, tiling);
        BucketizeV2::BucketizeV2FullLoad<DTYPE_X, DTYPE_BOUNDARIES, DTYPE_Y, true> op(&pipe);
        op.Init(x, boundaries, y, &tilingDataIn);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_CASCADE && BOUNDARY_MODE == TPL_MODE_NO_RIGHT) {
        GET_TILING_DATA_WITH_STRUCT(BucketizeV2CascadeTilingData, tilingDataIn, tiling);
        BucketizeV2::BucketizeV2Cascade<DTYPE_X, DTYPE_BOUNDARIES, DTYPE_Y, false> op(&pipe);
        op.Init(x, boundaries, y, &tilingDataIn);
        op.Process();        
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_CASCADE && BOUNDARY_MODE == TPL_MODE_RIGHT) {
        GET_TILING_DATA_WITH_STRUCT(BucketizeV2CascadeTilingData, tilingDataIn, tiling);
        BucketizeV2::BucketizeV2Cascade<DTYPE_X, DTYPE_BOUNDARIES, DTYPE_Y, true> op(&pipe);
        op.Init(x, boundaries, y, &tilingDataIn);
        op.Process();       
    }
}