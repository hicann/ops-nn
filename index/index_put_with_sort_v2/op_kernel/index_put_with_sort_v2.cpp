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
 * \file index_put_with_sort_v2.cpp
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/index_put_with_sort_v2.h"

extern "C" __global__ __aicore__ void index_put_with_sort_v2(GM_ADDR self, GM_ADDR linearIndex,
    GM_ADDR posIdx, GM_ADDR values, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingDataIn, tiling);
    const IndexPutWithSortV2TilingData* __restrict tilingData = &tilingDataIn;
    AscendC::TPipe tpipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
    if (TILING_KEY_IS(0)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, false, false, false> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, true, false, false> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, false, true, false> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, true, true, false> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(100)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, false, false, true> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, true, false, true> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(110)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, false, true, true> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    } else if (TILING_KEY_IS(111)) {
        AscendC::IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, true, true, true> op(&tpipe, tilingData);
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();
    }
}