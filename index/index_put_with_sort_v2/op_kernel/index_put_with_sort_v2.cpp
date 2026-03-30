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
#include "arch35/index_put_with_sort_v2_simd.h"

using namespace AscendC;
using namespace IndexPutWithSortV2;

template<bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE, bool IS_CAST, bool IS_SIMD>
__global__ __aicore__ void index_put_with_sort_v2(GM_ADDR self, GM_ADDR linearIndex,
    GM_ADDR posIdx, GM_ADDR values, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(IndexPutWithSortV2SimdTilingData);
 	REGISTER_TILING_FOR_TILINGKEY("IS_SIMD == false", IndexPutWithSortV2TilingData);
 	REGISTER_TILING_FOR_TILINGKEY("IS_SIMD == true", IndexPutWithSortV2SimdTilingData);
 	 
 	KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe tpipe;

    if constexpr (IS_SIMD) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutWithSortV2SimdTilingData, tilingData, tiling);
        if constexpr (ACCUMULATE) {
            if constexpr (IS_CAST) {
                if constexpr (IsSameType<DTYPE_SELF, float16_t>::value) {
                    IndexPutWithSortV2SIMDKernel<float16_t, DTYPE_LINEAR_INDEX, ACCUMULATE, true, float> op(&tpipe, &tilingData);
                    op.Init(self, linearIndex, posIdx, values, output, workSpace);
                    op.Process();
                } else if (IsSameType<DTYPE_SELF, bfloat16_t>::value) {
                    IndexPutWithSortV2SIMDKernel<bfloat16_t, DTYPE_LINEAR_INDEX, ACCUMULATE, true, float> op(&tpipe, &tilingData);
                    op.Init(self, linearIndex, posIdx, values, output, workSpace);
                    op.Process();                    
                }
            } else {
                IndexPutWithSortV2SIMDKernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, ACCUMULATE, false, DTYPE_SELF> op(&tpipe, &tilingData);
                op.Init(self, linearIndex, posIdx, values, output, workSpace);
                op.Process();
            }
        } else {
            IndexPutWithSortV2SIMDKernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, ACCUMULATE, IS_CAST, DTYPE_SELF> op(&tpipe, &tilingData);
            op.Init(self, linearIndex, posIdx, values, output, workSpace);
            op.Process();
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(IndexPutWithSortV2TilingData, tilingData, tiling);
 	    IndexPutWithSortV2Kernel<DTYPE_SELF, DTYPE_LINEAR_INDEX, ACCUMULATE, ALL_INDEXED, INDEXED_BLOCK_MODE> op(&tpipe, &tilingData);   
        op.Init(self, linearIndex, posIdx, values, output);
        op.Process();     
    }
}
