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
 * \file segment_sum.cpp
 * \brief segment_sum kernel
 */

#include "arch35/segment_sum_simt.h"
#include "arch35/segment_sum_simd.h"
#include "arch35/clear_output.h"
#include "arch35/segment_sum_struct.h"

using namespace AscendC;
using namespace SegmentSum;

#define TEMPLATE_SIMT_TILING_KEY  1000
#define SIMD_ATOMIC_SUPPORT_TILING_KEY  2000

template <typename T>
__aicore__ inline void invokeTemplateAllClear(
    GM_ADDR output, const SegmentSumSimdTilingData* tilingData, AscendC::TPipe& pipeIn)
{
    AllClear<T> allClearInstance;
    allClearInstance.Init(output, tilingData, pipeIn);
    allClearInstance.Process();
    allClearInstance.SyncALLCores();
    pipeIn.Reset();
}

extern "C" __global__ __aicore__ void segment_sum(GM_ADDR x, GM_ADDR segment_ids,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SegmentSumTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1000", SegmentSumSimtTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2000", SegmentSumSimdTilingData);

    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(TEMPLATE_SIMT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SegmentSumSimtTilingData, simtTilingData, tiling);
        const SegmentSumSimtTilingData* __restrict tilingData = &simtTilingData; 
        SegmentSumSimt<DTYPE_X, DTYPE_SEGMENT_IDS> op(tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_ATOMIC_SUPPORT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SegmentSumSimdTilingData, tilingData, tiling);
        if constexpr (!(std::is_same_v<DTYPE_X, uint32_t> || std::is_same_v<DTYPE_X, uint64_t> || std::is_same_v<DTYPE_X, int64_t>)) {
            invokeTemplateAllClear<DTYPE_X>(output, &tilingData, pipe);
            SegmentSumSimd<DTYPE_X, DTYPE_SEGMENT_IDS> op;
            op.Init(x, segment_ids, output, workspace, pipe, &tilingData);
            op.Process();
        }
    }
}