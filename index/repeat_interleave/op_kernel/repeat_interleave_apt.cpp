/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave.cpp
 * \brief
 */

#include "arch35/repeat_interleave.h"
#include "arch35/repeat_interleave_small.h"
#include "arch35/repeat_interleave_scalar.h"
#include "arch35/cum_sum_exclusive.h"
#include "arch35/cum_sum_split_batch_simt.h"
#include "arch35/cum_sum_split_repeat_simt.h"

using namespace RepeatInterleave;

#define BATCH_TILING_REPEAT_SCALAR 101
#define BATCH_TILING_REPEAT_TENSOR 102
#define BATCH_TILING_SPLIT_REPEATS_SHAPE_INT32 201
#define BATCH_TILING_SPLIT_REPEATS_SHAPE_INT64 202
#define SPLIT_REPEAT_SUM_SIMT_ADDR_INT64_CAST 4101
#define SPLIT_REPEAT_SUM_SIMT_ADDR_INT64 4100
#define SPLIT_REPEAT_SUM_SIMT_ADDR_INT32 4000
#define SPLIT_BATCH_SIMT_ADDR_INT64_CAST 5101
#define SPLIT_BATCH_SIMT_ADDR_INT64 5100
#define SPLIT_BATCH_SIMT_ADDR_INT32 5000
#define SPLIT_BATCH_SIMT_ADDR_INT64_CAST_CUMSUMINUB 15101
#define SPLIT_BATCH_SIMT_ADDR_INT64_CUMSUMINUB 15100
#define SPLIT_BATCH_SIMT_ADDR_INT32_CUMSUMINUB 15000

extern "C" __global__ __aicore__ void repeat_interleave(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace,
                                                        GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(BATCH_TILING_SPLIT_REPEATS_SHAPE_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveTilingKernelDataSmall, tilingData, tiling);
        const RepeatInterleaveTilingKernelDataSmall& tilingInfo = tilingData;
        RepeatInterleaveSmall<DTYPE_X, DTYPE_REPEATS, int32_t> op(pipe, tilingInfo);
        op.Init(x, repeats, y, userWs);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_TILING_SPLIT_REPEATS_SHAPE_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveTilingKernelDataSmall, tilingData, tiling);
        const RepeatInterleaveTilingKernelDataSmall& tilingInfo = tilingData;
        RepeatInterleaveSmall<DTYPE_X, DTYPE_REPEATS, int64_t> op(pipe, tilingInfo);
        op.Init(x, repeats, y, userWs);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_TILING_REPEAT_SCALAR)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveTilingKernelDataNorm, tilingData, tiling);
        const RepeatInterleaveTilingKernelDataNorm& tilingInfo = tilingData;
        RepeatInterleave::RepeatInterleaveScalarImpl<DTYPE_X, DTYPE_REPEATS> op(tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_TILING_REPEAT_TENSOR)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveTilingKernelDataNorm, tilingData, tiling);
        const RepeatInterleaveTilingKernelDataNorm& tilingInfo = tilingData;
        RepeatInterleave::RepeatInterleaveImpl<DTYPE_X, DTYPE_REPEATS> op(tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_BATCH_SIMT_ADDR_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::CumSumExclusive<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, RepeatInterleaveCumSumTilingData> op(
            tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
        SyncAll();
        RepeatInterleave::SplitBatchSimt<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, int32_t> simtSplitBatchOp(tilingInfo,
                                                                                                          pipe);
        simtSplitBatchOp.Init(x, repeats, y, userWs);
        simtSplitBatchOp.Process();
    } else if (TILING_KEY_IS(SPLIT_BATCH_SIMT_ADDR_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::CumSumExclusive<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, RepeatInterleaveCumSumTilingData> op(
            tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
        SyncAll();
        RepeatInterleave::SplitBatchSimt<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, int64_t> simtSplitBatchOp(tilingInfo,
                                                                                                          pipe);
        simtSplitBatchOp.Init(x, repeats, y, userWs);
        simtSplitBatchOp.Process();
    } else if (TILING_KEY_IS(SPLIT_BATCH_SIMT_ADDR_INT64_CAST)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::CumSumExclusive<DTYPE_X, DTYPE_REPEATS, int64_t, RepeatInterleaveCumSumTilingData> op(
            tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
        SyncAll();
        RepeatInterleave::SplitBatchSimt<DTYPE_X, DTYPE_REPEATS, int64_t, int64_t> simtSplitBatchOp(tilingInfo, pipe);
        simtSplitBatchOp.Init(x, repeats, y, userWs);
        simtSplitBatchOp.Process();
    } else if (TILING_KEY_IS(SPLIT_BATCH_SIMT_ADDR_INT32_CUMSUMINUB)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::SplitBatchSimt<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, int32_t, true> simtSplitBatchOp(
            tilingInfo, pipe);
        simtSplitBatchOp.Init(x, repeats, y, userWs);
        simtSplitBatchOp.Process();
    } else if (TILING_KEY_IS(SPLIT_BATCH_SIMT_ADDR_INT64_CUMSUMINUB)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::SplitBatchSimt<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, int64_t, true> simtSplitBatchOp(
            tilingInfo, pipe);
        simtSplitBatchOp.Init(x, repeats, y, userWs);
        simtSplitBatchOp.Process();
    } else if (TILING_KEY_IS(SPLIT_BATCH_SIMT_ADDR_INT64_CAST_CUMSUMINUB)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::SplitBatchSimt<DTYPE_X, DTYPE_REPEATS, int64_t, int64_t, true> simtSplitBatchOp(tilingInfo,
                                                                                                          pipe);
        simtSplitBatchOp.Init(x, repeats, y, userWs);
        simtSplitBatchOp.Process();
    } else if (TILING_KEY_IS(SPLIT_REPEAT_SUM_SIMT_ADDR_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::CumSumExclusive<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, RepeatInterleaveCumSumTilingData> op(
            tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
        SyncAll();
        pipe.Reset();
        RepeatInterleave::SplitRepeatSumSimt<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, int32_t> simtOp(tilingInfo, pipe);
        simtOp.Init(x, repeats, y, userWs);
        simtOp.Process();
    } else if (TILING_KEY_IS(SPLIT_REPEAT_SUM_SIMT_ADDR_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::CumSumExclusive<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, RepeatInterleaveCumSumTilingData> op(
            tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
        SyncAll();
        pipe.Reset();
        RepeatInterleave::SplitRepeatSumSimt<DTYPE_X, DTYPE_REPEATS, DTYPE_REPEATS, int64_t> simtOp(tilingInfo, pipe);
        simtOp.Init(x, repeats, y, userWs);
        simtOp.Process();
    } else if (TILING_KEY_IS(SPLIT_REPEAT_SUM_SIMT_ADDR_INT64_CAST)) {
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveCumSumTilingData, tilingData, tiling);
        const RepeatInterleaveCumSumTilingData& tilingInfo = tilingData;
        RepeatInterleave::CumSumExclusive<DTYPE_X, DTYPE_REPEATS, int64_t, RepeatInterleaveCumSumTilingData> op(
            tilingInfo, pipe);
        op.Init(x, repeats, y, userWs);
        op.Process();
        SyncAll();
        pipe.Reset();
        RepeatInterleave::SplitRepeatSumSimt<DTYPE_X, DTYPE_REPEATS, int64_t, int64_t> simtOp(tilingInfo, pipe);
        simtOp.Init(x, repeats, y, userWs);
        simtOp.Process();
    }
    return;
}
