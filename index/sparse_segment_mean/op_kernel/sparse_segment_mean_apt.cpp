/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_segment_mean_apt.cpp
 * \brief
 */

#include "arch35/sparse_segment_mean_simd.h"
#include "arch35/all_clear.h"
#include "arch35/sparse_segment_mean_multi_core_add.h"
#include "arch35/sparse_segment_mean_simt_large_inner.h"
#include "arch35/sparse_segment_mean_simt_small_inner.h"
#include "arch35/sparse_segment_mean_full_load.h"
#include "arch35/sparse_segment_mean_full_load_small_inner.h"
#include "arch35/sparse_segment_mean_simt_loop.h"

using namespace AscendC;
using namespace SparseSegmentMeanNameSpace;

#define SIMT_LARGE_INNER_TILING_KEY 200
#define SIMT_SMALL_INNER_TILING_KEY 300
#define SIMT_LOOP_TILING_KEY 400
#define SIMD_TILING_KEY 100
#define FULL_LOAD_TILING_KEY 10
#define FULL_LOAD_SMALL_INNER_TILING_KEY 11

template <typename T1, typename T2, typename T3>
__aicore__ inline void invokeTemplateMeanSimtLarge(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace,
    const SparseSegmentMeanSimtTilingData& tilingData)
{
    SparseSegmentMeanSimtLargeInner<T1, T2, T3> op;
    op.Init(x, indices, segment_ids, y, workspace, &tilingData);
    op.Process();
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void invokeTemplateMeanSimtSmall(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace,
    const SparseSegmentMeanSimtTilingData& tilingData)
{
    SparseSegmentMeanSimtSmallInner<T1, T2, T3> op;
    op.Init(x, indices, segment_ids, y, workspace, &tilingData);
    op.Process();
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void invokeTemplateMeanSimtLoop(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace,
    const SparseSegmentMeanSimtTilingData& tilingData)
{
    SparseSegmentMeanSimtLoop<T1, T2, T3> op;
    op.Init(x, indices, segment_ids, y, workspace, &tilingData);
    op.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateAllClear(
    GM_ADDR output, const SparseSegmentMeanSimdTilingData* tilingData, AscendC::TPipe& pipeIn)
{
    AllClear<T> allClearInstance;
    allClearInstance.Init(output, tilingData, pipeIn);
    allClearInstance.Process();
    allClearInstance.SyncALLCores();
    pipeIn.Reset();
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void invokeTemplateMeanSimd(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace,
    const SparseSegmentMeanSimdTilingData* tilingData, AscendC::TPipe& pipeIn)
{
    SparseSegmentMeanSimdKernel<T1, T2, T3> op;
    op.Init(x, indices, segment_ids, y, workspace, pipeIn, tilingData);
    op.Process();
    SyncAll();
    pipeIn.Reset();
}

template <typename T1>
__aicore__ inline void invokeTemplateMultiCoreAdd(
    GM_ADDR y, GM_ADDR workspace, const SparseSegmentMeanSimdTilingData* tilingData, AscendC::TPipe& pipeIn)
{
    SparseSegmentMeanMultiCoreKernel<T1> op;
    op.Init(y, workspace, pipeIn, tilingData);
    op.Process();
    pipeIn.Reset();
}

extern "C" __global__ __aicore__ void sparse_segment_mean(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SparseSegmentMeanSimtTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10", SparseSegmentMeanFullLoadTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 11", SparseSegmentMeanFullLoadTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 100", SparseSegmentMeanSimdTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 200", SparseSegmentMeanSimtTilingData);
    AscendC::TPipe pipeIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(FULL_LOAD_SMALL_INNER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SparseSegmentMeanFullLoadTilingData, tilingData, tiling);
        SparseSegmentMeanFullLoadSmallInner<DTYPE_X, DTYPE_INDICES, DTYPE_SEGMENT_IDS> op(tilingData, pipeIn);
        op.Init(x, indices, segment_ids, y, workspace);
        op.Process();
    } else if (TILING_KEY_IS(FULL_LOAD_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SparseSegmentMeanFullLoadTilingData, tilingData, tiling);
        SparseSegmentMeanFullLoad<DTYPE_X, DTYPE_INDICES, DTYPE_SEGMENT_IDS> op(tilingData, pipeIn);
        op.Init(x, indices, segment_ids, y, workspace);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_LARGE_INNER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SparseSegmentMeanSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtLarge<DTYPE_X, DTYPE_INDICES, DTYPE_SEGMENT_IDS>(x, indices, segment_ids, y, workspace, tilingData);
    } else if (TILING_KEY_IS(SIMT_SMALL_INNER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SparseSegmentMeanSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtSmall<DTYPE_X, DTYPE_INDICES, DTYPE_SEGMENT_IDS>(x, indices, segment_ids, y, workspace, tilingData);
    } else if (TILING_KEY_IS(SIMT_LOOP_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SparseSegmentMeanSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtLoop<DTYPE_X, DTYPE_INDICES, DTYPE_SEGMENT_IDS>(x, indices, segment_ids, y, workspace, tilingData);
    } else if (TILING_KEY_IS(SIMD_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SparseSegmentMeanSimdTilingData, tilingData, tiling);
        invokeTemplateAllClear<DTYPE_X>(y, &tilingData, pipeIn);
        invokeTemplateMeanSimd<DTYPE_X, DTYPE_INDICES, DTYPE_SEGMENT_IDS>(x, indices, segment_ids, y, workspace, &tilingData, pipeIn);
        invokeTemplateMultiCoreAdd<DTYPE_X>(y, workspace, &tilingData, pipeIn);
    }
}