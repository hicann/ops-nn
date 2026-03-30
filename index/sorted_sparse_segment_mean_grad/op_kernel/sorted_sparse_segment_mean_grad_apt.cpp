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
 * \file sorted_sparse_segment_mean_grad.cpp
 * \brief
 */

#include "arch35/sorted_sparse_segment_mean_grad_simt_large_inner.h"
#include "arch35/sorted_sparse_segment_mean_grad_simt_small_inner.h"

using namespace AscendC;
using namespace SparseSegmentMeanGradNameSpace;

#define LARGE_INNER_LT_INNER_LT_OUTTER_TILING_KEY 200
#define LARGE_INNER_LT_INNER_GT_OUTTER_TILING_KEY 201
#define LARGE_INNER_GT_INNER_LT_OUTTER_TILING_KEY 202
#define LARGE_INNER_GT_INNER_GT_OUTTER_TILING_KEY 203
#define SMALL_INNER_LT_INNER_LT_OUTTER_TILING_KEY 300
#define SMALL_INNER_LT_INNER_GT_OUTTER_TILING_KEY 301

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__aicore__ inline void invokeTemplateMeanSimtLarge(
    GM_ADDR x, GM_ADDR sorted_indices, GM_ADDR segment_ids, GM_ADDR output_dim0,
    GM_ADDR location, GM_ADDR y, GM_ADDR workspace,
    const SortedSparseSegmentMeanGradSimtTilingData& tilingData)
{
    SortedSparseSegmentMeanGradSimtLargeInner<T1, T2, T3, T4, T6, T5> op;
    op.Init(x, sorted_indices, segment_ids, output_dim0, location, y, workspace, &tilingData);
    op.Process();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__aicore__ inline void invokeTemplateMeanSimtSmall(
    GM_ADDR x, GM_ADDR sorted_indices, GM_ADDR segment_ids, GM_ADDR output_dim0,
    GM_ADDR location, GM_ADDR y, GM_ADDR workspace,
    const SortedSparseSegmentMeanGradSimtTilingData& tilingData)
{
    SortedSparseSegmentMeanGradSimtSmallInner<T1, T2, T3, T4, T6, T5> op;
    op.Init(x, sorted_indices, segment_ids, output_dim0, location, y, workspace, &tilingData);
    op.Process();
}

extern "C" __global__ __aicore__ void sorted_sparse_segment_mean_grad(
    GM_ADDR x, GM_ADDR sorted_indices, GM_ADDR pre_location_indices, GM_ADDR segment_ids, GM_ADDR output_dim0, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(SortedSparseSegmentMeanGradSimtTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 200", SortedSparseSegmentMeanGradSimtTilingData);
    AscendC::TPipe pipeIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(LARGE_INNER_LT_INNER_LT_OUTTER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SortedSparseSegmentMeanGradSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtLarge<DTYPE_X, DTYPE_SORTED_INDICES, DTYPE_PRE_LOCATION_INDICES, DTYPE_SEGMENT_IDS, uint32_t, uint32_t>(x, sorted_indices, segment_ids, output_dim0, pre_location_indices, y, userWs, tilingData);
    } else if (TILING_KEY_IS(LARGE_INNER_LT_INNER_GT_OUTTER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SortedSparseSegmentMeanGradSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtLarge<DTYPE_X, DTYPE_SORTED_INDICES, DTYPE_PRE_LOCATION_INDICES, DTYPE_SEGMENT_IDS, uint32_t, uint64_t>(x, sorted_indices, segment_ids, output_dim0, pre_location_indices, y, userWs, tilingData);
    } else if (TILING_KEY_IS(LARGE_INNER_GT_INNER_LT_OUTTER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SortedSparseSegmentMeanGradSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtLarge<DTYPE_X, DTYPE_SORTED_INDICES, DTYPE_PRE_LOCATION_INDICES, DTYPE_SEGMENT_IDS, uint64_t, uint32_t>(x, sorted_indices, segment_ids, output_dim0, pre_location_indices, y, userWs, tilingData);
    } else if (TILING_KEY_IS(LARGE_INNER_GT_INNER_GT_OUTTER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SortedSparseSegmentMeanGradSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtLarge<DTYPE_X, DTYPE_SORTED_INDICES, DTYPE_PRE_LOCATION_INDICES, DTYPE_SEGMENT_IDS, uint64_t, uint64_t>(x, sorted_indices, segment_ids, output_dim0, pre_location_indices, y, userWs, tilingData);
    } else if (TILING_KEY_IS(SMALL_INNER_LT_INNER_LT_OUTTER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SortedSparseSegmentMeanGradSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtSmall<DTYPE_X, DTYPE_SORTED_INDICES, DTYPE_PRE_LOCATION_INDICES, DTYPE_SEGMENT_IDS, uint32_t, uint32_t>(x, sorted_indices, segment_ids, output_dim0, pre_location_indices, y, userWs, tilingData);
    } else if (TILING_KEY_IS(SMALL_INNER_LT_INNER_GT_OUTTER_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SortedSparseSegmentMeanGradSimtTilingData, tilingData, tiling);
        invokeTemplateMeanSimtSmall<DTYPE_X, DTYPE_SORTED_INDICES, DTYPE_PRE_LOCATION_INDICES, DTYPE_SEGMENT_IDS, uint32_t, uint64_t>(x, sorted_indices, segment_ids, output_dim0, pre_location_indices, y, userWs, tilingData);
    }
}