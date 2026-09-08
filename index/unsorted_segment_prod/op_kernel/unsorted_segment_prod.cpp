/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unsorted_segment_prod.cpp
 * \brief unsorted_segment_prod kernel
 */

#include "../unsorted_segment_common/arch35/unsorted_segment_struct.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_prod_tiling_key.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_split_col.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_sort_simt.h"
#include "./unsorted_segment_prod.h"
#include "./unsorted_segment_prod_output_fullload.h"
#include "./unsorted_segment_prod_output_fullload_ws_merge.h"
#include "./unsorted_segment_prod_input_partition.h"
#include "./unsorted_segment_prod_segment_sort.h"

using namespace AscendC;
template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSegmentSortDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR workspace,
                                                 GM_ADDR tiling, TPipe& pipe)
{
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentProdSegmentSortTilingData, tilingData, tiling);
    if constexpr (sizeof(X_T) == 4) {
        UnsortedSegmentProd::InitGmProd<X_T, UnsortedSegmentProd::InitGmOneValue<X_T>>(
            output, tilingData.outputOuterDim * tilingData.innerDim);
        UnsortedSegmentProd::KernelProdSegmentSort<X_T, SEGMENT_IDS_T, int64_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    }
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelInputPartitionDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR workspace,
                                                    GM_ADDR tiling, TPipe& pipe)
{
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentProdInputPartTilingData, tilingData, tiling);
    if constexpr (!std::is_same<int64_t, X_T>::value && !std::is_same<uint64_t, X_T>::value) {
        GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
        UnsortedSegmentProd::KernelProdInputPartition<X_T, SEGMENT_IDS_T, UnsortedSegmentProd::GetInitOneValue<X_T>,
                                                      UnsortedSegmentProd::ComputeProd<X_T, uint64_t>>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output, userWs);
        op.Process();
    }
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSplitColDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling,
                                              TPipe& pipe)
{
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentSimdSplitColTilingData, tilingData, tiling);
    UnsortedSegment::KernelSimdSplitCol<X_T, SEGMENT_IDS_T, UnsortedSegmentProd::GetInitOneValue<X_T>,
                                        UnsortedSegmentProd::ComputeProd<X_T, uint64_t>>
        op(&tilingData, &pipe);
    op.Init(x, segment_ids, output);
    op.Process();
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSortSimtDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling,
                                              TPipe& pipe)
{
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentSortSimtTilingData, tilingData, tiling);
    UnsortedSegment::KernelUnsortedSegmentSortSimt<
        X_T, SEGMENT_IDS_T, UnsortedSegmentProd::SimtGatherProdValue<X_T>, UnsortedSegmentProd::SimtAtomicProd<X_T>,
        UnsortedSegmentProd::GetInitOneValue<X_T>, UnsortedSegmentProd::InitGmOneValue<X_T>>
        op(&tilingData, &pipe);
    op.Init(x, segment_ids, output);
    op.Process();
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelOutFlDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling, TPipe& pipe)
{
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentOutFlTilingData, tilingData, tiling);
    if constexpr (std::is_same<uint32_t, X_T>::value || std::is_same<uint64_t, X_T>::value ||
                  std::is_same<int64_t, X_T>::value) {
        return;
    } else {
        UnsortedSegmentProd::KernelUnsortedSegmentProdOutFl<
            X_T, SEGMENT_IDS_T, UnsortedSegmentProd::SimtGatherProdValue<X_T>,
            UnsortedSegmentProd::GetInitOneValue<X_T>, UnsortedSegmentProd::ComputeProd<X_T, uint64_t>>
            op(&pipe);
        op.Init(x, segment_ids, output, &tilingData);
        op.Process();
    }
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelOutFlWsMergeDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR workspace,
                                                  GM_ADDR tiling, TPipe& pipe)
{
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentOutFlTilingData, tilingData, tiling);
    if constexpr (std::is_same<uint32_t, X_T>::value || std::is_same<uint64_t, X_T>::value ||
                  std::is_same<int64_t, X_T>::value) {
        return;
    } else {
        GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
        UnsortedSegmentProd::KernelUnsortedSegmentProdOutFlWs<
            X_T, SEGMENT_IDS_T, UnsortedSegmentProd::SimtGatherProdValue<X_T>,
            UnsortedSegmentProd::GetInitOneValue<X_T>, UnsortedSegmentProd::ComputeProd<X_T, uint64_t>>
            op(&pipe);
        op.Init(x, segment_ids, output, userWs, &tilingData);
        op.Process();
    }
}

template <uint32_t TEMPLATE_MODE, uint32_t CAST_MODE>
__global__ __aicore__ void unsorted_segment_prod(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments, GM_ADDR output,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(UnsortedSegment::UnsortedSegmentSimtTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if constexpr (TEMPLATE_MODE == USS_TEMPLATE_OUT_FL) {
        KernelOutFlDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_OUT_FL_WS_MERGE) {
        KernelOutFlWsMergeDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, workspace, tiling, pipe);
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_INPUT_PART) {
        KernelInputPartitionDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, workspace, tiling, pipe);
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SEGMENT_SORT) {
        KernelSegmentSortDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, workspace, tiling, pipe);
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SIMD_SPLIT_COL) {
        KernelSplitColDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SORT_SIMT) {
        KernelSortSimtDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SIMT) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentSimtTilingData, tilingData, tiling);
        UnsortedSegmentProd::KernelUnsortedSegmentProd<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    }
}
