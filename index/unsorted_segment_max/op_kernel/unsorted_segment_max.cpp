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
 * \file unsorted_segment_max.cpp
 * \brief unsorted_segment_max kernel
 */

#include "../unsorted_segment_common/arch35/unsorted_segment_simt.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_dyn_sort.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_non_sort.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_split_col.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_out_full_load.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_sort_simt.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_struct.h"
#include "./unsorted_segment_max.h"

using namespace UnsortedSegment;
using namespace AscendC;

#define TEMPLATE_SIMT_TILING_KEY 1000
#define TEMPLATE_ADD_TILING_KEY 4000
#define TEMPLATE_SORT_SIMT 4100
#define TEMPLATE_SIMD_NON_SORT 6000
#define TEMPLATE_SIMD_DYN_SORT 7000
#define TEMPLATE_SIMD_SPLIT_COL 5000

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSimdDynSortWithCast(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments, GM_ADDR output,
                                                 GM_ADDR workspace, GM_ADDR tiling, TPipe& pipe)
{
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 7000", UnsortedSegmentSimdDynSortTilingData);
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimdDynSortTilingData, tilingData, tiling);
    uint32_t cast_mode = tilingData.idCastMode;
    if (cast_mode == CAST_INT32_TO_INT16) {
        UnsortedSegment::KernelSimdDynSort<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, InitGmMinValue<X_T>,
                                           ComputeMax<X_T, uint32_t>, SetAtomicMaxOp<X_T>, CAST_INT32_TO_INT16>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (cast_mode == CAST_INT64_TO_INT32) {
        UnsortedSegment::KernelSimdDynSort<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, InitGmMinValue<X_T>,
                                           ComputeMax<X_T, uint32_t>, SetAtomicMaxOp<X_T>, CAST_INT64_TO_INT32>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (cast_mode == CAST_INT64_TO_INT16) {
        UnsortedSegment::KernelSimdDynSort<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, InitGmMinValue<X_T>,
                                           ComputeMax<X_T, uint32_t>, SetAtomicMaxOp<X_T>, CAST_INT64_TO_INT16>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (cast_mode == CAST_INT32_TO_UINT8) {
        UnsortedSegment::KernelSimdDynSort<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, InitGmMinValue<X_T>,
                                           ComputeMax<X_T, uint32_t>, SetAtomicMaxOp<X_T>, CAST_INT32_TO_UINT8>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (cast_mode == CAST_INT64_TO_UINT8) {
        UnsortedSegment::KernelSimdDynSort<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, InitGmMinValue<X_T>,
                                           ComputeMax<X_T, uint32_t>, SetAtomicMaxOp<X_T>, CAST_INT64_TO_UINT8>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else {
        UnsortedSegment::KernelSimdDynSort<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, InitGmMinValue<X_T>,
                                           ComputeMax<X_T, uint32_t>, SetAtomicMaxOp<X_T>, CAST_NONE>
            op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    }
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSimtDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling, TPipe& pipe)
{
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1000", UnsortedSegmentSimtTilingData);
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimtTilingData, tilingData, tiling);
    UnsortedSegment::KernelUnsortedSegment<X_T, SEGMENT_IDS_T, SimtComputeSegmentMax<X_T>, InitGmMinValue<X_T>> op(
        &tilingData, &pipe);
    op.Init(x, segment_ids, output);
    op.Process();
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSplitColDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling,
                                              TPipe& pipe)
{
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5000", UnsortedSegmentSimdSplitColTilingData);
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimdSplitColTilingData, tilingData, tiling);
    UnsortedSegment::KernelSimdSplitCol<X_T, SEGMENT_IDS_T, GetInitMinValue<X_T>, ComputeMax<X_T, uint64_t>> op(
        &tilingData, &pipe);
    op.Init(x, segment_ids, output);
    op.Process();
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelNonSortDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling,
                                             TPipe& pipe)
{
    if constexpr (std::is_same<uint32_t, X_T>::value || std::is_same<uint64_t, X_T>::value ||
                  std::is_same<int64_t, X_T>::value) {
        return;
    } else {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 6000", UnsortedSegmentSimdNonSortTilingData);
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimdNonSortTilingData, tilingData, tiling);
        UnsortedSegment::KernelSimdNonSort<X_T, SEGMENT_IDS_T, InitGmMinValue<X_T>, SetAtomicMaxOp<X_T>> op(&tilingData,
                                                                                                            &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    }
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelOutFlDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling, TPipe& pipe)
{
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4000", UnsortedSegmentOutFlTilingData);
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentOutFlTilingData, tilingData, tiling);
    if constexpr (std::is_same<uint32_t, X_T>::value || std::is_same<uint64_t, X_T>::value ||
                  std::is_same<int64_t, X_T>::value) {
        UnsortedSegment::KernelUnsortedSegmentFL<X_T, SEGMENT_IDS_T, SimtGatherMaxValue<X_T>, GetInitMinValue<X_T>,
                                                 InitGmMinValue<X_T>, ComputeMax<X_T, int32_t>, SetAtomicMaxOp<X_T>>
            op(&pipe);
        op.Init(x, segment_ids, output, &tilingData);
    } else {
        UnsortedSegment::KernelUnsortedSegmentFL<X_T, SEGMENT_IDS_T, SimtGatherMaxValue<X_T>, GetInitMinValue<X_T>,
                                                 InitGmMinValue<X_T>, ComputeMax<X_T, int32_t>, SetAtomicMaxOp<X_T>>
            op(&pipe);
        op.Init(x, segment_ids, output, &tilingData);
        op.Process();
    }
}

template <typename X_T, typename SEGMENT_IDS_T>
__aicore__ inline void KernelSortSimtDispatch(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR output, GM_ADDR tiling,
                                              TPipe& pipe)
{
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4100", UnsortedSegmentSortSimtTilingData);
    GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSortSimtTilingData, tilingData, tiling);
    UnsortedSegment::KernelUnsortedSegmentSortSimt<X_T, SEGMENT_IDS_T, SimtGatherMaxValue<X_T>,
                                                   SimtComputeSegmentMax<X_T>, GetInitMinValue<X_T>,
                                                   InitGmMinValue<X_T>>
        op(&tilingData, &pipe);
    op.Init(x, segment_ids, output);
    op.Process();
}

extern "C" __global__ __aicore__ void unsorted_segment_max(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments,
                                                           GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(UnsortedSegmentSimtTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(TEMPLATE_SIMT_TILING_KEY)) {
        KernelSimtDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_SPLIT_COL)) {
        KernelSplitColDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_NON_SORT)) {
        KernelNonSortDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_DYN_SORT)) {
        if constexpr (std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                      std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            KernelSimdDynSortWithCast<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, num_segments, output, workspace,
                                                                  tiling, pipe);
        }
    } else if (TILING_KEY_IS(TEMPLATE_ADD_TILING_KEY)) {
        KernelOutFlDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    } else if (TILING_KEY_IS(TEMPLATE_SORT_SIMT)) {
        KernelSortSimtDispatch<DTYPE_X, DTYPE_SEGMENT_IDS>(x, segment_ids, output, tiling, pipe);
    }
}