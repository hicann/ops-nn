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
 * \file unsorted_segment_sum_apt.cpp
 * \brief unsorted_segment_sum kernel
 */

/*!
 * \file unsorted_segment_sum_apt.cpp
 * \brief
 */

#include "arch35/unsorted_segment_sum.h"
#include "arch35/uss_deterministic.h"
#include "arch35/uss_simd_dyn_sort.h"
#include "arch35/uss_simd_non_sort.h"
#include "arch35/uss_simd_split_col.h"
#include "arch35/unsorted_segment_add.h"
#include "arch35/unsorted_segment_sort_simt.h"
#include "arch35/uss_deterministic_big_innerdim.h"

using namespace AscendC;
using namespace UnsortedSegmentSum;

#define TEMPLATE_SIMT_TILING_KEY  1000
#define TEMPLATE_ADD_TILING_KEY  4000
#define BF16_INT32_TILING_KEY  3102
#define FP16_INT32_TILING_KEY  3101
#define FP32_INT32_TILING_KEY  3100
#define BF16_INT64_TILING_KEY  3202
#define FP16_INT64_TILING_KEY  3201
#define FP32_INT64_TILING_KEY  3200
#define TEMPLATE_SIMD_SPLIT_COL  5000
#define TEMPLATE_SIMD_NON_SORT  6000
#define TEMPLATE_SIMD_DYN_SORT  7000
#define TEMPLATE_SORT_SIMT 4100
#define TEMPLATE_DETERMINISTIC_BIG_INNERDIM 8000

extern "C" __global__ __aicore__ void unsorted_segment_sum(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(TEMPLATE_SIMT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimtTilingData, tilingData, tiling);
        UnsortedSegmentSum::KernelUnsortedSegmentSum<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_SPLIT_COL)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimdSplitColTilingData, tilingData, tiling);
        UnsortedSegmentSum::USSKernelSimdSplitCol<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_NON_SORT)) {
        if constexpr (
                    std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                    std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimdNonSortTilingData, tilingData, tiling);
            UnsortedSegmentSum::USSKernelSimdNonSort<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_DYN_SORT)) {
        if constexpr (
                    std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                    std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimdDynSortTilingData, tilingData, tiling);
            if (tilingData.indicesCastMode == CAST_NO){
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, DTYPE_SEGMENT_IDS, CAST_NO> op(&tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if (tilingData.indicesCastMode == CAST_INT32_2_INT16){
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT32_2_INT16> op(&tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if (tilingData.indicesCastMode == CAST_INT64_2_INT32){
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, int32_t, CAST_INT64_2_INT32> op(&tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if (tilingData.indicesCastMode == CAST_INT64_2_INT16){
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT64_2_INT16> op(&tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if (tilingData.indicesCastMode == CAST_INT32_2_UINT8){
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT32_2_UINT8> op(&tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if (tilingData.indicesCastMode == CAST_INT64_2_UINT8){
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT64_2_UINT8> op(&tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(FP32_INT32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<float, int32_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if (TILING_KEY_IS(FP16_INT32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<half, int32_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if (TILING_KEY_IS(BF16_INT32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<bfloat16_t, int32_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if (TILING_KEY_IS(FP32_INT64_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<float, int64_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if (TILING_KEY_IS(FP16_INT64_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<half, int64_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if (TILING_KEY_IS(BF16_INT64_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<bfloat16_t, int64_t> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if (TILING_KEY_IS(TEMPLATE_ADD_TILING_KEY)) {
        if constexpr (
                    std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                    std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumOutFlTilingData, tilingData, tiling);
            UnsortedSegmentSum::KernelUnsortedSegmentAddSum<DTYPE_X, DTYPE_SEGMENT_IDS> op(&pipe);
            op.Init(x, segment_ids, output, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_SORT_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSortSimtTilingData, tilingData, tiling);
        if (tilingData.indicesCastMode == CAST_NO){
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, DTYPE_SEGMENT_IDS, CAST_NO> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_INT32_2_INT16){
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT32_2_INT16> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_INT64_2_INT32){
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, int32_t, CAST_INT64_2_INT32> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_INT64_2_INT16){
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT64_2_INT16> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_INT32_2_UINT8){
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT32_2_UINT8> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_INT64_2_UINT8){
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT64_2_UINT8> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_DETERMINISTIC_BIG_INNERDIM)) {
        if constexpr (
                    std::is_same<float, DTYPE_X>::value || std::is_same<half, DTYPE_X>::value ||
                    std::is_same<bfloat16_t, DTYPE_X>::value) {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDeterministicBigInnerDimTilingData, tilingData, tiling);
            UnsortedSegmentSum::USSKernelDeterministicBigInnerDim<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    }
}