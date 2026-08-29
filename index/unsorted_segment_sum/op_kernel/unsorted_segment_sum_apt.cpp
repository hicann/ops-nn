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

#include "arch35/unsorted_segment_sum_tiling_data.h"
#include "arch35/unsorted_segment_sum_tiling_key.h"
#include "arch35/unsorted_segment_sum.h"
#include "arch35/uss_deterministic.h"
#include "arch35/uss_simd_dyn_sort.h"
#include "arch35/uss_simd_non_sort.h"
#include "arch35/uss_simd_split_col.h"
#include "arch35/unsorted_segment_add.h"
#include "arch35/unsorted_segment_sort_simt.h"
#include "arch35/uss_deterministic_big_innerdim.h"
#include "arch35/uss_deterministic_small_innerdim.h"

using namespace AscendC;
using namespace UnsortedSegmentSum;

template <uint32_t TEMPLATE_MODE, uint32_t CAST_MODE>
__global__ __aicore__ void unsorted_segment_sum(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments, GM_ADDR output,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(UnsortedSegmentSumSimtTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SIMT) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimtTilingData, tilingData, tiling);
        UnsortedSegmentSum::KernelUnsortedSegmentSum<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SIMD_SPLIT_COL) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimdSplitColTilingData, tilingData, tiling);
        UnsortedSegmentSum::USSKernelSimdSplitCol<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SIMD_NON_SORT) {
        if constexpr (std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                      std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimdNonSortTilingData, tilingData, tiling);
            UnsortedSegmentSum::USSKernelSimdNonSort<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SIMD_DYN_SORT) {
        if constexpr (std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                      std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSimdDynSortTilingData, tilingData, tiling);
            if constexpr (CAST_MODE == USS_CAST_NONE) {
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, DTYPE_SEGMENT_IDS, CAST_NO> op(
                    &tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if constexpr (CAST_MODE == USS_CAST_INT32_TO_INT16) {
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT32_2_INT16> op(
                    &tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if constexpr (CAST_MODE == USS_CAST_INT64_TO_INT32) {
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, int32_t, CAST_INT64_2_INT32> op(
                    &tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if constexpr (CAST_MODE == USS_CAST_INT64_TO_INT16) {
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT64_2_INT16> op(
                    &tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if constexpr (CAST_MODE == USS_CAST_INT32_TO_UINT8) {
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT32_2_UINT8> op(
                    &tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            } else if constexpr (CAST_MODE == USS_CAST_INT64_TO_UINT8) {
                UnsortedSegmentSum::USSKernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT64_2_UINT8> op(
                    &tilingData, &pipe);
                op.Init(x, segment_ids, output);
                op.Process();
            }
        }
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_DETERM) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermTilingData, tilingData, tiling);
        KernelUSSDeterministic<DTYPE_X, DTYPE_SEGMENT_IDS> op(tilingData, pipe);
        op.Init(x, segment_ids, output, workspace);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_OUT_FL) {
        if constexpr (std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
                      std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumOutFlTilingData, tilingData, tiling);
            UnsortedSegmentSum::KernelUnsortedSegmentAddSum<DTYPE_X, DTYPE_SEGMENT_IDS> op(&pipe);
            op.Init(x, segment_ids, output, &tilingData);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_SORT_SIMT) {
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumSortSimtTilingData, tilingData, tiling);
        if constexpr (CAST_MODE == USS_CAST_NONE) {
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, DTYPE_SEGMENT_IDS, CAST_NO>
                op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if constexpr (CAST_MODE == USS_CAST_INT32_TO_INT16) {
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT32_2_INT16>
                op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if constexpr (CAST_MODE == USS_CAST_INT64_TO_INT32) {
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, int32_t, CAST_INT64_2_INT32>
                op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if constexpr (CAST_MODE == USS_CAST_INT64_TO_INT16) {
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, int16_t, CAST_INT64_2_INT16>
                op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if constexpr (CAST_MODE == USS_CAST_INT32_TO_UINT8) {
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT32_2_UINT8>
                op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        } else if constexpr (CAST_MODE == USS_CAST_INT64_TO_UINT8) {
            UnsortedSegmentSum::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, uint8_t, CAST_INT64_2_UINT8>
                op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_DETERMINISTIC_BIG_INNERDIM) {
        if constexpr (std::is_same<float, DTYPE_X>::value || std::is_same<half, DTYPE_X>::value ||
                      std::is_same<bfloat16_t, DTYPE_X>::value) {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDeterministicBigInnerDimTilingData, tilingData, tiling);
            UnsortedSegmentSum::USSKernelDeterministicBigInnerDim<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == USS_TEMPLATE_DETERMINISTIC_SMALL_INNERDIM) {
        if constexpr (std::is_same<float, DTYPE_X>::value || std::is_same<half, DTYPE_X>::value ||
                      std::is_same<bfloat16_t, DTYPE_X>::value) {
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSumDetermSmallInnerDimTilingData, tilingData, tiling);
            UnsortedSegmentSum::USSKernelDeterministicSmallInnerDim<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    }
}
