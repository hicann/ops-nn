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
 * \file unsorted_segment_min_apt.cpp
 * \brief unsorted_segment_min kernel
 */

#include "../unsorted_segment_common/arch35/unsorted_segment_simt.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_dyn_sort.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_non_sort.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_simd_split_col.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_out_full_load.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_sort_simt.h"
#include "../unsorted_segment_common/arch35/unsorted_segment_struct.h"

using namespace AscendC;
using namespace UnsortedSegment;

#define TEMPLATE_SIMT_TILING_KEY  1000
#define TEMPLATE_ADD_TILING_KEY  4000
#define TEMPLATE_SIMD_SPLIT_COL  5000
#define TEMPLATE_SIMD_NON_SORT  6000
#define TEMPLATE_SIMD_DYN_SORT  7000
#define TEMPLATE_SORT_SIMT 4100
constexpr uint8_t MODE_FLAGE = 0; // 0:unsorted_segment_min;

extern "C" __global__ __aicore__ void unsorted_segment_min(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(UnsortedSegmentSimtTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(TEMPLATE_SIMT_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1000", UnsortedSegmentSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimtTilingData, tilingData, tiling);
        UnsortedSegment::KernelUnsortedSegment<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_SPLIT_COL)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5000", UnsortedSegmentSimdSplitColTilingData);
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimdSplitColTilingData, tilingData, tiling);
        UnsortedSegment::KernelSimdSplitCol<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_NON_SORT)) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
            std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 6000", UnsortedSegmentSimdNonSortTilingData);
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimdNonSortTilingData, tilingData, tiling);
            UnsortedSegment::KernelSimdNonSort<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_DYN_SORT)) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
            std::is_same<int64_t, DTYPE_X>::value) {
            return;
        } else {
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 7000", UnsortedSegmentSimdDynSortTilingData);
            GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSimdDynSortTilingData, tilingData, tiling);
            UnsortedSegment::KernelSimdDynSort<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&tilingData, &pipe);
            op.Init(x, segment_ids, output);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_ADD_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4000", UnsortedSegmentOutFlTilingData);
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentOutFlTilingData, tilingData, tiling);
        if constexpr (
            std::is_same<uint32_t, DTYPE_X>::value || std::is_same<uint64_t, DTYPE_X>::value ||
            std::is_same<int64_t, DTYPE_X>::value) { // 空tensor处理
            UnsortedSegment::KernelUnsortedSegmentFL<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&pipe);
            op.Init(x, segment_ids, output, &tilingData);
        } else {
            UnsortedSegment::KernelUnsortedSegmentFL<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&pipe);
            op.Init(x, segment_ids, output, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_SORT_SIMT)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4100", UnsortedSegmentSortSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegmentSortSimtTilingData, tilingData, tiling);
        UnsortedSegment::KernelUnsortedSegmentSortSimt<DTYPE_X, DTYPE_SEGMENT_IDS, MODE_FLAGE> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    }
}