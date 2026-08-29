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
 * \file unsorted_segment_max_tiling_key.h
 * \brief unsorted_segment_max_tiling_key
 */

#ifndef UNSORTED_SEGMENT_MAX_TILING_KEY_H_
#define UNSORTED_SEGMENT_MAX_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define USS_TEMPLATE_SIMT 0
#define USS_TEMPLATE_SORT_SIMT 1
#define USS_TEMPLATE_OUT_FL 2
#define USS_TEMPLATE_SIMD_SPLIT_COL 3
#define USS_TEMPLATE_SIMD_NON_SORT 4
#define USS_TEMPLATE_SIMD_DYN_SORT 5

#define USS_CAST_NONE 0
#define USS_CAST_INT32_TO_INT16 1
#define USS_CAST_INT64_TO_INT32 2
#define USS_CAST_INT64_TO_INT16 3
#define USS_CAST_INT32_TO_UINT8 4
#define USS_CAST_INT64_TO_UINT8 5

namespace UnsortedSegmentMax {
ASCENDC_TPL_ARGS_DECL(UnsortedSegmentMax,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 3, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMT,
                                            USS_TEMPLATE_SORT_SIMT, USS_TEMPLATE_OUT_FL, USS_TEMPLATE_SIMD_SPLIT_COL,
                                            USS_TEMPLATE_SIMD_NON_SORT, USS_TEMPLATE_SIMD_DYN_SORT),
                      ASCENDC_TPL_UINT_DECL(CAST_MODE, 3, ASCENDC_TPL_UI_LIST, USS_CAST_NONE, USS_CAST_INT32_TO_INT16,
                                            USS_CAST_INT64_TO_INT32, USS_CAST_INT64_TO_INT16, USS_CAST_INT32_TO_UINT8,
                                            USS_CAST_INT64_TO_UINT8));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentSimtTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SORT_SIMT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentSortSimtTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_OUT_FL),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentOutFlTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMD_SPLIT_COL),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentSimdSplitColTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMD_NON_SORT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentSimdNonSortTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMD_DYN_SORT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE, USS_CAST_INT32_TO_INT16,
                                              USS_CAST_INT64_TO_INT32, USS_CAST_INT64_TO_INT16, USS_CAST_INT32_TO_UINT8,
                                              USS_CAST_INT64_TO_UINT8),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentSimdDynSortTilingData)));
} // namespace UnsortedSegmentMax

#endif // UNSORTED_SEGMENT_MAX_TILING_KEY_H_
