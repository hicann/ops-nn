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
 * \file unsorted_segment_sum_tiling_key.h
 * \brief ascend960dt/ascend960pr kernel 侧模板参数声明。
 *        与 op_host/arch35 共用的 tiling key 编码(ASCENDC_TPL_ARGS_DECL)保持一致，
 *        本文件仅声明 arch92 kernel 支持的模板组合(无 deterministic small innerdim 模板)。
 */

#ifndef UNSORTED_SEGMENT_SUM_TILING_KEY_H_
#define UNSORTED_SEGMENT_SUM_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define USS_TEMPLATE_SIMT 0
#define USS_TEMPLATE_DETERM 1
#define USS_TEMPLATE_OUT_FL 2
#define USS_TEMPLATE_SORT_SIMT 3
#define USS_TEMPLATE_SIMD_SPLIT_COL 4
#define USS_TEMPLATE_SIMD_NON_SORT 5
#define USS_TEMPLATE_SIMD_DYN_SORT 6
#define USS_TEMPLATE_DETERMINISTIC_BIG_INNERDIM 7
#define USS_TEMPLATE_DETERMINISTIC_SMALL_INNERDIM 8

#define USS_CAST_NONE 0
#define USS_CAST_INT32_TO_INT16 1
#define USS_CAST_INT64_TO_INT32 2
#define USS_CAST_INT64_TO_INT16 3
#define USS_CAST_INT32_TO_UINT8 4
#define USS_CAST_INT64_TO_UINT8 5

namespace UnsortedSegmentSum {
// ARGS_DECL 与 arch35 版本保持一致(模板参数的位宽与取值空间决定 tiling key 编码，两侧必须相同)
ASCENDC_TPL_ARGS_DECL(UnsortedSegmentSum,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 4, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMT,
                                            USS_TEMPLATE_DETERM, USS_TEMPLATE_OUT_FL, USS_TEMPLATE_SORT_SIMT,
                                            USS_TEMPLATE_SIMD_SPLIT_COL, USS_TEMPLATE_SIMD_NON_SORT,
                                            USS_TEMPLATE_SIMD_DYN_SORT, USS_TEMPLATE_DETERMINISTIC_BIG_INNERDIM,
                                            USS_TEMPLATE_DETERMINISTIC_SMALL_INNERDIM),
                      ASCENDC_TPL_UINT_DECL(CAST_MODE, 3, ASCENDC_TPL_UI_LIST, USS_CAST_NONE, USS_CAST_INT32_TO_INT16,
                                            USS_CAST_INT64_TO_INT32, USS_CAST_INT64_TO_INT16, USS_CAST_INT32_TO_UINT8,
                                            USS_CAST_INT64_TO_UINT8));

// arch92 kernel 不提供 USS_TEMPLATE_DETERMINISTIC_SMALL_INNERDIM 模板，该场景由 host 侧 tiling
// 在 960 上分发到 USS_TEMPLATE_DETERM
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumSimtTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_DETERM),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumDetermTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_OUT_FL),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumOutFlTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SORT_SIMT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE, USS_CAST_INT32_TO_INT16,
                                              USS_CAST_INT64_TO_INT32, USS_CAST_INT64_TO_INT16, USS_CAST_INT32_TO_UINT8,
                                              USS_CAST_INT64_TO_UINT8),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumSortSimtTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMD_SPLIT_COL),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumSimdSplitColTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMD_NON_SORT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumSimdNonSortTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMD_DYN_SORT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE, USS_CAST_INT32_TO_INT16,
                                              USS_CAST_INT64_TO_INT32, USS_CAST_INT64_TO_INT16, USS_CAST_INT32_TO_UINT8,
                                              USS_CAST_INT64_TO_UINT8),
                         ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumSimdDynSortTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_DETERMINISTIC_BIG_INNERDIM),
        ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
        ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegmentSum::UnsortedSegmentSumDeterministicBigInnerDimTilingData)));
} // namespace UnsortedSegmentSum

#endif // UNSORTED_SEGMENT_SUM_TILING_KEY_H_
