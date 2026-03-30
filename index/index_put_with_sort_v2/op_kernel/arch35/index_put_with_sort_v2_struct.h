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
* \file index_put_with_sort_v2_struct.h
* \brief
*/
#ifndef INDEX_PUT_WITH_SORT_V2_STRUCT_H
#define INDEX_PUT_WITH_SORT_V2_STRUCT_H
#include "ascendc/host_api/tiling/template_argument.h"

static constexpr size_t MAX_DIM_NUM = 8;
class IndexPutWithSortV2SimdTilingData {
public:
    int64_t nonIndexedDimNum = 0;
    int64_t indexedDimSize = 0;
    int64_t nonIndexedDimSize = 0;
    int64_t indicesFactor = 0;
    int64_t ubFactor = 0;
    int64_t rowBlockFactor = 0;
    int64_t rowUseCoreNum = 0;
    int64_t rowTailBlockFactor = 0;
    int64_t colBlockFactor = 0;
    int64_t colUseCoreNum = 0;
    int64_t colTailBlockFactor = 0;
};

class IndexPutWithSortV2TilingData {
public:
int64_t nonIndexedDimNum = 0;
int64_t indexedDimSize = 0;
int64_t nonIndexedDimSize = 0;
int64_t nonIdxedStride[MAX_DIM_NUM] = {0};
int64_t nonIdxedSelfStride[MAX_DIM_NUM] = {0};
int64_t nonIdxedValueStride[MAX_DIM_NUM] = {0};
int64_t idxedValueStride = 0;
int64_t indexedThreadNum = 0;
int64_t nonIndexedThreadNum = 0;
};

ASCENDC_TPL_ARGS_DECL(IndexPutWithSortV2,
    ASCENDC_TPL_BOOL_DECL(ACCUMULATE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(ALL_INDEXED, 0, 1),
    ASCENDC_TPL_BOOL_DECL(INDEXED_BLOCK_MODE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_CAST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(KERNEL_SIMD, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(ACCUMULATE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(ALL_INDEXED, 0, 1),
        ASCENDC_TPL_BOOL_SEL(INDEXED_BLOCK_MODE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CAST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(KERNEL_SIMD, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(IndexPutWithSortV2TilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(ACCUMULATE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(ALL_INDEXED, 0, 1),
        ASCENDC_TPL_BOOL_SEL(INDEXED_BLOCK_MODE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CAST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(KERNEL_SIMD, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(IndexPutWithSortV2SimdTilingData)
    ),
)

#endif // INDEX_PUT_WITH_SORT_V2_STRUCT_H