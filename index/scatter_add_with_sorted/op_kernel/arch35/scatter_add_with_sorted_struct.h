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
 * \file scatter_add_with_sorted_struct.h
 * \brief tiling base data
 */

#ifndef SCATTER_ADD_WITH_SORTED_STRUCT_H
#define SCATTER_ADD_WITH_SORTED_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_SIMD 0
#define TPL_MODE_SIMT 1
#define TPL_MODE_EMPTY 2

#define TPL_SCALAR_FALSE 0
#define TPL_SCALAR_TRUE 1

#define TPL_DETERM_FALSE 0
#define TPL_DETERM_TRUE 1

#define TPL_ADDR_B32 0
#define TPL_ADDR_B64 1

ASCENDC_TPL_ARGS_DECL(
    ScatterAddWithSorted,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMD, TPL_MODE_SIMT, TPL_MODE_EMPTY),
    ASCENDC_TPL_UINT_DECL(IS_SCALAR, 1, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE, TPL_SCALAR_TRUE),
    ASCENDC_TPL_UINT_DECL(IS_DETERM, 1, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE, TPL_DETERM_TRUE),
    ASCENDC_TPL_UINT_DECL(ADDR_TYPE, 1, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32, TPL_ADDR_B64));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMD),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMD),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_TRUE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMD),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_TRUE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_TRUE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B64)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_TRUE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B64)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_TRUE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_TRUE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B64)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_EMPTY),
        ASCENDC_TPL_UINT_SEL(IS_SCALAR, ASCENDC_TPL_UI_LIST, TPL_SCALAR_FALSE),
        ASCENDC_TPL_UINT_SEL(IS_DETERM, ASCENDC_TPL_UI_LIST, TPL_DETERM_FALSE),
        ASCENDC_TPL_UINT_SEL(ADDR_TYPE, ASCENDC_TPL_UI_LIST, TPL_ADDR_B32)));

struct ScatterAddWithSortedSimtTilingData {
    int64_t varShape[2];
    int64_t indicesNum;
    int64_t normBlockIndices;
    int64_t tailBlockIndices;
    int64_t usedCoreNum;
    bool withPos;
    uint64_t tilingKey{0};
};
struct ScatterAddWithSortedSimdTilingData {
    int64_t needCoreNum{0};
    int64_t indicesNum{0};
    int64_t updatesInner{0};

    int64_t updatesBufferSize{0};
    int64_t outBufferSize{0};
    int64_t indicesBufferSize{0};
    int64_t posBufferSize{0};
    int64_t FrontAndBackIndexSize{0};

    int64_t normalCoreColNum{0};
    int64_t tailCoreColNum{0};
    int64_t normalCoreRowNum{0};
    int64_t tailCoreRowNum{0};

    int64_t normalCoreRowUbLoop{0};
    int64_t normalCoreNormalLoopRows{0};
    int64_t normalCoreTailLoopRows{0};
    int64_t tailCoreRowUbLoop{0};
    int64_t tailCoreNormalLoopRows{0};
    int64_t tailCoreTailLoopRows{0};

    int64_t normalCoreColUbLoop{0};
    int64_t normalCoreNormalLoopCols{0};
    int64_t normalCoreTailLoopCols{0};
    int64_t tailCoreColUbLoop{0};
    int64_t tailCoreNormalLoopCols{0};
    int64_t tailCoreTailLoopCols{0};
    int64_t coreNumInRow{0};
    int64_t coreNumInCol{0};

    int64_t vecAlignSize{0};
    int64_t indicesWorkspaceBufferSize{0};
    int64_t coreNumInColDeterm{0};
    int64_t tailCoreColUbDetermLoop{0};
    int64_t normalCoreColUbDetermLoop{0};
    int64_t tailCoreNormalLoopDetermCols{0};
    int64_t normalCoreNormalLoopDetermCols{0};
    int64_t tailCoreTailLoopDetermCols{0};
    int64_t normalCoreTailLoopDetermCols{0};
    int64_t updatesDeterminBufferSize{0};
    int64_t outBufferDeterminSize{0};
    int64_t normalCoreColDetermNum{0};
    int64_t tailCoreColNumDeterm{0};
    int64_t ubBlock{0};
    bool withPos{false};
    uint64_t tilingKey{0};
};
#endif