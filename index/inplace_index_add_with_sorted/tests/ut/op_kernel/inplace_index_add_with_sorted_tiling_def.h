/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_add_with_sorted_tiling.h
 * \brief
 */

#ifndef _FAST_OP_TEST_INPLACE_INDEX_ADD_WITH_SORTED_TILING_H_
#define _FAST_OP_TEST_INPLACE_INDEX_ADD_WITH_SORTED_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)

struct InplaceIndexAddWithSortedTilingData {
    uint32_t usedCoreNum = 1;
    uint32_t enableAlpha = 1;
    uint32_t eachIndexCount = 1;
    uint32_t lastIndexCount = 1;
    uint32_t batchCount = 1;
    uint32_t eachBatchNum = 1;
    uint32_t lastBatchNum = 1;
    uint32_t inputCount = 1;
    uint32_t indicesCount = 1;
    uint32_t updatesCount = 1;
    uint32_t updatesOneTime = 1;
    uint32_t maxSize = 1;
    uint32_t eachNum = 1;
    uint32_t eachLoop = 1;
    uint32_t eachTail = 0;
    uint32_t varDimNum = 1;
    uint32_t eachUBIndexRound = 1;
    uint32_t eachUBIndexCount = 1;
    uint32_t eachUBIndexTail = 1;
    uint32_t lastUBIndexRound = 1;
    uint32_t lastUBIndexCount = 1;
    uint32_t lastUBIndexTail = 1;
};

#pragma pack()

inline void InitInplaceIndexAddWithSortedTilingData(uint8_t* tiling, InplaceIndexAddWithSortedTilingData* constData)
{
    memcpy(constData, tiling, sizeof(InplaceIndexAddWithSortedTilingData));
}

#undef GET_TILING_DATA
#define GET_TILING_DATA(tilingData, tilingArg)      \
    InplaceIndexAddWithSortedTilingData tilingData; \
    InitInplaceIndexAddWithSortedTilingData(tilingArg, &tilingData)

#endif // _FAST_OP_TEST_INPLACE_INDEX_ADD_WITH_SORTED_TILING_H_
