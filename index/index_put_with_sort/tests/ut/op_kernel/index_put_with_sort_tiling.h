/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_with_sort_tiling.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_TILING_H
#define INDEX_PUT_WITH_SORT_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

#pragma pack(1)
struct IndexPutWithSortTilingData {
    uint32_t coreNums;
    uint64_t sliceSizeAligned;
    uint64_t sliceSize;
    uint32_t frontCoreNums;
    uint64_t frontCoreIndicesNums;
    uint64_t tailCoreIndicesNums;
    uint64_t frontBlocks;
    uint32_t frontBlockSize;
    uint32_t lastBlockSize;
    uint64_t frontCoreDataLength;
    uint64_t tailCoreDataLength;
    uint64_t indicesNums;
    uint32_t accumulate;
};
#pragma pack()

inline void InitIndexPutWithSortTilingData(uint8_t* tiling, IndexPutWithSortTilingData* data)
{
    memcpy(data, tiling, sizeof(IndexPutWithSortTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                               \
    IndexPutWithSortTilingData tiling_data;                                    \
    InitIndexPutWithSortTilingData(tiling_arg, &tiling_data)
#endif // INDEX_PUT_WITH_SORT_TILING_H