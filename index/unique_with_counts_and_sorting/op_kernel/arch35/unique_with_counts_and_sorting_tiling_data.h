/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_WITH_COUNTS_AND_SORTING_TILING_DATA_H
#define UNIQUE_WITH_COUNTS_AND_SORTING_TILING_DATA_H
#include "sort/unique_sort_tiling_data.h"
struct UniqueConsecutiveTilingData {
    int64_t totalSize;
    int64_t useCoreNums;
    int64_t tileLengthPerCore;
    int64_t tileLengthTailCore;
    int64_t adjUbTileLength;
    int64_t valueQueueSize;
    int64_t countQueueSize;
    int64_t idxQueueSize;
    int64_t collectingCntBufSize;
    int64_t offsetCntBufSize;
    int64_t prevIdxBufSize;
    int64_t shapeBufSize;
};
struct UniqueWithCountsAndSortingTilingData {
    UniqueSortRegBaseTilingData sort;
    UniqueConsecutiveTilingData unique;
    uint64_t valuesBytes;
    uint64_t indicesBytes;
    uint64_t uniqueSingleCore;
};
#endif
