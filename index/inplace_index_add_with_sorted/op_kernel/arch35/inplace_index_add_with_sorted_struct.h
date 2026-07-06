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
 * \file inplace_index_add_with_sorted_struct.h
 * \brief A5 (ascend950) kernel-side tiling data struct
 */
#ifndef INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_STRUCT_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_STRUCT_H_

#include "kernel_tiling/kernel_tiling.h"

struct InplaceIndexAddWithSortedTilingData {
    int32_t usedCoreNum;
    int32_t enableAlpha;
    int64_t eachIndexCount;
    int64_t lastIndexCount;
    int64_t batchCount;
    int64_t eachBatchNum;
    int64_t lastBatchNum;
    int64_t inputCount;
    int64_t indicesCount;
    int64_t updatesCount;
    int64_t updatesOneTime;
    int64_t maxSize;
    int64_t eachNum;
    int64_t eachLoop;
    int64_t eachTail;
    int64_t varDimNum;
    int64_t eachUBIndexRound;
    int64_t eachUBIndexCount;
    int64_t eachUBIndexTail;
    int64_t lastUBIndexRound;
    int64_t lastUBIndexCount;
    int64_t lastUBIndexTail;
};

#endif // INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_STRUCT_H_
