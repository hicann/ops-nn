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
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexPutWithSortTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreNums);
    TILING_DATA_FIELD_DEF(uint64_t, sliceSizeAligned);
    TILING_DATA_FIELD_DEF(uint64_t, sliceSize);
    TILING_DATA_FIELD_DEF(uint32_t, frontCoreNums);
    TILING_DATA_FIELD_DEF(uint64_t, frontCoreIndicesNums);
    TILING_DATA_FIELD_DEF(uint64_t, tailCoreIndicesNums);
    TILING_DATA_FIELD_DEF(uint64_t, frontBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, frontBlockSize);
    TILING_DATA_FIELD_DEF(uint32_t, lastBlockSize);
    TILING_DATA_FIELD_DEF(uint64_t, frontCoreDataLength);
    TILING_DATA_FIELD_DEF(uint64_t, tailCoreDataLength);
    TILING_DATA_FIELD_DEF(uint64_t, indicesNums);
    TILING_DATA_FIELD_DEF(uint32_t, accumulate);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(IndexPutWithSort, IndexPutWithSortTilingData)

struct IndexPutWithSortCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
};
}
#endif // INDEX_PUT_WITH_SORT_TILING_H