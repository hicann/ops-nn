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
 * \file index_put_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _INDEX_PUT_V2_TILING_DATA_H_
#define _INDEX_PUT_V2_TILING_DATA_H_

#include <cstdint>

namespace IndexPutV2 {

struct IndexPutV2SimdTilingData {
    int64_t inputLength = 0;
    int64_t valueLength = 0;
    int64_t inputDimNum = 0;
    int64_t indexedSizesNum = 0;
    int64_t indexedDimNum = 0;
    int64_t nonIndexedDimNum = 0;
    int64_t accumulateMode = 0;
    int64_t indexedLength = 0;
    int64_t nonIndexedLength = 0;
    int64_t normalCoreRowsNum = 0;
    int64_t normalCoreColsNum = 0;
    int64_t tailCoreRowsNum = 0;
    int64_t tailCoreColsNum = 0;
    int64_t blockNumInRow = 0;
    int64_t blockNumInCol = 0;
    int64_t rowsFactor = 0;
    int64_t colsFactor = 0;
    int64_t coreNum = 0;
    int64_t inputShapes[8] = {0};
    int64_t indexedStrides[8] = {0};
};

} // namespace IndexPutV2

#endif // _INDEX_PUT_V2_TILING_DATA_H_
