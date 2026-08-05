/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEEP_NORM_GRAD_TILING_DATA_ARCH35_H
#define DEEP_NORM_GRAD_TILING_DATA_ARCH35_H

#include <cstdint>

struct DeepNormGradTilingDataArch35 {
    uint64_t numRows = 0;
    uint64_t numCols = 0;
    uint64_t rowsPerCore = 0;
    uint64_t colsPerCore = 0;
    uint32_t backwardBlockDim = 0;
    uint32_t gammaBetaBlockDim = 0;
    uint32_t tileLength = 0;
    uint32_t tileLengthAlign = 0;
    float alpha = 0.0f;
    float invCols = 0.0f;
    uint32_t gammaBetaRowSplit = 0;
    uint32_t smallRowStride = 0;
    uint32_t smallRowsPerTile = 0;
    uint32_t smallColsAlign = 0;
};

#endif // DEEP_NORM_GRAD_TILING_DATA_ARCH35_H
