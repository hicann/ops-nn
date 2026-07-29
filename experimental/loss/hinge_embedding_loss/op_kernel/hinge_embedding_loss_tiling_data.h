/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HINGE_EMBEDDING_LOSS_TILING_DATA_H_
#define HINGE_EMBEDDING_LOSS_TILING_DATA_H_
#include <cstdint>
struct HingeEmbeddingLossTilingData {
    uint32_t smallCoreDataNum;       // Logical elements assigned to a normal core.
    uint32_t bigCoreDataNum;         // Logical elements assigned to a core with one extra element.
    uint32_t finalBigTileNum;        // Tile count on a core using bigCoreDataNum.
    uint32_t finalSmallTileNum;      // Tile count on a core using smallCoreDataNum.
    uint32_t tileDataNum;            // Logical elements in every non-final tile.
    uint32_t smallTailDataNum;       // Logical elements in the final small-core tile.
    uint32_t bigTailDataNum;         // Logical elements in the final big-core tile.
    uint32_t tailBlockNum;           // Number of leading cores using bigCoreDataNum.
    uint32_t blockNum;               // Number of vector cores launched.
    uint32_t workspaceFloatsPerCore; // FLOAT slots reserved for each core's partial sum.
    float margin;                    // Margin from the public operator attribute.
    float meanScale;                 // 1 / totalElements for mean reduction.
};
#endif
