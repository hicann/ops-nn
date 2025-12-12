/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef EMBEDDING_BAG_TILING_DEF_H
#define EMBEDDING_BAG_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct EmbeddingBagTilingData {
    int64_t formerOffsetNum;
    int64_t tailOffsetNum;
    int64_t numEmbeddings;
    int64_t computeRepTime;
    int64_t indicesMaxMoveLength;
    int64_t usedCoreNum;
    int64_t paddingIdx;
    int64_t numIndices;
    int64_t mode;
    int64_t tilingKey;
    int64_t hasPerSampleWeights;
};

inline void InitEmbeddingBagTilingData(uint8_t* tiling, EmbeddingBagTilingData* data)
{
    memcpy(data, tiling, sizeof(EmbeddingBagTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    EmbeddingBagTilingData tiling_data;          \
    InitEmbeddingBagTilingData(tiling_arg, &tiling_data)

#endif // EMBEDDING_BAG_TILING_DEF_H