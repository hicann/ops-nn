/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef EMBEDDING_DENSE_GRAD_V2_TILING_DEF_H
#define EMBEDDING_DENSE_GRAD_V2_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct EmbeddingDenseGradV2TilingParam {
    uint32_t tailRowNum;
    uint32_t formerRowNum;
    uint32_t formerRowRepTime;
    uint32_t computeMask;
    uint32_t formerComputeRepTime;
    uint32_t formerComputeFormerNum;
    uint32_t formerComputeTailNum;
    uint32_t numWeights;
    uint32_t embeddingDim;
    uint32_t paddingIdx;
    uint32_t scaleGradByFreq;
    uint32_t formerDimRepTime;
    uint32_t formerEmbeddingDim;
    uint32_t tailEmbeddingDim;
    uint32_t tailComputeRepTime;
    uint32_t tailComputeFormerNum;
    uint32_t tailComputeTailNum;
};

struct EmbeddingDenseGradV2ScaleTiling {
    uint32_t tailCoreRowNum;
    uint32_t formerCoreRowNum;
    uint32_t formerCoreRowRepTime;
    uint32_t mask;
    uint32_t formerComputeRepTime;
    uint32_t formerComputeFormerNum;
    uint32_t formerComputeTailNum;
    uint32_t tailComputeRepTime;
    uint32_t tailComputeFormerNum;
    uint32_t tailComputeTailNum;
};

struct EmbeddingDenseGradV2SmallDimTiling {
    uint64_t partNum;
    uint64_t formerCopyRow;
    uint64_t tailCopyRow;
    uint64_t formerCopyTime;
    uint64_t tailCopyTime;
    uint64_t maxRowInUb;
    uint64_t formerLastRow;
    uint64_t tailLastRow;
};

struct EmbeddingDenseGradV2DeterminTiling {
    uint32_t gradRow;
    uint32_t tailRowNum;
    uint32_t formerRowNum;
    uint32_t formerRowNumRepTime;
    uint32_t computeMask;
    uint32_t formerComputeRepTime;
    uint32_t formerComputeFormerNum;
    uint32_t formerComputeTailNum;
    uint32_t tailComputeRepTime;
    uint32_t tailComputeFormerNum;
    uint32_t tailComputeTailNum;
};

struct EmbeddingDenseGradV2TilingData {
    EmbeddingDenseGradV2TilingParam params;
    EmbeddingDenseGradV2ScaleTiling scaleTiling;
    EmbeddingDenseGradV2DeterminTiling determinTiling;
    EmbeddingDenseGradV2SmallDimTiling smallDimTiling;
};

inline void InitEmbeddingDenseGradV2TilingData(uint8_t* tiling, EmbeddingDenseGradV2TilingData* data)
{
    memcpy(data, tiling, sizeof(EmbeddingDenseGradV2TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                               \
    EmbeddingDenseGradV2TilingData tiling_data;                                \
    InitEmbeddingDenseGradV2TilingData(tiling_arg, &tiling_data)
#endif // EMBEDDING_DENSE_GRAD_V2_TILING_DEF_H