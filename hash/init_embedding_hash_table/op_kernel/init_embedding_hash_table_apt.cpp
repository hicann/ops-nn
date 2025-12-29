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
 * \file init_embedding_hash_table.cpp
 * \brief Ascendc InitEmbeddingHashTable kernel Interface
 */

#include "./arch35/init_embedding_hash_table.h"

extern "C" __global__ __aicore__ void init_embedding_hash_table(GM_ADDR tableHandle, GM_ADDR sampledValues,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    InitEmbeddingHashTable::KernelInitEmbeddingHashTable<int64_t, float> op;
    op.Init(tableHandle, sampledValues, tilingData.bucketSize, tilingData.embeddingDim, tilingData.initializerMode,
            tilingData.constantValue, tilingData.bucketLength, tilingData.useThreadDim);
    op.Process();
}
