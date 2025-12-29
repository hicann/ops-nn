/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define FP32_TILING_KEY 100

#include "./arch35/embedding_hash_table_import.h"

using namespace EmbeddingHashTable;

extern "C" __global__ __aicore__ void embedding_hash_table_import(
    GM_ADDR tableHandles, GM_ADDR embeddingDims, GM_ADDR bucketSizes, GM_ADDR keys,
    GM_ADDR counters, GM_ADDR filterFlags, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(FP32_TILING_KEY)) {
        EmbeddingHashTable::EmbeddingHashTableImport<float> op;
        op.Init(tableHandles, embeddingDims, bucketSizes, keys, counters, filterFlags, values, workspace, &tilingData);
        op.Process();
    }
}
