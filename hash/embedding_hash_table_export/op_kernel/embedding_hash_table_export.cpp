/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file embedding_hash_table_export.cpp
 * \brief
 */

#include "./arch35/embedding_hash_table_export.h"

using namespace EmbeddingHashTableExportAicore;

#define BIT4WIDTH_TILING_KEY 6
extern "C" __global__ __aicore__ void embedding_hash_table_export(GM_ADDR table_handles, GM_ADDR table_sizes,
    GM_ADDR embedding_dims, GM_ADDR bucket_sizes, GM_ADDR keys, GM_ADDR counters, GM_ADDR filter_flags, GM_ADDR values,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(BIT4WIDTH_TILING_KEY)) {
        EmbeddingHashTableExport<float> op;
        op.Init(table_handles, table_sizes, embedding_dims, bucket_sizes, keys, counters, filter_flags, values, userWS,
            tilingData);
        op.Process();
    }
}
