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
 * \file init_embedding_hash_table_tiling_arch35.h
 * \brief Define InitEmbeddingHashTable tiling data
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INIT_EMBEDDING_HASH_TABLE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INIT_EMBEDDING_HASH_TABLE_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"


namespace optiling {
struct InitEmbeddingHashTableCompileInfo {
    int64_t coreNum{0};
    int64_t maxThread{512};
};

BEGIN_TILING_DATA_DEF(InitEmbeddingHashTableTilingData)
TILING_DATA_FIELD_DEF(int64_t, bucketSize);
TILING_DATA_FIELD_DEF(int64_t, embeddingDim);
TILING_DATA_FIELD_DEF(int64_t, initializerMode);
TILING_DATA_FIELD_DEF(float, constantValue);
TILING_DATA_FIELD_DEF(int64_t, bucketLength);
TILING_DATA_FIELD_DEF(int64_t, useThreadDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InitEmbeddingHashTable, InitEmbeddingHashTableTilingData)

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_INIT_EMBEDDING_HASH_TABLE_TILING_H