/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_hash_table_lookup_or_insert_tiling_arch35.h
 * \brief embedding_hash_table_lookup_or_insert_tiling
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_H_
#pragma once

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
// ///////////////////////////////////
// tilingdata define
// ///////////////////////////////////
BEGIN_TILING_DATA_DEF(LookupOrInsertTilingData)
TILING_DATA_FIELD_DEF(int64_t, size);
TILING_DATA_FIELD_DEF(int64_t, embeddingDim);
TILING_DATA_FIELD_DEF(int64_t, filterFreq);
TILING_DATA_FIELD_DEF(int64_t, keyNum);
TILING_DATA_FIELD_DEF(uint32_t, filterMode);
TILING_DATA_FIELD_DEF(uint32_t, defaultKeyOrValue);
TILING_DATA_FIELD_DEF(int64_t, defaultKey);
TILING_DATA_FIELD_DEF(float, defaultValue);
TILING_DATA_FIELD_DEF(uint32_t, filterKeyFlag);
TILING_DATA_FIELD_DEF(int64_t, filterKey);
TILING_DATA_FIELD_DEF(uint32_t, threadXNum);
TILING_DATA_FIELD_DEF(uint32_t, threadYNum);
END_TILING_DATA_DEF;

// simt template ascendc tools
REGISTER_TILING_DATA_CLASS(EmbeddingHashTableLookupOrInsert, LookupOrInsertTilingData)

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_H_