/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_hash_table_evict_tiling_arch35.h
 * \brief embedding_hash_table_evict_tiling
 */

#ifndef EMBEDDING_HASH_TABLE_EVICT_TILING_ARCH35_H_
#define EMBEDDING_HASH_TABLE_EVICT_TILING_ARCH35_H_
#pragma once

#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EvictTilingData)
TILING_DATA_FIELD_DEF(int64_t, tableCap);
TILING_DATA_FIELD_DEF(int64_t, embeddingDim);
TILING_DATA_FIELD_DEF(int64_t, initMode);
TILING_DATA_FIELD_DEF(float, constVal);
TILING_DATA_FIELD_DEF(int64_t, keyNum);
TILING_DATA_FIELD_DEF(uint32_t, usedThreadNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingHashTableEvict, EvictTilingData)

struct EvictCompileInfo {
    uint32_t maxThreadNum;
    uint32_t coreNumAiv;
};
} // namespace optiling

#endif // EMBEDDING_HASH_TABLE_EVICT_TILING_ARCH35_H_
