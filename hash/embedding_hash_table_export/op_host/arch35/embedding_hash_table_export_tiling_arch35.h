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
 * \file embedding_hash_table_export_tiling_arch35.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_EXPORT_H_
#define EMBEDDING_HASH_TABLE_EXPORT_H_
#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingHashTableExportTilingData)
TILING_DATA_FIELD_DEF(int64_t, tableNum);
TILING_DATA_FIELD_DEF(int64_t, exportMode);
TILING_DATA_FIELD_DEF(int64_t, filteredExportFlag);
TILING_DATA_FIELD_DEF(int64_t, bitWidth);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxThreadNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingHashTableExport, EmbeddingHashTableExportTilingData)

struct EmbeddingHashTableExportCompileInfo {
    int64_t maxThreadNum;
    int64_t coreNumAiv;
};

} // namespace optiling
#endif // EMBEDDING_HASH_TABLE_EXPORT_H_