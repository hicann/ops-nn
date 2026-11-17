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
 * \file chunk_loss_tiling.h
 * \brief ChunkLoss Host tiling 数据结构与编译信息（arch22 / arch35 共用字段超集）
 */
#ifndef CHUNK_LOSS_TILING_H_
#define CHUNK_LOSS_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkLossTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, v);
TILING_DATA_FIELD_DEF(uint32_t, aicNum);
TILING_DATA_FIELD_DEF(uint32_t, nSplit);        // arch35；arch22 固定写 1
TILING_DATA_FIELD_DEF(uint32_t, deterministic); // 0 非确定性；1 AtomicAdd；2 TempDh RMW（arch35）
TILING_DATA_FIELD_DEF(uint32_t, tileM);         // 32/64/112/128（arch35）；arch22 写 128
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(int64_t, ignoreIndex);
TILING_DATA_FIELD_DEF(uint64_t, syncBaseAddr); // Soft FFTS 基址（CrossCore）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkLoss, ChunkLossTilingData)

struct ChunkLossCompileInfo {
    uint32_t aicNum = 0;
    int64_t sysWorkspaceSize = 0;
};
} // namespace optiling

#endif
