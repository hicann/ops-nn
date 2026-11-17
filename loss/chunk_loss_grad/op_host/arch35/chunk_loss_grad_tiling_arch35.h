/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkLossGradArch35TilingData)
TILING_DATA_FIELD_DEF(uint64_t, numTokens);
TILING_DATA_FIELD_DEF(uint64_t, hiddenDim);
TILING_DATA_FIELD_DEF(uint64_t, vocabSize);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, tokenTile);
TILING_DATA_FIELD_DEF(uint32_t, vocabTile);
TILING_DATA_FIELD_DEF(uint32_t, vectorChunk);
TILING_DATA_FIELD_DEF(uint64_t, logitsBytes);
TILING_DATA_FIELD_DEF(uint64_t, logitsGradBytes);
TILING_DATA_FIELD_DEF(uint64_t, partialBytes);
TILING_DATA_FIELD_DEF(uint64_t, coreWorkspaceBytes);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, logitsTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, weightTiling);
TILING_DATA_FIELD_DEF(uint32_t, scheduleMode);
TILING_DATA_FIELD_DEF(uint32_t, tokenPitch);
TILING_DATA_FIELD_DEF(uint32_t, cacheElements);
TILING_DATA_FIELD_DEF(uint32_t, residentAccum);
TILING_DATA_FIELD_DEF(uint32_t, accumElements);
TILING_DATA_FIELD_DEF(uint32_t, slotCount);
TILING_DATA_FIELD_DEF(int64_t, ignoreIndex);
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ChunkLossGrad, ChunkLossGradArch35TilingData)
} // namespace optiling
