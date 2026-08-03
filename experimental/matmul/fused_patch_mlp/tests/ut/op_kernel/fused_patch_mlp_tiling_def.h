/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_PATCH_MLP_TILING_DEF_H
#define FUSED_PATCH_MLP_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

struct FusedPatchMlpTilingData {
    uint32_t totalN;
    uint32_t inFeatures;
    uint32_t hiddenSize;
    uint32_t geluTileSize;
    uint32_t geluMode;
    uint32_t numLayers;
    TCubeTiling mm0Tiling;
    TCubeTiling mmHTiling;
};

inline void IFusedPatchMlpTilingData(uint8_t* tiling, FusedPatchMlpTilingData* constData)
{
    memcpy(constData, tiling, sizeof(FusedPatchMlpTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    FusedPatchMlpTilingData tilingData;            \
    IFusedPatchMlpTilingData(tilingPointer, &tilingData)

#endif // FUSED_PATCH_MLP_TILING_DEF_H
