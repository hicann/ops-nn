/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_PATCH_MLP_TILING_H
#define FUSED_PATCH_MLP_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// Keep this field order identical to the kernel-side FusedPatchMlpTilingData definition.
BEGIN_TILING_DATA_DEF(FusedPatchMlpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalN);
TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, geluTileSize);
TILING_DATA_FIELD_DEF(uint32_t, geluMode);
TILING_DATA_FIELD_DEF(uint32_t, numLayers);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm0Tiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmHTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedPatchMlp, FusedPatchMlpTilingData)

struct FusedPatchMlpCompileInfo {};

} // namespace optiling

#endif // FUSED_PATCH_MLP_TILING_H
