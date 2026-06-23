/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_tiling.h
 * \brief Tiling data and structure definitions for SwiGLU Group Quant operator
 */

#ifndef SWIGLU_GROUP_QUANT_TILING_H
#define SWIGLU_GROUP_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

// Max core count for per-core group assignment arrays
constexpr uint16_t MAX_CORE_COUNT = 64;

// Tiling data definition
BEGIN_TILING_DATA_DEF(SwigluGroupQuantTilingData)
    // Shape info
    TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
    TILING_DATA_FIELD_DEF(uint32_t, dim2H);
    TILING_DATA_FIELD_DEF(uint32_t, dimH);

    // Mode flags
    TILING_DATA_FIELD_DEF(uint32_t, isGroup);
    TILING_DATA_FIELD_DEF(uint32_t, hasWeight);
    TILING_DATA_FIELD_DEF(uint32_t, hasClamp);
    TILING_DATA_FIELD_DEF(uint32_t, outputOrigin);

    // Attribute parameters
    TILING_DATA_FIELD_DEF(float, clampLimit);
    TILING_DATA_FIELD_DEF(float, dstTypeMaxFinite);

    // Tile parameters
    TILING_DATA_FIELD_DEF(uint32_t, tileTokens);

    // Core distribution
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, groupTokensSum);    // sum(group_index), tail start for group mode
    TILING_DATA_FIELD_DEF(uint32_t, minLoadCoreIdx);    // lightest core, handles group-mode tail

    // Core distribution for group mode (continuous segment split, per-core arrays indexed by blockIdx)
    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, coreGroupStartArr);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, coreGroupCountArr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuant, SwigluGroupQuantTilingData)

// Compile info structure
struct SwigluGroupQuantCompileInfo {
    uint32_t totalCore = 1;
    uint32_t ubSize = 0;
};

// Tiling param structure
struct SwigluGroupQuantTilingParam {
    uint32_t usedCoreNum = 1;
    uint32_t tokensPerCore = 0;
    uint32_t tileTokens = 0;
    uint32_t groupTokensSum = 0;
    uint32_t minLoadCoreIdx = 0;
    uint32_t coreGroupStartArr[MAX_CORE_COUNT] = {0};
    uint32_t coreGroupCountArr[MAX_CORE_COUNT] = {0};
};

} // namespace optiling

#endif // SWIGLU_GROUP_QUANT_TILING_H
