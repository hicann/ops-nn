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
 * \file swiglu_group_quant_tiling_def.h
 * \brief Tiling data definition for kernel unit tests (must match op_host struct layout)
 */
#ifndef SWIGLU_GROUP_QUANT_TILING_DEF_H_
#define SWIGLU_GROUP_QUANT_TILING_DEF_H_

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint16_t MAX_CORE_COUNT = 64;

BEGIN_TILING_DATA_DEF(SwigluGroupQuantTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
    TILING_DATA_FIELD_DEF(uint32_t, dim2H);
    TILING_DATA_FIELD_DEF(uint32_t, dimH);
    TILING_DATA_FIELD_DEF(uint32_t, isGroup);
    TILING_DATA_FIELD_DEF(uint32_t, hasWeight);
    TILING_DATA_FIELD_DEF(uint32_t, hasClamp);
    TILING_DATA_FIELD_DEF(uint32_t, outputOrigin);
    TILING_DATA_FIELD_DEF(float, clampLimit);
    TILING_DATA_FIELD_DEF(float, dstTypeMaxFinite);
    TILING_DATA_FIELD_DEF(uint32_t, tileTokens);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, groupTokensSum);
    TILING_DATA_FIELD_DEF(uint32_t, minLoadCoreIdx);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, coreGroupStartArr);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, coreGroupCountArr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuant, SwigluGroupQuantTilingData)
} // namespace optiling
#endif // SWIGLU_GROUP_QUANT_TILING_DEF_H_
