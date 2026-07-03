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
 * \file swiglu_group_quant_grad_tiling.h
 * \brief SwiGLU Group Dynamic Quant Backward tiling data definition
 */

#ifndef SWIGLU_GROUP_QUANT_GRAD_TILING_H
#define SWIGLU_GROUP_QUANT_GRAD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SwigluGroupQuantGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreNumAll);
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
    
    TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
    TILING_DATA_FIELD_DEF(uint32_t, dim2H);
    TILING_DATA_FIELD_DEF(uint32_t, dimH);
    
    TILING_DATA_FIELD_DEF(uint32_t, hasWeight);
    TILING_DATA_FIELD_DEF(uint32_t, hasYOrigin);
    TILING_DATA_FIELD_DEF(uint32_t, hasGroupIndex);
    TILING_DATA_FIELD_DEF(uint32_t, hasClampLimit);
    TILING_DATA_FIELD_DEF(uint32_t, needSplitH);
    
    TILING_DATA_FIELD_DEF(float, clampLimit);
    
    TILING_DATA_FIELD_DEF(uint32_t, groupNum);
    
    TILING_DATA_FIELD_DEF(uint32_t, tileTokens);
    TILING_DATA_FIELD_DEF(uint32_t, tileH);
    TILING_DATA_FIELD_DEF(uint32_t, numHTiles);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuantGrad, SwigluGroupQuantGradTilingData)

struct CoreCompileInfo {
};

struct SwigluGroupQuantGradCompileInfo {
    uint32_t totalCore = 1;
    uint32_t ubSize = 0;
    uint32_t inputDataByte = 4;
    float clampLimit = -1.0f;
    uint32_t hasClampLimit = 0;
    uint32_t hasWeight = 0;
    uint32_t hasYOrigin = 0;
    uint32_t hasGroupIndex = 0;
    
    uint32_t dataNumSingleUb = 1;
    uint32_t blockNum = 8;
};

} // namespace optiling

#endif // SWIGLU_GROUP_QUANT_GRAD_TILING_H