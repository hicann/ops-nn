/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (28) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_AVG_POOL3D_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_AVG_POOL3D_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AvgPool3DTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inN);
    TILING_DATA_FIELD_DEF(uint64_t, inC);
    TILING_DATA_FIELD_DEF(uint64_t, tileC);
    TILING_DATA_FIELD_DEF(uint64_t, inD);
    TILING_DATA_FIELD_DEF(uint64_t, inH);
    TILING_DATA_FIELD_DEF(uint64_t, inW);
    TILING_DATA_FIELD_DEF(uint64_t, outD);
    TILING_DATA_FIELD_DEF(uint64_t, outH);
    TILING_DATA_FIELD_DEF(uint64_t, outW);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(int64_t, divisorOverride);
    TILING_DATA_FIELD_DEF(uint64_t, countIncludePad);
    TILING_DATA_FIELD_DEF(uint64_t, ceilMode);
    TILING_DATA_FIELD_DEF(uint64_t, formerLength);
    TILING_DATA_FIELD_DEF(uint64_t, formerNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailLength);
    TILING_DATA_FIELD_DEF(uint64_t, tailNum);
    TILING_DATA_FIELD_DEF(uint64_t, indexBufLen);
    TILING_DATA_FIELD_DEF(uint64_t, windowWNum);
    TILING_DATA_FIELD_DEF(uint64_t, tileInput);
    TILING_DATA_FIELD_DEF(uint64_t, tileHW);
    TILING_DATA_FIELD_DEF(uint64_t, atomicAddNum);
    TILING_DATA_FIELD_DEF(uint64_t, useCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, ncFactor);
    TILING_DATA_FIELD_DEF(uint64_t, doFactor);
    TILING_DATA_FIELD_DEF(uint64_t, hoFactor);
    TILING_DATA_FIELD_DEF(uint64_t, woFactor);
    TILING_DATA_FIELD_DEF(uint64_t, ncOuter);
    TILING_DATA_FIELD_DEF(uint64_t, doOuter);
    TILING_DATA_FIELD_DEF(uint64_t, hoOuter);
    TILING_DATA_FIELD_DEF(uint64_t, woOuter);
    TILING_DATA_FIELD_DEF(uint64_t, ncTail);
    TILING_DATA_FIELD_DEF(uint64_t, doTail);
    TILING_DATA_FIELD_DEF(uint64_t, hoTail);
    TILING_DATA_FIELD_DEF(uint64_t, woTail);
    TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
    TILING_DATA_FIELD_DEF(uint64_t, blockTail);
    TILING_DATA_FIELD_DEF(uint64_t, totalIdx);
    TILING_DATA_FIELD_DEF(uint64_t, coreNums);
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AvgPool3D, AvgPool3DTilingData)
REGISTER_TILING_DATA_CLASS(AvgPool3DV2, AvgPool3DTilingData)
}  // namespace optiling

#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_AVG_POOL3D_H_
