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
 * \file dequant_bias_tiling.h
 * \brief dequant_bias_tiling
 */

#ifndef DEQUANT_BIAS_TILING_H
#define DEQUANT_BIAS_TILING_H

#include <cmath>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DequantBiasTilingData)
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, nAlign);
TILING_DATA_FIELD_DEF(uint32_t, asExist);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, perCoreRow);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreRow);
TILING_DATA_FIELD_DEF(uint32_t, inBufferSize);
TILING_DATA_FIELD_DEF(uint32_t, wsBufferSize);
TILING_DATA_FIELD_DEF(uint32_t, asBufferSize);
TILING_DATA_FIELD_DEF(uint32_t, biasBufferSize);
TILING_DATA_FIELD_DEF(uint32_t, perCoreLoopRow);
TILING_DATA_FIELD_DEF(uint32_t, perCoreTailLoopRow);
TILING_DATA_FIELD_DEF(uint32_t, perCoreLoops);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLoopRow);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreTailLoopRow);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLoops);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantBias, DequantBiasTilingData)

struct DequantBiasCompileInfo {};
}  // namespace optiling

#endif  // DEQUANT_BIAS_TILING_H