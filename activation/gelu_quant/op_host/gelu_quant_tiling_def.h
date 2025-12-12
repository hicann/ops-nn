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
 * \file gelu_quant_tiling_def.h
 * \brief
 */

#ifndef GELU_QUANT_TILING_DEF_H
#define GELU_QUANT_TILING_DEF_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluQuantTilingData)

TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, endAxisLen);
TILING_DATA_FIELD_DEF(int64_t, endAxisLenAligned);
TILING_DATA_FIELD_DEF(int64_t, rowOuter);
TILING_DATA_FIELD_DEF(int64_t, colOuter);
TILING_DATA_FIELD_DEF(uint32_t, rowInner);
TILING_DATA_FIELD_DEF(uint32_t, rowTail);
TILING_DATA_FIELD_DEF(uint32_t, colInner);
TILING_DATA_FIELD_DEF(uint32_t, colTail);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, coexistentNodeNum);
TILING_DATA_FIELD_DEF(uint32_t, coexistentNodeElementNum);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, approximate);
TILING_DATA_FIELD_DEF(uint32_t, inputScaleType);
TILING_DATA_FIELD_DEF(uint32_t, inputOffsetType);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, dstType);
TILING_DATA_FIELD_DEF(uint32_t, roundMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluQuant, GeluQuantTilingData)
} // namespace optiling
#endif // GELU_QUANT_TILING_DEF_H