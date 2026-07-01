/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GATHER_V2_TILING_DATA_H
#define GATHER_V2_TILING_DATA_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalCount);
TILING_DATA_FIELD_DEF(uint64_t, outerSize);
TILING_DATA_FIELD_DEF(uint64_t, axisDim);
TILING_DATA_FIELD_DEF(uint64_t, innerSize);
TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
TILING_DATA_FIELD_DEF(uint64_t, perCoreCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreCount);
TILING_DATA_FIELD_DEF(uint32_t, tileCount);
TILING_DATA_FIELD_DEF(uint32_t, xTypeBytes);
TILING_DATA_FIELD_DEF(uint32_t, indexTypeBytes);
TILING_DATA_FIELD_DEF(int64_t, axis);
TILING_DATA_FIELD_DEF(int64_t, batchDims);
TILING_DATA_FIELD_DEF(uint32_t, negativeIndexSupport);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV2, GatherV2TilingData)
} // namespace optiling

#endif // GATHER_V2_TILING_DATA_H
