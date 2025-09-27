/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file linear_index_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearIndexTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, eachCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCount);
TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
TILING_DATA_FIELD_DEF(uint64_t, indicesAlign);
TILING_DATA_FIELD_DEF(uint64_t, maxSize);
TILING_DATA_FIELD_DEF(uint64_t, eachNum);
TILING_DATA_FIELD_DEF(uint64_t, eachLoop);
TILING_DATA_FIELD_DEF(uint64_t, eachTail);
TILING_DATA_FIELD_DEF(uint64_t, lastNum);
TILING_DATA_FIELD_DEF(uint64_t, lastLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastTail);
TILING_DATA_FIELD_DEF(uint64_t, target);
TILING_DATA_FIELD_DEF(uint64_t, selfStride);
TILING_DATA_FIELD_DEF(uint64_t, indicesStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearIndex, LinearIndexTilingData)
struct LinearIndexCompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_H
