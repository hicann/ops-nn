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
 * \file scatter_add_with_sorted_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_WITH_SORTED_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_WITH_SORTED_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterAddWithSortedTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, extraTaskCore);
TILING_DATA_FIELD_DEF(uint64_t, eachCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCount);
TILING_DATA_FIELD_DEF(uint64_t, inputCount);
TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
TILING_DATA_FIELD_DEF(uint64_t, updatesCount);
TILING_DATA_FIELD_DEF(uint64_t, inputOneTime);
TILING_DATA_FIELD_DEF(uint64_t, updatesOneTime);
TILING_DATA_FIELD_DEF(uint64_t, updatesAlign);
TILING_DATA_FIELD_DEF(uint64_t, maxSize);
TILING_DATA_FIELD_DEF(uint64_t, eachNum);
TILING_DATA_FIELD_DEF(uint64_t, eachLoop);
TILING_DATA_FIELD_DEF(uint64_t, eachTail);
TILING_DATA_FIELD_DEF(uint64_t, lastNum);
TILING_DATA_FIELD_DEF(uint64_t, lastLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastTail);
TILING_DATA_FIELD_DEF(uint64_t, updatesLoop);
TILING_DATA_FIELD_DEF(uint64_t, updatesEach);
TILING_DATA_FIELD_DEF(uint64_t, updatesLast);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAddWithSorted, ScatterAddWithSortedTilingData)
struct ScatterAddWithSortedCompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_WITH_SORTED_H
