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
 * \file linear_index_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_V2_H

#include "register/tilingdata_base.h"
constexpr size_t MAX_DIM_NUM = 8;
namespace optiling {
BEGIN_TILING_DATA_DEF(LinearIndexV2TilingParam)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tensorId)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreDataNum)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreFormerDataNum)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreTailDataNum)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreFormerTime)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreTailTime)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreFormerDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreFormerTime)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailTime)
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM_NUM, indicesMask)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(LinearIndexV2TilingParamOp, LinearIndexV2TilingParam)

BEGIN_TILING_DATA_DEF(LinearIndexV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(LinearIndexV2TilingParam, params)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(LinearIndexV2, LinearIndexV2TilingData)

struct LinearIndexV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
    bool isAscend950 = false;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_V2_H