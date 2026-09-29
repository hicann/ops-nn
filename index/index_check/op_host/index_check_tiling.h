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
 * \file index_check_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INDEX_CHECK_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INDEX_CHECK_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexCheckTilingParam)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tensorId)
TILING_DATA_FIELD_DEF(uint64_t, maxBatchSize)
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, tensorLens)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(IndexCheckTilingParamOp, IndexCheckTilingParam)

BEGIN_TILING_DATA_DEF(IndexCheckTilingData)
TILING_DATA_FIELD_DEF_STRUCT(IndexCheckTilingParam, params)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(IndexCheck, IndexCheckTilingData)

struct IndexCheckCompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t workspaceSize = 0;
    uint64_t ubSizePlatform = 0;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_INDEX_CHECK_H
