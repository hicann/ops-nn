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
 * \file scatter_elements_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, eachNum);
TILING_DATA_FIELD_DEF(uint64_t, extraTaskCore);
TILING_DATA_FIELD_DEF(uint64_t, eachPiece);
TILING_DATA_FIELD_DEF(uint64_t, inputOnePiece);
TILING_DATA_FIELD_DEF(uint64_t, inputCount);
TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
TILING_DATA_FIELD_DEF(uint64_t, updatesCount);
TILING_DATA_FIELD_DEF(uint64_t, inputOneTime);
TILING_DATA_FIELD_DEF(uint64_t, indicesOneTime);
TILING_DATA_FIELD_DEF(uint64_t, updatesOneTime);
TILING_DATA_FIELD_DEF(uint64_t, inputEach);
TILING_DATA_FIELD_DEF(uint64_t, indicesEach);
TILING_DATA_FIELD_DEF(uint64_t, inputLast);
TILING_DATA_FIELD_DEF(uint64_t, indicesLast);
TILING_DATA_FIELD_DEF(uint64_t, inputLoop);
TILING_DATA_FIELD_DEF(uint64_t, indicesLoop);
TILING_DATA_FIELD_DEF(uint64_t, inputAlign);
TILING_DATA_FIELD_DEF(uint64_t, indicesAlign);
TILING_DATA_FIELD_DEF(uint64_t, updatesAlign);
TILING_DATA_FIELD_DEF(uint64_t, lastIndicesLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastIndicesEach);
TILING_DATA_FIELD_DEF(uint64_t, lastIndicesLast);
TILING_DATA_FIELD_DEF(uint64_t, oneTime);
TILING_DATA_FIELD_DEF(uint64_t, lastOneTime);
TILING_DATA_FIELD_DEF(uint64_t, modeFlag);
// 310P tiling parameter
TILING_DATA_FIELD_DEF(uint64_t, M); //indices行数
TILING_DATA_FIELD_DEF(uint64_t, varN);
TILING_DATA_FIELD_DEF(uint64_t, indicesN);
TILING_DATA_FIELD_DEF(uint64_t, updatesN);
TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, frontCoreData);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreData);
TILING_DATA_FIELD_DEF(uint64_t, computeMode);
TILING_DATA_FIELD_DEF(uint64_t, MLeft);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElementsV2, ScatterElementsV2TilingData)
struct ScatterElementsV2CompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H
