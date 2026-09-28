/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
TILING_DATA_FIELD_DEF(uint64_t, includeSelf);
TILING_DATA_FIELD_DEF(uint64_t, mode);
// Execution plan selector shared by the legacy and cache kernels.
TILING_DATA_FIELD_DEF(uint64_t, M);
// 低内存分支参数
TILING_DATA_FIELD_DEF(int32_t, coreNums);
TILING_DATA_FIELD_DEF(uint64_t, xDim0);
TILING_DATA_FIELD_DEF(uint64_t, xDim1);
TILING_DATA_FIELD_DEF(uint64_t, indicesDim0);
TILING_DATA_FIELD_DEF(uint64_t, indicesDim1);
TILING_DATA_FIELD_DEF(uint64_t, updatesDim0);
TILING_DATA_FIELD_DEF(uint64_t, updatesDim1);
TILING_DATA_FIELD_DEF(uint64_t, batchSize);
TILING_DATA_FIELD_DEF(uint64_t, realDim);
// 分桶散射分支参数（末轴 reduction=none 且 varN >> indicesN 的稀疏大 var 场景）：
// bktMode 非 0 时启用该分支，其余字段仅在该分支下有效。
// 这些维度不复用 xDim*/indicesDim*（那几个字段是 cache-op 路径的归约结果，
// 复用会与该路径相互干扰），故另设独立字段，与既有分支零耦合。
// 字段一律追加在 realDim 之后，不得插入中间：UT 按 uint64 下标读取序列化 payload。
TILING_DATA_FIELD_DEF(uint64_t, bktMode);
TILING_DATA_FIELD_DEF(uint64_t, bktRows);        // 独立散射行数
TILING_DATA_FIELD_DEF(uint64_t, bktVarN);        // 每行 var 元素数
TILING_DATA_FIELD_DEF(uint64_t, bktIndicesN);    // 每行更新元素数
TILING_DATA_FIELD_DEF(uint64_t, bktTileLen);     // 单个输出 tile 元素数（2 的幂）
TILING_DATA_FIELD_DEF(uint64_t, bktNumTiles);    // ceil(bktVarN / bktTileLen) = 桶数
TILING_DATA_FIELD_DEF(uint64_t, bktShift);       // log2(bktTileLen)
TILING_DATA_FIELD_DEF(uint64_t, bktFifoDepth);   // 每桶 UB FIFO 深度（16 的倍数）
TILING_DATA_FIELD_DEF(uint64_t, bktStride);      // 每核 GM 桶区容量（条目数）
TILING_DATA_FIELD_DEF(uint64_t, bktRowsPerCore); // 每核基础行数
TILING_DATA_FIELD_DEF(uint64_t, bktFrontCore);   // 前 bktFrontCore 个核各多处理 1 行
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElementsV2, ScatterElementsV2TilingData)
struct ScatterElementsV2CompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
    bool is_regbase = false;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H
