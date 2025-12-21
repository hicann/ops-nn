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
 * \file dynamic_block_quant_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_QUANT_H
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
constexpr uint16_t MAX_CORE_COUNT = 50;

BEGIN_TILING_DATA_DEF(DynamicBlockQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);                     // 核心数
TILING_DATA_FIELD_DEF(int64_t, ubSize);                           // UB大小
TILING_DATA_FIELD_DEF(int64_t, vfLen);                            // 向量长度
TILING_DATA_FIELD_DEF(float, minScale);                           // 最小缩放比例
TILING_DATA_FIELD_DEF(int64_t, roundMode);                        // 数据类型转换的模式
TILING_DATA_FIELD_DEF(int64_t, dstType);                          // 输出数据类型
TILING_DATA_FIELD_DEF(int64_t, blockSizeRow);                     // 行方向的块大小
TILING_DATA_FIELD_DEF(int64_t, blockSizeCol);                     // 列方向的块大小
TILING_DATA_FIELD_DEF(int64_t, rowNum);                           // 行数
TILING_DATA_FIELD_DEF(int64_t, colNum);                           // 列数
TILING_DATA_FIELD_DEF(int64_t, rowBlockLoopNum);                  // 行方向块的循环次数
TILING_DATA_FIELD_DEF(int64_t, colBlockLoopNum);                  // 列方向块的循环次数
TILING_DATA_FIELD_DEF(int64_t, rowUbBlockLoopNum);                // 行方向UB块的循环次数
TILING_DATA_FIELD_DEF(int64_t, colUbBlockLoopNum);                // 列方向UB块的循环次数
TILING_DATA_FIELD_DEF(int64_t, rowUbFactor);                      // 行方向UB因子
TILING_DATA_FIELD_DEF(int64_t, colUbFactor);                      // 列方向UB因子
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);                      // 实际使用的核心数
TILING_DATA_FIELD_DEF(int64_t, rowTileNum);                       // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, colTileNum);                       // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, normalCoreRowTileNum);             // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, normalCoreColTileNum);             // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreRowTileNum);               // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreColTileNum);               // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, rowNormalCoreNum);                 // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, colNormalCoreNum);                 // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, rowTailCoreNum);                   // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, colTailCoreNum);                   // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, perCoreRowNum);                    // 每个核处理的行数
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tailRowList); // 每个核在尾块中处理数据的行数
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tailColBlockStartList); // 每个核在尾块中处理数据的开始列
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tailColBlockEndList); // 每个核在尾块中处理数据的结束列
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicBlockQuant, DynamicBlockQuantTilingData)

struct DynamicBlockQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicBlockQuantTilingParam {
    int64_t tilingKey = 0;
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
    int64_t vfLen = 0;
    float minScale = 0.0;
    int64_t roundMode = 0;
    int64_t dstType = 0;
    int64_t blockSizeRow = 0;
    int64_t blockSizeCol = 0;
    int64_t rowNum = 0;
    int64_t colNum = 0;
    int64_t rowBlockLoopNum = 0;
    int64_t colBlockLoopNum = 0;
    int64_t rowUbBlockLoopNum = 0;
    int64_t colUbBlockLoopNum = 0;
    int64_t rowUbFactor = 0;
    int64_t colUbFactor = 0;
    int64_t usedCoreNum = 0;
    int64_t rowTileNum = 0;
    int64_t colTileNum = 0;
    int64_t normalCoreRowTileNum = 0;
    int64_t normalCoreColTileNum = 0;
    int64_t tailCoreRowTileNum = 0;
    int64_t tailCoreColTileNum = 0;
    int64_t rowNormalCoreNum = 0;
    int64_t colNormalCoreNum = 0;
    int64_t rowTailCoreNum = 0;
    int64_t colTailCoreNum = 0;
};

enum class RoundModeList : int64_t
{
    MODE_UNDEFINED = -1,
    MODE_NONE = 0,
    MODE_RINT = 1,
    MODE_FLOOR = 2,
    MODE_CEIL = 3,
    MODE_ROUND = 4,
    MODE_TRUNC = 5,
    MODE_ODD = 6,
    MODE_HYBRID = 7,
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_QUANT_H