/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file group_dynamic_block_quant_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_DYNAMIC_BLOCK_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_DYNAMIC_BLOCK_QUANT_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GroupedDynamicBlockQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);          // 核数
TILING_DATA_FIELD_DEF(int64_t, ubSize);                // Ub字节数大小
TILING_DATA_FIELD_DEF(int64_t, vfLen);                 // 寄存器长度
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);           // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, headCoreNum);           // 头核数
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);           // 尾核数
TILING_DATA_FIELD_DEF(float, minScale);                // 最小缩放比例
TILING_DATA_FIELD_DEF(int64_t, roundMode);             // 数据类型转换的模式
TILING_DATA_FIELD_DEF(int64_t, dstType);               // 输出数据类型
TILING_DATA_FIELD_DEF(int64_t, rowBlockSize);          // 行方向的块大小
TILING_DATA_FIELD_DEF(int64_t, colBlockSize);          // 列方向的块大小
TILING_DATA_FIELD_DEF(int64_t, batchNum);              // batch数
TILING_DATA_FIELD_DEF(int64_t, rowNum);                // x行数
TILING_DATA_FIELD_DEF(int64_t, colNum);                // x列数
TILING_DATA_FIELD_DEF(int64_t, scaleRowNum);           // scale行数
TILING_DATA_FIELD_DEF(int64_t, scaleColNum);           // scale列数
TILING_DATA_FIELD_DEF(int64_t, uo);                    // 切分轴上的循环次数
TILING_DATA_FIELD_DEF(int64_t, groupNum);              // group的个数
TILING_DATA_FIELD_DEF(int64_t, blockFactor);           // 切分轴上每次循环的长度
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);       // 切分轴上尾循环的长度
TILING_DATA_FIELD_DEF(int64_t, groupBlockNumHeadCore); // 头核计算groupBlock个数
TILING_DATA_FIELD_DEF(int64_t, groupBlockNumTailCore); // 尾核计算groupBlock个数
TILING_DATA_FIELD_DEF(int64_t, maxUbRow);              // 每个group单次循环的数据大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedDynamicBlockQuant, GroupedDynamicBlockQuantTilingData)

struct GroupedDynamicBlockQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct GroupedDynamicBlockQuantTilingParam {
    int64_t tilingKey = 0;
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
    int64_t vfLen = 0;
    int64_t usedCoreNum = 0;
    int64_t headCoreNum = 0;
    int64_t tailCoreNum = 0;
    float minScale = 0.0;
    int64_t roundMode = 0;
    int64_t dstType = 0;
    int64_t rowBlockSize = 0;
    int64_t colBlockSize = 0;
    int64_t batchNum = 1;
    int64_t rowNum = 0;
    int64_t colNum = 0;
    int64_t scaleRowNum = 0;
    int64_t scaleColNum = 0;
    int64_t uo = 0;
    int64_t groupNum = 0;
    int64_t blockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t groupBlockNumHeadCore = 0;
    int64_t groupBlockNumTailCore = 0;
    int64_t maxUbRow = 0;
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
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_DYNAMIC_BLOCK_QUANT_H