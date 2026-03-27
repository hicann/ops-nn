/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DYNAMIC_BLOCK_MX_QUANT_TILINGDATA_H
#define DYNAMIC_BLOCK_MX_QUANT_TILINGDATA_H

#include <cstdint>

struct DynamicBlockMxQuantTilingData {
    int64_t tilingKey;
    int64_t totalCoreNum;               // 总核数
    int64_t usedCoreNum;                // 实际使用的核数
    int64_t ubSize;                     // UB大小
    int64_t roundMode;                  // 数据类型转换的模式
    int64_t dstType;                    // 输出数据类型
    int64_t scaleAlg;                   // OCP Microscaling Formats (Mx) Specification或Dynamic Dtype Range实现
    int64_t blockSizeRow;               // 行方向的块大小
    int64_t blockSizeCol;               // 列方向的块大小
    int64_t batchNum;                   // batch数量
    int64_t rowNum;                     // 行数
    int64_t colNum;                     // 列数
    int64_t singleBatchRowBlockLoopNum; // 单batch行方向基本块的数量
    int64_t rowBlockLoopNum;            // 行方向基本块的数量
    int64_t colBlockLoopNum;            // 列方向基本块的数量
    int64_t rowUbBlockLoopNum;          // 行方向UB块的一次循环的基本块数量
    int64_t colUbBlockLoopNum;          // 列方向UB块的一次循环的基本块数量
    int64_t rowUbFactor;                // 行方向UB因子
    int64_t colUbFactor;                // 列方向UB因子
    int64_t rowTileNum;                 // 行方向的核数
    int64_t colTileNum;                 // 列方向的核数
    int64_t normalCoreRowTileNum;       // 头核行方向基本块的数量
    int64_t normalCoreColTileNum;       // 头核列方向基本块的数量
    int64_t tailCoreRowTileNum;         // 尾核行方向基本块的数量
    int64_t tailCoreColTileNum;         // 尾核列方向基本块的数量
    int64_t rowNormalCoreNum;           // 行方向头核数量
    int64_t colNormalCoreNum;           // 列方向头核数量
    int64_t rowTailCoreNum;             // 行方向尾核数量
    int64_t colTailCoreNum;             // 列方向尾核数量
    int64_t blockH;                     // 基本块的高
    int64_t blockW;                     // 基本块的宽
    int64_t rowScaleNum;                // 行方向scale的数量
    int64_t colScaleNum;                // 列方向scale的数量
    float dstTypeMax;                   // 计算scale所需值
};

#endif // DYNAMIC_BLOCK_MX_QUANT_TILINGDATA_H