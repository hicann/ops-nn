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
 * \file dynamic_dual_level_mx_quant_struct.h
 * \brief
 */

#ifndef OPS_QUANT_DYNAMIC_DUAL_LEVEL_MX_QUANT_TILING_DATA_H
#define OPS_QUANT_DYNAMIC_DUAL_LEVEL_MX_QUANT_TILING_DATA_H

struct DynamicDualLevelMxQuantTilingData {
    int64_t tilingKey = 0;
    int64_t totalCoreNum = 0;          // 总核数
    int64_t usedCoreNum = 0;           // 实际使用的核数
    int64_t roundMode = 0;             // 数据类型转换的模式
    int64_t level0BlockSize = 0;       // level0量化的块大小
    int64_t level1BlockSize = 0;       // level1量化的块大小
    int64_t rowSize = 0;               // 行长度
    int64_t colSize = 0;               // 列长度
    int64_t blockSizeRow = 0;          // 行方向的基本块大小 512
    int64_t blockSizeCol = 0;          // 列方向的基本块大小 1
    int64_t rowBlockNum = 0;           // 行方向基本块的个数
    int64_t colBlockNum = 0;           // 列方向基本块的个数
    int64_t rowTileNum = 0;            // 行方向分核循环次数
    int64_t colTileNum = 0;            // 列方向分核循环次数
    int64_t normalTileRowBlockNum = 0; // 正常核行方向基本块个数
    int64_t normalTileColBlockNum = 0; // 正常核列方向基本块个数
    int64_t tailTileRowBlockNum = 0;   // 尾块核行方向基本块个数
    int64_t tailTileColBlockNum = 0;   // 尾块核列方向基本块个数
    int64_t normalTileRowSize = 0;     // 正常核行方向长度
    int64_t tailTileRowSize = 0;       // 尾块核行方向长度
    int64_t normalTileRowLoopNum = 0;  // 正常核内行方向循环次数
    int64_t normalTileColLoopNum = 0;  // 正常核内列方向循环次数
    int64_t tailTileRowLoopNum = 0;    // 尾块核内行方向循环次数
    int64_t tailTileColLoopNum = 0;    // 尾块核内列方向尾循环次数
    int64_t ubFactor = 0;              // ub内最多处理多少个基本块
    int64_t tailAlignNum = 0;          // 尾核尾块补齐个数 0-整除，1-128，2-256，3-384，4-512
    int64_t copyMethod = 0;            // 核内搬入模式 0-多行搬入，1-单行搬入
    int64_t needSmoothScale = 0;       // 是否需要进行smooth scale
};

#endif // OPS_QUANT_DYNAMIC_DUAL_LEVEL_MX_QUANT_TILING_DATA_H
