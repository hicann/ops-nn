/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 
#ifndef SMOOTH_L1_LOSS_GRAD_V2_TILING_DATA_H_
#define SMOOTH_L1_LOSS_GRAD_V2_TILING_DATA_H_

#include <cstdint>

struct SmoothL1LossGradV2TilingData {
    uint64_t smallCoreDataNum;      // 数据量较小的 core 处理元素数
    uint64_t bigCoreDataNum;        // 数据量较大的 core 处理元素数
    uint64_t finalBigTileNum;       // 大 core 的 tile 数量
    uint64_t finalSmallTileNum;     // 小 core 的 tile 数量
    uint64_t tileDataNum;           // 每个 tile 处理元素数
    uint64_t smallTailDataNum;      // 小 core 尾块元素数
    uint64_t bigTailDataNum;        // 大 core 尾块元素数
    uint64_t tailBlockNum;          // 尾块数量（大 core 数量）
    uint64_t totalDataNum;          // predict 总元素数（用于 mean 归约）
    uint64_t doutDataNum;           // dout 元素个数（1 表示标量）
    float sigma;                    // 平滑系数
    int reduction;                  // 归约模式（0:none, 1:mean, 2:sum）
};

#endif