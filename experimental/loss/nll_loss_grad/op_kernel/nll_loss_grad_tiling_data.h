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
 * \file nll_loss_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef NLLLOSSGRAD_TILING_DATA_H
#define NLLLOSSGRAD_TILING_DATA_H

#include <cstdint>

// reduction: 0 = none, 1 = sum, 2 = mean
struct NllLossGradTilingData {
    int64_t nDim = 0;      // 样本行数 N
    int64_t cDim = 0;      // 类别数 C
    int64_t coreNum = 1;   // 实际参与计算的核数
    int64_t reduction = 2; // 0 none, 1 sum, 2 mean
    int64_t ignoreIndex = -100;
    int64_t bigWeight = 0;     // 0: NormalWeight(key=2000), 1: BigWeight(key=2001)
    int64_t maxLine = 0;       // 前 redundantLine 个核处理的行数
    int64_t lowerLine = 0;     // 其余核处理的行数
    int64_t redundantLine = 0; // 处理 maxLine 行的核数量
    int64_t lineTile = 1;      // NormalWeight 单次 UB tile 处理的行数
    int64_t cAlign = 0;        // C 按 8(float) 对齐后的元素数
    int64_t outUbSize = 0;     // 输出 float buffer 元素数
    int64_t colTile = 0;       // BigWeight 列方向 tile 元素数
    int64_t moveOutTime = 1;   // BigWeight 列方向拆分次数 ceil(C/colTile)
};
#endif
