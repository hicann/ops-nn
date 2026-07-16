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
 * \file mse_loss_tiling_data.h
 * \brief tiling data struct
 */

#ifndef MSELOSS_TILING_DATA_H_
#define MSELOSS_TILING_DATA_H_

struct MseLossTilingData {
    int64_t totalNum = 0;               // 总元素数量
    int64_t blockFactor = 1;            // 每个核处理的元素数量
    int64_t ubFactor = 0;               // 每次 UB 循环处理的元素数量
    int64_t reduction = 2;              // 0:none, 1:sum, 2:mean
    int64_t blockNum = 1;               // 实际使用核数
    int64_t workspaceFloatsPerCore = 1; // 每核 workspace 槽位的 float 元素数
    float meanScale = 1.0f;             // mean 模式缩放系数
};
#endif // MSELOSS_TILING_DATA_H_
