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
 * \file softmax_focal_loss_tiling_data.h
 * \brief softmax_focal_loss tiling data shared by host and kernel
 */

#ifndef SOFTMAX_FOCAL_LOSS_TILING_DATA_H
#define SOFTMAX_FOCAL_LOSS_TILING_DATA_H

#include <cstdint>

struct SoftmaxFocalLossArch35TilingData {
    int64_t a = 0;               // 非归约轴元素总数(行数 B)
    int64_t r = 0;               // 归约轴长度(类别数 D)
    int64_t realCoreNum = 0;     // 实际使用核数
    int64_t blockFactor = 0;     // 主核负责的行数
    int64_t tailBlockFactor = 0; // 尾核负责的行数
    int64_t rowsPerTile = 0;     // 一次处理的行数
    int64_t colsPerChunk = 0;    // 一次处理的列数, 除末块外均为该值(vfLen 的整数倍)
    int64_t chunkNum = 0;        // 列方向分块数
    float gamma = 2.0f;
    float alpha = 0.25f;
};

#endif // SOFTMAX_FOCAL_LOSS_TILING_DATA_H
