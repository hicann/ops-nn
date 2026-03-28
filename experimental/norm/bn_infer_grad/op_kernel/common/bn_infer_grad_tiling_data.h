/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file bn_infer_grad_tiling_data.h
 * \brief BnInferGrad TilingData 结构体定义
 */

#ifndef _BN_INFER_GRAD_TILING_DATA_H_
#define _BN_INFER_GRAD_TILING_DATA_H_

struct BnInferGradTilingData {
    // 基本参数
    int64_t totalElements = 0;     // grads 的总元素数
    int64_t channelSize = 0;       // 通道数 C
    int64_t spatialSize = 0;       // 空间维度大小：NCHW: H*W; NHWC: H*W; NC1HWC0: H*W

    // 格式相关参数
    int64_t formatMode = 0;        // 0=NCHW, 1=NHWC, 2=NC1HWC0
    int64_t N = 0;                 // batch 维度
    int64_t C1 = 0;                // NC1HWC0: C1 维度
    int64_t C0 = 0;                // NC1HWC0: C0 维度（16 或 32）

    // 多核切分参数（CONTIGUOUS 分支）
    int64_t usedCoreNum = 0;       // 使用的核数
    int64_t elemsPerCore = 0;      // 每核处理的元素数
    int64_t tailCoreElems = 0;     // 尾核处理的元素数

    // 多核切分参数（NC1HWC0 分支）
    int64_t totalTasks = 0;        // 总任务数 = N * C1，每任务处理 H*W*C0 个元素
    int64_t tasksPerCore = 0;      // 每核处理的任务数
    int64_t tailCoreTasks = 0;     // 尾核处理的任务数

    // UB 切分参数
    int64_t tileLen = 0;           // 每次搬入 UB 的元素数
    int64_t numTiles = 0;          // 每核内的 tile 数
    int64_t lastTileLen = 0;       // 最后一个 tile 的元素数

    // NC1HWC0 UB 切分参数
    int64_t tileHW = 0;            // NC1HWC0: 每次处理的 H*W 数量
    int64_t numTilesHW = 0;        // NC1HWC0: 每个任务内的 tile 数
    int64_t lastTileHW = 0;        // NC1HWC0: 最后一个 tile 的 H*W 数量

    // 对齐参数
    int64_t alignedC = 0;          // 通道数对齐到 32 字节后的长度（fp32 元素数）
    int64_t alignedC0 = 0;         // NC1HWC0: C0 对齐到 32 字节后的长度（fp32 元素数）

    // epsilon（存储为 int64_t 位模式以保持对齐，kernel 侧 reinterpret 为 float）
    int64_t epsilonBits = 0;       // epsilon 的 float 位模式（低 32 位有效）
};

#endif // _BN_INFER_GRAD_TILING_DATA_H_
