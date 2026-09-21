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
 * \file cla_gate_backward_tiling_data.h
 * \brief
 */

#ifndef CLA_GATE_BACKWARD_TILING_DATA_H_
#define CLA_GATE_BACKWARD_TILING_DATA_H_

#include <cstdint>

struct ClaGateBackwardTilingData {
    int64_t headNum;              // N：attention head 数
    int64_t headDim;              // D：value head dim，归约轴（仅 128 / 256）
    int64_t totalHeads;           // TN = T * N：待分核的行数（每行是一个 (t,n)）
    int64_t usedCoreNum;          // 实际启用核数
    int64_t baseCoreHeads;        // 每核基础处理的 TN 数（尾核 TN）
    int64_t extraCoreCount;       // 前 extraCoreCount 个核（头核）各多处理 1 个 TN
    int64_t batch;                // 一次搬运/计算的最大 TN 数（队列按此配）
    int64_t headCoreLoopCount;    // 头核（TN = baseCoreHeads + 1）核内循环次数
    int64_t headCoreHeadsPerLoop; // 头核每轮处理的 TN 数
    int64_t tailCoreLoopCount;    // 尾核（TN = baseCoreHeads）核内循环次数
    int64_t tailCoreHeadsPerLoop; // 尾核每轮处理的 TN 数
    int64_t reduceTmpSize;        // ReduceSum(AR) 沿 D 归约所需 tmp 字节数
};

#endif // CLA_GATE_BACKWARD_TILING_DATA_H_
