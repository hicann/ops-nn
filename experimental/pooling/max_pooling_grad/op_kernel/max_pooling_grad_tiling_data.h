/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pooling_grad_tiling_data.h
 * \brief MaxPoolingGrad 算子 TilingData 结构体定义
 *
 * MaxPoolingGrad: 最大池化反向传播，计算逻辑为非重叠窗口下的梯度分配:
 *   dx[i] = (x[i] == y[i]) ? dy[i] : 0
 *
 * Tiling 策略:
 *   - BLOCK_SIZE = 256 (CompareScaler / Select API 的对齐要求)
 *   - 按 blocks (256B 对齐单元) 划分各 core 的数据量
 *   - Big core / small core 非均匀分配以处理余数
 *   - UB 内 tile 划分: tileBlockNum 为每 tile 可容纳的 256B block 数
 */

#ifndef MAX_POOLING_GRAD_TILING_DATA_H
#define MAX_POOLING_GRAD_TILING_DATA_H

struct MaxPoolingGradTilingData {
    // ---- 多核切分 ----
    uint64_t smallCoreDataNum = 0; // small core 数据量 (元素数)
    uint64_t bigCoreDataNum = 0;   // big core 数据量 (元素数)
    // ---- UB tile 切分 ----
    uint64_t ubPartDataNum = 0; // 每个 UB tile 可处理的元素数
    // ---- 尾块处理 ----
    uint64_t smallCoreTailDataNum = 0; // small core 最后一个 tile 的元素数
    uint64_t bigCoreTailDataNum = 0;   // big core 最后一个 tile 的元素数
    // ---- 循环次数 ----
    uint64_t smallCoreLoopNum = 0; // small core 的总循环次数
    uint64_t bigCoreLoopNum = 0;   // big core 的总循环次数
    uint64_t tailBlockNum = 0;     // big core 的数量 (前 tailBlockNum 个 core)
    // ---- Boundary clamp ----
    uint64_t lastCoreValidDataNum = 0; // last small core actual valid data (非零=需要钳位)
};

#endif
