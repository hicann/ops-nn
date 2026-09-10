/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_TILING_DATA_H_
#define _RENORM_TILING_DATA_H_

#include <cstdint>

// normMode 枚举
// 0: p > 0 且 p != inf (p-范数)
// 1: p = 0 (非零计数，0-范数)
// 2: p = inf (最大绝对值，inf-范数)
// 3: maxNorm = 0 (直接输出全零)

// 模板编号 (与 tiling_key TEMPLATE 维度对应)
// 0: Template A (SM-CT) - Slice-Major Continuous Single-Level
// 1: Template B (SM-TL) - Slice-Major Continuous Two-Level
// 2: Template C (SM-CR) - Slice-Major Cross-Core Reduction
// 3: Template D (SM-ST) - Slice-Major Stride
// 4: Template E (BM-VD) - Block-Major Vector Direct
// 5: Template F (BM-VG) - Block-Major Vector Grouped

struct RenormTilingData {
    // === 公共字段（所有模板使用）===
    int64_t totalElements = 0; // 输入总元素数
    int64_t dim = 0;           // 保留维度（正值），沿此轴切片
    int64_t sliceCount = 0;    // 子张量个数 = shape[dim]
    int64_t blockSize = 0;     // 每个连续段大小 = prod(shape[dim+1:])
    int64_t numBlocks = 0;     // 块数 = prod(shape[:dim])
    int64_t tileLength = 0;    // chunk 大小（元素数）
    int64_t slicesPerCore = 0; // 每核处理的子张量数
    float p = 0.0f;            // 范数幂次
    float maxNorm = 0.0f;      // 最大范数值
    float eps = 0.0f;          // 防除零极小值
    int32_t normMode = 0;      // 场景分支标识

    // === Template B (SM-TL) 字段 ===
    int64_t blockFactor = 0;  // 一级切分粒度 (每核处理的 block 数)
    int64_t blockFactor2 = 0; // 二级切分粒度 (每核迭代次数)

    // === Template C (SM-CR) 字段 ===
    int64_t reduceSplitsPerCore = 0; // 归约轴每核切分数
    int64_t workspaceSize = 0;       // workspace 大小 (字节)

    // === Template D (SM-ST) 字段 ===
    int64_t stride = 0; // stride 访问步长 (元素数)

    // === Template E/F (BM-VD/BM-VG) 字段 ===
    int64_t sliceTileLength = 0; // sliceCount 方向的 tile 大小 (block-major)
};

#endif // _RENORM_TILING_DATA_H_
