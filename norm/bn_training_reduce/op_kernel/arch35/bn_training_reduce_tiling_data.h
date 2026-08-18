/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_TILING_DATA_H_
#define BN_TRAINING_REDUCE_TILING_DATA_H_

#include <cstdint>

constexpr int32_t MAX_PATTERN_RANK = 9;

struct BNTrainingReduceTilingData {
    int32_t axisNum;                      // 合轴后 pattern 轴数，实际范围 2..4
    int64_t axisShape[MAX_PATTERN_RANK];  // 各轴 valid 元素数，未用槽填 1
    int64_t axisStride[MAX_PATTERN_RANK]; // 各轴 GM 元素 stride，未用槽填 0

    int64_t aLoopCntTotal;     // 外层 A fused loop 总数
    int64_t aSplitChunkCnt;    // A 切分轴 chunk 总数
    int64_t aBigCoreLoopCnt;   // 每个大核处理的 A loop 数
    int64_t aSmallCoreLoopCnt; // 每个小核处理的 A loop 数
    int32_t aBigCoreCnt;       // 大核个数
    int32_t usedCoreNum;       // 实际参与计算的核数

    int32_t aSplitAxisIdx;   // 被 UB 切分的 A 轴下标
    int32_t rSplitAxisIdx;   // 被 UB 切分的 R 轴下标
    int64_t aUbFactor;       // A split valid 元素数
    int64_t aUbFactorAlign;  // A split padded 元素数
    int64_t rUbFactor;       // R split valid 元素数
    int64_t rUbFactorAlign;  // R split padded 元素数
    int64_t innerAProd;      // aSplit 右侧 A 轴 valid 乘积
    int64_t innerAProdAlign; // aSplit 右侧 A 轴 padded 乘积
    int64_t innerRProd;      // rSplit 右侧 R 轴 valid 乘积
    int64_t innerRProdAlign; // rSplit 右侧 R 轴 padded 乘积

    int64_t rLoopCntTotal; // 外层 R fused loop 总数

    int64_t preReduceUbSize;  // 单路输入 preIn buffer 字节数，32B 对齐
    int64_t postReduceUbSize; // 单路 fp32 输出 buffer 字节数，32B 对齐
    int64_t tmpBufUbSize;     // 单份 fp32 reduce tmp buffer 字节数，32B 对齐
    int64_t cacheBufUbSize;   // 二分缓存树字节数，Normal/Group 固定 16KB

    int64_t rGroupCnt; // Group Phase 1 分组数及 Phase 2 workspace R 维
};

#endif // BN_TRAINING_REDUCE_TILING_DATA_H_
