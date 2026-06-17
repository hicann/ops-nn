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
 * \file euclidean_norm_tiling_data.h
 * \brief EuclideanNorm 算子 TilingData 定义
 *
 * 直接套用 ReduceNormalGenericTilingData 布局，
 * 无 EuclideanNorm 专属字段（sqrt 是 in-register、不需要传系数；与 mean 需要 invRTotal 不同）。
 *
 * 字段分 5 组：
 *   - pattern 描述：axisNum / axisShape[MAX_PATTERN_RANK] / axisStride[MAX_PATTERN_RANK]（去 1 / 合轴 / 补 R 增广后的 pattern）
 *   - 多核切分（fused aLoop）：aLoopCntTotal / aSplitChunkCnt / aBigCoreLoopCnt /
 *              aSmallCoreLoopCnt / aBigCoreCnt / usedCoreNum
 *   - UB 切分：aSplitAxisIdx / rSplitAxisIdx / aUbFactor / aUbFactorAlign /
 *              rUbFactor / innerAProd[Align] / innerRProd[Align]
 *   - 外层 R loop 扁平化：rLoopCntTotal
 *   - UB buffer 字节数：preReduceUbSize / postReduceUbSize / tmpBufUbSize / cacheBufUbSize
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_TILING_DATA_H_
#define OPS_NORM_EUCLIDEAN_NORM_TILING_DATA_H_

#include <cstdint>

constexpr int32_t MAX_PATTERN_RANK = 9;

struct EuclideanNormTilingData {
    // ─── pattern 描述（去 1 / 合轴 / 补 leading A / 补 R 增广之后）───
    int32_t axisNum = 0;                                       // 2~MAX_PATTERN_RANK
    int64_t axisShape[MAX_PATTERN_RANK] = {0};      // 合轴后每根轴的 size，未用位置填 1
    int64_t axisStride[MAX_PATTERN_RANK] = {0};     // 每根轴的 GM stride（按 element）
    // axisType[i] 由位置 i 的奇偶决定：i 偶→A，i 奇→R（A 起头 + 严格交替），不入 tilingdata

    // ─── 多核切分（normal 模板：外层 A loop 全部 fuse 成线性计数，按 coreNum 均衡分核） ───
    int64_t aLoopCntTotal = 0;                                 // ∏(outer A 整根) × aSplitChunkCnt
    int64_t aSplitChunkCnt = 0;                                // CeilDiv(axisShape[aSplitAxisIdx], aUbFactor)
    int64_t aBigCoreLoopCnt = 0;                               // 前 aBigCoreCnt 个核处理的 aLoop 数 (= aSmallCoreLoopCnt + 1，当 aBigCoreCnt>0)
    int64_t aSmallCoreLoopCnt = 0;                             // 其余核处理的 aLoop 数 (= aLoopCntTotal / coreNum，floor)
    int32_t aBigCoreCnt = 0;                                   // 大核个数 (= aLoopCntTotal % coreNum)
    int32_t usedCoreNum = 0;                                   // 实际使用核数 (aSmallCoreLoopCnt>0 ? coreNum : aBigCoreCnt)

    // ─── UB 切分 ───
    int32_t aSplitAxisIdx = 0;
    int32_t rSplitAxisIdx = 0;
    int64_t aUbFactor = 0;
    int64_t aUbFactorAlign = 0;
    int64_t rUbFactor = 0;                                     // actual valid R count
    int64_t rUbFactorAlign = 0;                                // padded R count，仅 tail-R + rSplit==lastR 时 = CeilAlign(rUbFactor, bsElem)；
                                                               //   其它情况 = rUbFactor。kernel 用于 UB padded 步距 / ReduceSum srcShape[1]
    int64_t innerAProd = 0;                                    // actual（未补）
    int64_t innerAProdAlign = 0;                               // padded（含 LastA CeilAlign）
    int64_t innerRProd = 0;                                    // actual
    int64_t innerRProdAlign = 0;                               // padded（含 LastR CeilAlign，仅 tail R 时生效）

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal = 0;

    // ─── UB buffer 字节数 ───
    int64_t preReduceUbSize = 0;
    int64_t postReduceUbSize = 0;
    int64_t tmpBufUbSize = 0;
    int64_t cacheBufUbSize = 0;                                // 固定 16 KB

    // ─── group 模板 ───
    int64_t rGroupCnt = 0;                                     // Phase 1 分组数 = Phase 2 workspace R 维大小
};

#endif // OPS_NORM_EUCLIDEAN_NORM_TILING_DATA_H_
