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
 * \file act_ulq_clamp_min_grad_tiling_data.h
 * \brief ActULQClampMinGrad 算子 TilingData 定义（arch35 / RegBase）。
 *
 * 算子类别 = All Reduce（axis_source: implicit_all，全轴 reduce_sum，输出 0 维标量）。
 *   合轴后 pattern 恒为 AR（tail-R），isTailR 恒 true，不入 TilingKey / kernel 模板参数。
 *   MAX_PATTERN_RANK = 2（All Reduce），host 与 kernel 共享此唯一定义。
 *
 * ✅ 使用标准 C++ struct 定义 TilingData
 * ❌ 禁止使用废弃的 BEGIN_TILING_DATA_DEF 宏
 *
 * 字段分 5 组：
 *   - pattern 描述：axisNum / axisShape[2] / axisStride[2]
 *   - 多核切分（fused aLoop）：aLoopCntTotal / aSplitChunkCnt / aBigCoreLoopCnt /
 *              aSmallCoreLoopCnt / aBigCoreCnt / usedCoreNum
 *   - UB 切分：aSplitAxisIdx / rSplitAxisIdx / aUbFactor[Align] / rUbFactor[Align] /
 *              innerAProd[Align] / innerRProd[Align]
 *   - 外层 R loop 扁平化：rLoopCntTotal
 *   - UB buffer 字节数：preReduceUbSize / postReduceUbSize / tmpBufUbSize / cacheBufUbSize
 *   - group 模板：rGroupCnt
 */
#ifndef OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_DATA_H_
#define OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_DATA_H_

#include <cstdint>

// All Reduce → 合轴后恒为 AR（axisNum=2）。host 与 kernel 共享此唯一定义（跨 H4 不变量 #8）。
constexpr int32_t MAX_PATTERN_RANK = 2;

struct ActULQClampMinGradTilingData {
    // ─── pattern 描述（去 1 / 合轴 / 补 leading A / 补 R 增广后）───
    int32_t axisNum = 0;                        // 恒 2（AR），All Reduce
    int64_t axisShape[MAX_PATTERN_RANK] = {0};  // 合轴后每根轴的 size（[A=1, R=元素总数]）
    int64_t axisStride[MAX_PATTERN_RANK] = {0}; // 每根轴的 GM stride（按 element）
    // axisType[i] 由位置 i 的奇偶决定：i 偶→A，i 奇→R；不入 TilingData

    // ─── 多核切分（fused aLoop） ───
    int64_t aLoopCntTotal = 0;     // ∏(outer A 整根) × aSplitChunkCnt（All Reduce 下常为 1）
    int64_t aSplitChunkCnt = 0;    // CeilDiv(axisShape[aSplitAxisIdx], aUbFactor)
    int64_t aBigCoreLoopCnt = 0;   // 大核处理的块数
    int64_t aSmallCoreLoopCnt = 0; // 小核处理的块数
    int32_t aBigCoreCnt = 0;       // 大核个数
    int32_t usedCoreNum = 0;       // 实际使用核数

    // ─── UB 切分 ───
    int32_t aSplitAxisIdx = 0;   // All Reduce 下 = 0
    int32_t rSplitAxisIdx = 0;   // All Reduce 下 = 1
    int64_t aUbFactor = 0;       // valid：A 维实际元素数
    int64_t aUbFactorAlign = 0;  // padded：UB 行 stride
    int64_t rUbFactor = 0;       // valid：R 维实际元素数
    int64_t rUbFactorAlign = 0;  // padded：UB 行 stride（burst tail 非对齐时 > rUbFactor）
    int64_t innerAProd = 0;      // actual
    int64_t innerAProdAlign = 0; // padded
    int64_t innerRProd = 0;      // actual
    int64_t innerRProdAlign = 0; // padded

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal = 0;

    // ─── UB buffer 字节数 ───
    int64_t preReduceUbSize = 0;  // 单路 preIn buffer 大小，blockSize 对齐
    int64_t postReduceUbSize = 0; // 单路 outBuf 大小，blockSize 对齐
    int64_t tmpBufUbSize = 0;     // 单份 tmpBuf 大小（实占 2×），blockSize 对齐
    int64_t cacheBufUbSize = 0;   // 固定 16 KB（needs_bisection=true）

    // ─── group 模板唯一新增 ───
    int64_t rGroupCnt = 0; // Phase 1 分组数 = Phase 2 workspace R 维大小
};

#endif // OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_DATA_H_
