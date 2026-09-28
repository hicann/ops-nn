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
 * \file gn_training_reduce_tiling_struct.h
 * \brief GNTrainingReduce tiling data structures shared by host and kernel.
 */

#ifndef GN_TRAINING_REDUCE_TILING_STRUCT_H
#define GN_TRAINING_REDUCE_TILING_STRUCT_H

#include <cstdint>

// MAX_PATTERN_RANK = 合轴后 pattern 最大可能轴数。
constexpr int32_t GN_TRAINING_REDUCE_MAX_PATTERN_RANK = 4;

// 无 mask 的 AscendC::Reg::LoadAlign 按整寄存器宽度读取（dav_3510 向量寄存器宽 256B），
// 末次 repeat 的读会越过逻辑元素末尾至多一个向量，故 pre 阶段三份 buffer 与二分缓存在
// 逻辑大小之外各留一个向量的读余量。host 侧按 buffer 份数（3 份 pre + 1 份 cache）计入
// UB 预算，kernel 侧按同一常量分配（见 base.h Init 中的 static_assert）。
constexpr int64_t GN_TRAINING_REDUCE_UB_READ_SLACK = 256; // 字节

// Base / Group 共用 TilingData。
struct GNTrainingReduceTilingData {
    // ─── pattern 描述 ───
    int32_t axisNum; // 合轴后轴数 (A 起头、严格交替；偶数→tail-R)
    int64_t axisShape[GN_TRAINING_REDUCE_MAX_PATTERN_RANK];  // 合轴后每根轴 size
    int64_t axisStride[GN_TRAINING_REDUCE_MAX_PATTERN_RANK]; // 每根轴 GM stride（按 element 计）

    // ─── 多核切分（外层 A loop 扁平为线性计数，按 coreNum 均匀分核）───
    int64_t aLoopCntTotal;     // ∏(外层 A 轴) × aSplitChunkCnt
    int64_t aSplitChunkCnt;    // CeilDiv(axisShape[aSplitIdx], aUbFactor)
    int64_t aBigCoreLoopCnt;   // 大核处理块数
    int64_t aSmallCoreLoopCnt; // 小核处理块数
    int32_t aBigCoreCnt;       // 大核个数
    int32_t usedCoreNum;       // 实际使用核数

    // ─── UB 切分 ───
    int32_t aSplitIdx;       // UB 内 A 切分轴下标
    int32_t rSplitIdx;       // UB 内 R 切分轴下标
    int64_t aUbFactor;       // valid：A 维实际元素数（不保证 block 对齐）
    int64_t rUbFactor;       // valid：R 维实际元素数
    int64_t rUbFactorAlign;  // padded：UB 行 stride
    int64_t innerAProdAlign; // padded：切分轴右侧 A 轴乘积
    int64_t innerRProdAlign; // padded：切分轴右侧 R 轴乘积

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal; // ∏(外层 R 轴) × CeilDiv(axisShape[rSplitIdx], rUbFactor)

    // ─── UB buffer 字节数 ───
    int64_t preBufSize;     // pre 阶段单份 buffer 字节（含 R 维）
    int64_t postBufSize;    // post 阶段单份 buffer 字节（不含 R 维）
    int64_t cacheBufUbSize; // 固定 16 × 1024

    // ─── group 模板 ───
    int64_t rGroupCnt; // Phase 1 分组数 / workspace R 维（Base 不读）
};

// Empty 模板独立 TilingData。
struct GNTrainingReduceEmptyTilingData {
    // ─── 多核切分 ───
    int32_t usedCoreNum;       // EMPTY_A: 0；EMPTY_R: 切分算出的核数
    int64_t aTotal;            // ∏(所有 A 轴 axisShape)，EMPTY_A 不读
    int64_t aUbFactor;         // 单 chunk a 元素数（4 约束取 min）
    int32_t aBigCoreCnt;       // 大核个数
    int64_t aBigCoreLoopCnt;   // 每大核 chunk 数
    int64_t aSmallCoreLoopCnt; // 每小核 chunk 数

    // ─── UB buffer ───
    int64_t postBufSize; // post 阶段单份 buffer 字节数
};

#endif
