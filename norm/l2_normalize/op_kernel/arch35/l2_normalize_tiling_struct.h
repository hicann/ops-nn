/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_kernel/arch35/l2_normalize_tiling_struct.h
// =============================================================================
//
// ROLE: Tiling data structures shared between host-side tiling and device-side
//   kernel for the L2Normalize operator on arch35 (Ascend 950).
//   - L2NormalizeTilingData        base / group 共用（ReduceGenericTilingData 布局）
//   - L2NormalizeEmptyTilingData   empty 独立（ReduceEmptyTilingData 布局，不复用 Base/Group）
//   Host 侧填充（l2_normalize_tiling_arch35.cpp）与 kernel 侧读取
//   （GET_TILING_DATA_WITH_STRUCT）看到同一份结构体布局。
// =============================================================================

#pragma once

#include <cstdint>

// rank≤8、axis 任意子集：合轴+补leadingA 最坏 2rseg+1=9
constexpr int32_t MAX_PATTERN_RANK = 9;

// L2Normalize TilingData — 按 reduction 范式 ReduceGenericTilingData 通用布局 + 算子自定义字段 eps
struct L2NormalizeTilingData {
    // ─── pattern 描述（合轴预处理四步产出：去1→合轴→补leadingA→补R增广）───
    int32_t axisNum = 0;                        // 2~MAX_PATTERN_RANK（合轴后轴数）
    int64_t axisShape[MAX_PATTERN_RANK] = {0};  // 合轴后每根轴的 size
    int64_t axisStride[MAX_PATTERN_RANK] = {0}; // 每根轴的 GM stride（按 element 计）
    // axisType[i] 不需要：i 偶→A，i 奇→R（A 起头 + 严格交替）

    // ─── 多核切分（外层 A loop 扁平为线性计数，按 coreNum 均匀分核，大小核均衡）───
    int64_t aLoopCntTotal = 0;   // ∏(outer A 整根) × aSplitChunkCnt
    int64_t aSplitChunkCnt = 0;  // CeilDiv(axisShape[aSplitIdx], aUbFactor)
    int64_t aBigCoreLoopCnt = 0; // 大核处理的块数
    int64_t aSmallCoreLoopCnt = 0; // 小核处理的块数（==0 等价 aLoopCntTotal < coreNum，仅前 aBigCoreCnt 核工作）
    int32_t aBigCoreCnt = 0; // 大核个数（= aLoopCntTotal % coreNum）
    int32_t usedCoreNum = 0; // 实际使用核数

    // ─── UB 切分 ───
    int32_t aSplitIdx = 0; // UB 内被切分的 A 轴下标
    int32_t rSplitIdx = 0; // UB 内被切分的 R 轴下标
    int64_t aUbFactor = 0; // valid：A 维实际元素数（可能非 block 对齐，UB 行 stride 由 postBufSize 兜底）
    int64_t rUbFactor = 0; // valid：R 维实际元素数
    int64_t rUbFactorAlign = 0; // padded：UB 行 stride（tail-R 且 UB 切尾轴且尾轴全载且非对齐时 > rUbFactor，其余相等）
    int64_t innerAProdAlign = 0; // padded：含最内 burst-tail A 的 CeilAlign
    int64_t innerRProdAlign = 0; // padded：含最内 burst-tail R 的 CeilAlign

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal = 0; // ∏(外层 R 轴 size) × CeilDiv(axisShape[rSplitIdx], rUbFactor)

    // ─── UB buffer 字节数 ───
    int64_t preBufSize = 0;  // pre 阶段单份 buffer 大小（含 R 维，按 maxDtypeSize），blocksize 对齐
    int64_t postBufSize = 0; // post 阶段单份 buffer 大小（不含 R 维，按 maxDtypeSize），blocksize 对齐
    int64_t cacheBufUbSize = 0; // 固定 16 × 1024（= 16 KB，二分缓存树专用）

    // ─── group 模板 ───
    int64_t rGroupCnt = 0; // Phase 1 分组数 = Phase 2 workspace R 维大小（base 不读）

    // ─── 算子自定义字段（范式约定：追加在末尾）───
    float eps = 0.0f; // attr eps（默认 1e-4）：kernel 侧 clamp_eps 用（分母 = sqrt(max(s, eps))）
};

struct L2NormalizeEmptyTilingData {
    // ─── 多核切分 ───
    int32_t usedCoreNum = 0;       // 本算子 EMPTY_A / EMPTY_R 均 = 0（全核早退）
    int64_t aTotal = 0;            // ∏(所有 A 轴 axisShape)
    int64_t aUbFactor = 0;         // 单 chunk a 元素数（范式 4 约束取 min；本算子早退路径不消费）
    int32_t aBigCoreCnt = 0;       // 大核个数（同上，保留范式布局）
    int64_t aBigCoreLoopCnt = 0;   // 每大核 chunk 数
    int64_t aSmallCoreLoopCnt = 0; // 每小核 chunk 数

    // ─── UB buffer ───
    int64_t postBufSize = 0; // post 阶段单份 buffer 字节数（本算子早退路径不消费，保留范式布局）
};
