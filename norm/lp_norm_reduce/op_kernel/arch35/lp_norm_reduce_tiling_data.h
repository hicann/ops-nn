/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_kernel/arch35/lp_norm_reduce_tiling_data.h
// =============================================================================
//
// ROLE: LpNormReduce TilingData 汇总结构体 —— host 侧 TilingFunc 与 device 侧
//   kernel 共用（同一布局，两侧必须同构）。
//
//   据 docs/LpNormReduce/design/TilingData.md 落码（公共 TilingData
//   汇总结构体），套用 reduction 范式 ReduceGenericTilingData /
//   ReduceEmptyTilingData 布局（范式 references/reduction-template-overview.md
//   §4.1.1/§4.1.2、reduction-binary-group-tiling.md §5.4、
//   reduction-tiling-preprocess.md §2）：
//   - LpNormReduceTilingData       —— Base（tilingKey=0）/ Group（tilingKey=1）
//                                     模板共用（group 仅多读 rGroupCnt，base 不读）
//   - LpNormReduceEmptyTilingData  —— Empty 模板专用（tilingKey=2，EMPTY_A /
//                                     EMPTY_R 共用，kernel 内按 usedCoreNum 区分；
//                                     独立 struct，不复用 Generic struct）
//   字段与 TilingData.md 登记逐字段一致（不增不删）；dtype 不进 TilingData /
//   TilingKey（DTYPE_X 编译期实例化，fp16 中间升 fp32 由 kernel Cast 链处理）；
//   epsilon / keepdim 不进 TilingData（前者不参与 reduce 计算、后者仅 InferShape）。
//
//   lp_norm_reduce_tiling_struct.h 仅作向后兼容转发，include 本文件。
//
// =============================================================================

// 本算子 TilingData — 套用 reduction 范式 ReduceGenericTilingData / ReduceEmptyTilingData 布局
#ifndef LP_NORM_REDUCE_TILING_DATA_H_
#define LP_NORM_REDUCE_TILING_DATA_H_

#include <cstdint>

// MAX_PATTERN_RANK 推导（范式 reduction-tiling-preprocess.md §2「算子级 pattern 上界分析」）：
//   axes 为运行时任意 ListInt（rseg 不定 → 按范式兜底规则取 9）；
//   且 rank(x) ≤ 8 下最坏 pattern：axes 取间隔轴 {0,2,4,6} → R A R A R A R A
//   → 补 leading A → A R A R A R A R A（9 轴、A 结尾，rseg=4 → 2×rseg+1 = 9）；
//   范式框架上界亦为 9 → MAX_PATTERN_RANK = 9
constexpr int32_t MAX_PATTERN_RANK = 9;

// ---------------------------------------------------------------------------
// LpNormReduceTilingData —— Base / Group 模板共用（范式 §4.1.1；tilingKey=0 / 1）
//
// 字段单位约定：切分 / 因子类字段为元素数；preBufSize / postBufSize /
// cacheBufUbSize 为字节数；aBigCoreCnt / usedCoreNum 为核数。
// ---------------------------------------------------------------------------
struct LpNormReduceTilingData {
    // ─── pattern 描述 ───
    int32_t axisNum = 0; // 合轴后轴数 2~MAX_PATTERN_RANK；偶→tail-R、奇→tail-A（kernel 现算）
    int64_t axisShape[MAX_PATTERN_RANK] = {0}; // 合轴后每根轴 size（i 偶→A，i 奇→R，A 起头严格交替）
    int64_t axisStride[MAX_PATTERN_RANK] = {0}; // 每根轴 GM stride（按 element 计）

    // ─── 多核切分（外层 A loop 扁平为线性计数，按 coreNum 均匀分核）───
    int64_t aLoopCntTotal = 0;   // ∏(outer A 整根) × aSplitChunkCnt
    int64_t aSplitChunkCnt = 0;  // CeilDiv(axisShape[aSplitIdx], aUbFactor)
    int64_t aBigCoreLoopCnt = 0; // 大核处理块数 = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0)
    int64_t
        aSmallCoreLoopCnt = 0; // 小核处理块数 = aLoopCntTotal / coreNum（floor；== 0 等价只有前 aBigCoreCnt 核工作）
    int32_t aBigCoreCnt = 0; // 大核个数 = aLoopCntTotal % coreNum
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
    int64_t cacheBufUbSize = 0; // 二分缓存树专用 buffer，恒 16 × 1024 = 16 KB

    // ─── group 模板 ───
    int64_t rGroupCnt = 0; // Phase 1 分组数 = Phase 2 workspace R 维大小（base 不读）

    // ─── 算子自定义字段（本算子唯一追加）───
    int64_t pOrder = 0; // 范数阶数 p 原值（attr p；+inf 哨兵 2147483647 / -inf 哨兵 -2147483648），
                        // kernel 据此运行时分发：max / min / 非零计数 / abs 求和 / 二进制快速幂幂和
};

// ---------------------------------------------------------------------------
// LpNormReduceEmptyTilingData —— Empty 模板专用（范式 §4.1.2，独立 struct
// 不复用 LpNormReduceTilingData；tilingKey=2，EMPTY_A / EMPTY_R 共用）
// ---------------------------------------------------------------------------
struct LpNormReduceEmptyTilingData {
    // ─── 多核切分 ───
    int32_t usedCoreNum = 0; // EMPTY_A: 0（所有核早退，SetBlockDim(1)）；EMPTY_R: 按 aTotal 切分算出的核数
    int64_t aTotal = 0;      // ∏(所有 A 轴 axisShape)，EMPTY_A 不读
    int64_t aUbFactor = 0; // 单 chunk a 元素数（4 约束取 min：4KB 下界 / 优先多核 / UB 上限 / aTotal 兜底）
    int32_t aBigCoreCnt = 0;       // 大核个数 = aLoopCntTotal % coreNum
    int64_t aBigCoreLoopCnt = 0;   // 每大核 chunk 数
    int64_t aSmallCoreLoopCnt = 0; // 每小核 chunk 数

    // ─── UB buffer ───
    int64_t postBufSize = 0; // post 阶段单份 buffer 字节数（单 buf 封顶 64KB）
};

#endif // LP_NORM_REDUCE_TILING_DATA_H_
