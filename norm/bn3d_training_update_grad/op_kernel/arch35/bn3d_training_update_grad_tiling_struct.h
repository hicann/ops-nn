/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_kernel/arch35/bn3d_training_update_grad_tiling_struct.h
// =============================================================================
//
// ROLE: Tiling data structures shared between host-side tiling and device-side
//   kernel for the BN3DTrainingUpdateGrad operator on arch35 (Ascend 950).
//
//   AUTHORITATIVE ("份数终值 = 2 份"):
//     - BN3DTrainingUpdateGradTilingData        (base/group shared, 22 fields)
//     - BN3DTrainingUpdateGradEmptyTilingData   (empty EMPTY_A/EMPTY_R, 7 fields)
//   Struct names copied verbatim from REG_OP(BN3DTrainingUpdateGrad) (⛔ NOT the
//   auto Bn3d* camel form). Field order/type mirror the field contract
//   field-for-field (no missing, no extra). The host TilingFunc
//   (bn3d_training_update_grad_tiling_arch35.cpp) fills exactly these and the
//   device kernel reads them.
//
//   LEGACY SCAFFOLD (retained, see bottom): the not-yet-migrated device kernel
//   scaffold (bn3d_training_update_grad_*_kernel.h / _apt.cpp — out of scope for the
//   Tiling task) still references Bn3dTrainingUpdateGradTilingData<kRank>,
//   SplitResult, MultiCoreResult and kMaxInputSlots/kMaxOutputSlots/kPhysNodes.
//   These are kept UNCHANGED so the run-package device build keeps compiling; the
//   authoritative structs above are differently named and do not clash.
// =============================================================================

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Shared compile-time constants
// ---------------------------------------------------------------------------
// 本算子单 A(通道)轴、通道居中最坏 ARAR=4 → base/group pattern 上界 4。
constexpr int32_t MAX_PATTERN_RANK = 4;        // base/group 用；empty struct 线性、无 pattern 数组
constexpr int64_t kFp32Bytes = 4;              // sizeof(float)
constexpr int64_t kCacheBufUbSize = 16 * 1024; // 二分缓存树固定 16KB
constexpr int64_t kNPreTile = 7;               // ⑤ VF 融合后动态 tile buffer 份数（= dynamicNode）

// ===========================================================================
// (1.1) base/group 共用 struct（BN3DTrainingUpdateGradTilingData，22 字段；
//       对齐 reduction ReduceGenericTilingData）。
//       单位：元素数（coreNum/idx/count 为核数/下标/计数；字节字段以 Bytes/UbSize 命名）。
// ===========================================================================
struct BN3DTrainingUpdateGradTilingData {
    // ─── pattern 描述（去1/合轴/补 leading A/补 R 增广之后）───
    int32_t axisNum;                      // 合轴后轴数（2~MAX_PATTERN_RANK）；轴类型由下标奇偶定
    int64_t axisShape[MAX_PATTERN_RANK];  // 合轴后每根轴 size（元素；未用位填 1）
    int64_t axisStride[MAX_PATTERN_RANK]; // 每根轴 GM stride（按 element；未用位填 0）

    // ─── 多核切分（外层 A loop 扁平为线性计数，按 coreNum 大小核均衡）───
    int64_t aLoopCntTotal;     // ∏(outer A 整根) × aSplitChunkCnt（row-major）
    int64_t aSplitChunkCnt;    // CeilDiv(axisShape[aSplitIdx], aUbFactor)
    int64_t aBigCoreLoopCnt;   // 大核 aLoop 数
    int64_t aSmallCoreLoopCnt; // 小核 aLoop 数（floor；==0 → 仅前 aBigCoreCnt 核工作）
    int32_t aBigCoreCnt;       // 大核个数（= aLoopCntTotal % coreNum）
    int32_t usedCoreNum;       // 实际使用核数（= SetBlockDim 值，≤ 物理 coreNum）

    // ─── UB 切分 ───
    int32_t aSplitIdx;       // UB 内被切的 A 轴下标（偶数）
    int32_t rSplitIdx;       // UB 内被切的 R 轴下标（奇数）
    int64_t aUbFactor;       // valid：A 维实际元素数（GM stride/partial 判定）
    int64_t rUbFactor;       // valid：R 维实际元素数
    int64_t rUbFactorAlign;  // padded：UB 行 stride（仅 tail-R+切尾轴+非对齐时 > rUbFactor）
    int64_t innerAProdAlign; // padded：含最内 burst-tail A 的 CeilAlign
    int64_t innerRProdAlign; // padded：含最内 burst-tail R 的 CeilAlign

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal; // ∏(外层 R 轴 size) × CeilDiv(axisShape[rSplitIdx], rUbFactor)

    // ─── UB buffer 字节数 ───
    int64_t preBufSize;     // pre 阶段单份 buffer 字节（2D 含 R，按 maxDtypeSize），block 对齐
    int64_t postBufSize;    // post 阶段单份 buffer 字节（base A-only 兜底；group 实用）
    int64_t cacheBufUbSize; // 固定 16×1024（二分缓存树容量；runtime 填 kCacheBufUbSize）

    // ─── group 模板专属（base 不读，填 0）───
    int64_t rGroupCnt; // Phase1 分组数 = Phase2 workspace R 维大小

    // ─── group workspace 维度（跨 kernel 共用 struct：与 base 22 字段布局逐字段一致）───
    int64_t aTotal; // ∏(所有 A 轴 axisShape)：A/通道轴线性元素总数

    // ─── ATTR 透传（eps 由 host 透传）───
    // rstd = 1/sqrt(batch_variance + epsilon)。host 从 attr epsilon（GetFloat(0)，
    // nullptr→默认 0.0001）读取并填此字段，kernel 逐用例读取，禁止硬编码默认值。
    // 追加在 22 字段末尾：不改动既有字段偏移（tiling UT 按字段名读取，容量充裕）。
    float epsilon;
};

// ===========================================================================
// (1.2) empty 独立 struct（BN3DTrainingUpdateGradEmptyTilingData，7 字段；
//       对齐 reduction ReduceEmptyTilingData）。
//       EMPTY_A / EMPTY_R 共用：EMPTY_A usedCoreNum=0、其余填 0；EMPTY_R 填切分产出。
// ===========================================================================
struct BN3DTrainingUpdateGradEmptyTilingData {
    // ─── 多核切分 ───
    int32_t usedCoreNum; // EMPTY_A: 0（所有核早退）；EMPTY_R: 切分算出的用核数（≤ 物理核数）
    int64_t aTotal;      // ∏(所有 A 轴 axisShape)；EMPTY_A 不读（填 0），EMPTY_R > 0
    int64_t aUbFactor;   // 单 chunk 的 a 元素数（UB 四约束取 min）；EMPTY_A 填 0
    int32_t aBigCoreCnt; // 大核个数（= aLoopCntTotal % coreNum）；EMPTY_A 填 0
    int64_t aBigCoreLoopCnt;   // 每大核 chunk 数；EMPTY_A 填 0
    int64_t aSmallCoreLoopCnt; // 每小核 chunk 数（= aLoopCntTotal / coreNum，floor）；EMPTY_A 填 0

    // ─── UB buffer ───
    int64_t postBufSize; // post 阶段单份 buffer 字节数（block 对齐）；EMPTY_A 填 0
};

// ===========================================================================
// LEGACY SCAFFOLD — retained UNCHANGED for the out-of-scope device kernel
// scaffold (bn3d_training_update_grad_*_kernel.h / _apt.cpp). Do NOT reuse for
// host tiling; the authoritative host/UT structs are the two above. Removing
// these would break the run-package device build (kernel task will migrate it).
// ===========================================================================
constexpr int64_t kMaxInputSlots = 3;
constexpr int64_t kMaxOutputSlots = 1;
constexpr int64_t kPhysNodes = 4;

struct SplitResult {
    int64_t axis;
    int64_t a_i;
    int64_t a_o;
    int64_t a_i_tail;
};

struct MultiCoreResult {
    int64_t num_cores;
    int64_t total_tiles;
    int64_t tiles_main;
    int64_t cores_tail;
};

template <int64_t kRank>
struct Bn3dTrainingUpdateGradTilingData {
    SplitResult split;
    MultiCoreResult multicore;
    int64_t rank;
    int64_t per_buf_bytes;
    int64_t per_buf_elems;
    int64_t max_bro_shape[kRank];
    int64_t num_inputs;
    int64_t num_outputs;
    int64_t has_bias;
    int64_t input_shapes[kMaxInputSlots][kRank];
    int64_t input_strides[kMaxInputSlots][kRank];
    int64_t output_shapes[kMaxOutputSlots][kRank];
    int64_t output_strides[kMaxOutputSlots][kRank];
};
