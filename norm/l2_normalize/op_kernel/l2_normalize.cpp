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
// norm/l2_normalize/op_kernel/l2_normalize.cpp
// =============================================================================
//
// ROLE: Ascend C kernel entry point for L2Normalize (Ascend 950 / arch35 only).
//   核函数签名：
//     template <bool isGroup, bool isEmptyTensor>
//     __global__ __aicore__ void l2_normalize(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
//   - 形参顺序 = 算子原型 tensor 定义顺序（x, y）+ 末尾固定 workspace + tiling；
//     axis / eps 均为 attr，非 tensor 输入，无对应 GM 形参。
//   - 模板参数为 TilingKey 双 bool 轴（isGroup / isEmptyTensor）；
//     dtype 走编译期 DTYPE_X 实例化（每 dtype 组合一个 binary），不走 TilingKey 运行时分支。
//   - base 模板处理通用非空输入；empty 模板对空 tensor 全核早退；group 模板以
//     A×R 二维分核执行三阶段计算：partial 写 workspace → SyncAll → RA 二次归约
//     得 denom → keepdims 广播除法。
// =============================================================================

#include "kernel_operator.h"                   // Ascend C kernel framework (AscendC:: namespace)
#include "arch35/l2_normalize_tiling_struct.h" // L2NormalizeTilingData / L2NormalizeEmptyTilingData
#include "arch35/l2_normalize_struct.h"        // TPL 声明（isGroup / isEmptyTensor 双 bool 轴）
#include "arch35/l2_normalize_base.h"          // Base 模板 kernel 类（TPL_SEL_0）
#include "arch35/l2_normalize_empty.h"         // Empty 模板 kernel 类（TPL_SEL_1）
#include "arch35/l2_normalize_group.h"         // Group 模板 kernel 类（TPL_SEL_2）

// ===========================================================================
// __global__ __aicore__ void l2_normalize(x, y, workspace, tiling)
//
// This is the main NPU kernel function. Each AIV core executes this once.
//
// Template parameters (TilingKey 双 bool 轴):
//   isGroup       — group 模板（A 用不满核且 R 有并行度 → A×R 2D 分核 + workspace 二次归约）
//   isEmptyTensor — 空 tensor 模板（某维为 0 → 全核早退，零计算零 IO）
//
// Parameters (all GM_ADDR = global memory pointers on device):
//   x         — input data tensor
//   y         — output tensor（shape/dtype 与 x 一致）
//   workspace — workspace buffer（base R 切分两遍 denom 中转 / group 二次归约用）
//   tiling    — tiling data buffer（empty → L2NormalizeEmptyTilingData；其余 → L2NormalizeTilingData）
// ===========================================================================
template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void l2_normalize(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // REGISTER_NONE_TILING: Registers the kernel with the AICore runtime.
    // The tiling data is read manually via GET_TILING_DATA_WITH_STRUCT.
    REGISTER_NONE_TILING;

    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY):
    //   Sets this kernel to run on AIV (AI Vector) cores only.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TPipe 统一在入口申请，经 Init 指针传入各模板。
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        // 空 tensor 模板（TPL_SEL_1）：EMPTY_A / EMPTY_R 合一全核早退——
        // 零计算零 IO（不绑 x/y GM）、零 TBuf、零同步。
        GET_TILING_DATA_WITH_STRUCT(L2NormalizeEmptyTilingData, tilingData, tiling);
        NsL2Normalize::L2NormalizeEmptyKernel<DTYPE_X> op;
        op.Init(&tilingData, &pipe); // 仅缓存 TilingData/TPipe（零 IO：不传 GM 形参）
        op.Process();                // usedCoreNum=0 → 进入即 return（全核早退）
    } else if constexpr (isGroup) {
        // group 模板（TPL_SEL_2）：A×R 2D 分核三阶段——
        // Phase 1 各核 [1 A chunk × 1 段 R 分组] 局部平方和写 ws partial 区
        // [rGroupCnt, aTotal] → SyncAll 全核同步（SetScheduleMode(1) 配套）→
        // Phase 2 RA mini-kernel 二次归约 + eps 钳制 + sqrt 得 denom 写 ws
        // denom 区本核槽位 → Phase 3 keepdims 广播除法（A 切分沿用 Phase 2
        // 槽位自产自销，无第二次 SyncAll；tail-R / tail-A 运行时子路径）。
        GET_TILING_DATA_WITH_STRUCT(L2NormalizeTilingData, tilingData, tiling);
        NsL2Normalize::L2NormalizeGroupKernel<DTYPE_X> op;
        op.InitGroup(x, y, workspace, &tilingData, &pipe);
        op.ProcessGroup();
    } else {
        // base 模板（TPL_SEL_0）：非空输入兜底分支——A 方向分核、核间零依赖
        GET_TILING_DATA_WITH_STRUCT(L2NormalizeTilingData, tilingData, tiling);
        NsL2Normalize::L2NormalizeBaseKernel<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
}
