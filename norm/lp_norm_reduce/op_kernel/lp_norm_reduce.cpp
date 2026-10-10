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
// LpNormReduce_package/op_kernel/lp_norm_reduce.cpp
// =============================================================================
//
// ROLE: Ascend C kernel entry point for LpNormReduce.
//   本文件按 tilingKey 三分发接入三套模板的真实计算链：Base（tilingKey=0）、
//   Group（tilingKey=1）、Empty（tilingKey=2）。
//   核函数签名与 docs/LpNormReduce/develop/proto.md「kernel 函数签名」声明严格一致：
//
//     template <bool isGroup, bool isEmptyTensor>
//     __global__ __aicore__ void lp_norm_reduce(GM_ADDR x, GM_ADDR y,
//                                              GM_ADDR workspace, GM_ADDR tiling)
//
//   - 模板参数 (isGroup, isEmptyTensor) 与 arch35/lp_norm_reduce_tiling_key.h 的
//     TPL 声明一一对应、顺序一致（TilingKey.md：isGroup 占 bit0、isEmptyTensor
//     占 bit1，位编码 base=0 / group=1 / empty=2）；
//   - 入参顺序 = 算子原型定义顺序（x / y）+ 末尾固定 workspace / tiling；
//   - dtype 不进 Key：框架按 REG_OP 输入名 x 以 DTYPE_X 编译期实例化
//     fp16 / fp32 / bf16 三档（公共原型 dtype 组合列表）。
//
//   分发状态：
//   - tilingKey=0（base）：真实 Kernel 计算链已落码
//     （arch35/lp_norm_reduce_base.h，据 design/Kernel.md §9 + DESIGN-BRANCH-0.md
//     §3–§5：CopyIn → PreElewise → 二分缓存树归约 → PostElewise → CopyOut）；
//   - tilingKey=1（group）：真实 Kernel 计算链已落码
//     （arch35/lp_norm_reduce_group.h，据 design/Kernel.md §9.8 + DESIGN-BRANCH-1.md
//     §3–§5：Phase1Process 局部归约写 workspace → SyncAll → Phase2Process RA 二次归约）；
//   - tilingKey=2（empty）：真实 Kernel 计算链已落码
//     （arch35/lp_norm_reduce_empty.h，据 design/Kernel.md §9.7 + DESIGN-BRANCH-2.md
//     §3–§5：EMPTY_A 全核早退 / EMPTY_R Duplicate 固化值 0 → V_MTE3 → CopyOut
//     循环，不搬入输入数据）。
//
// =============================================================================

#include "kernel_operator.h"                   // Ascend C kernel framework (AscendC:: namespace)
#include "arch35/lp_norm_reduce_tiling_data.h" // TilingData 汇总结构体（base/group + empty）
#include "arch35/lp_norm_reduce_tiling_key.h"  // ASCENDC_TPL 双 bool 声明（isGroup / isEmptyTensor）
#include "arch35/lp_norm_reduce_base.h"        // Base 模板 kernel 类（tilingKey=0）
#include "arch35/lp_norm_reduce_group.h"       // Group 模板 kernel 类（tilingKey=1）
#include "arch35/lp_norm_reduce_empty.h"       // Empty 模板 kernel 类（tilingKey=2）

// ===========================================================================
// __global__ __aicore__ void lp_norm_reduce(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
//
// LpNormReduce NPU kernel 入口。每个 AIV core 执行一次。
//
// Template parameters（与 TilingKey 字段一一对应、顺序一致）:
//   isGroup       —— tilingKey bit0：1 = group 分支（A×R 2D 分核 + 二次归约）
//   isEmptyTensor —— tilingKey bit1：1 = empty 分支（EMPTY_A / EMPTY_R 共用）
//
// Parameters (all GM_ADDR = global memory pointers on device):
//   x         —— 输入数据张量（fp16 / fp32 / bf16，ND，rank ∈ [0,8]）
//   y         —— 输出张量（y.dtype = x.dtype；reduce 后 shape）
//   workspace —— workspace 缓冲（group 分支二次归约用，经 GetUserWorkspace
//                换算到用户区；base/empty 不消费）
//   tiling    —— tiling 数据缓冲（LpNormReduceTilingData / LpNormReduceEmptyTilingData）
//
// 分发（TilingKey.md「TPL_SEL 组合表」）：
//   tilingKey=0：base  （isGroup=false, isEmptyTensor=false）
//   tilingKey=1：group （isGroup=true,  isEmptyTensor=false）
//   tilingKey=2：empty （isGroup=false, isEmptyTensor=true）
//   （(1,1) 互斥不实例化，tilingKey=3 为设计性空位）
// ===========================================================================
template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void lp_norm_reduce(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // REGISTER_NONE_TILING: 手动经 GET_TILING_DATA_WITH_STRUCT 读取 tiling
    // （框架不自动展开 tiling 结构，保留完整控制权）
    REGISTER_NONE_TILING;

    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY):
    //   本算子为纯 Vector 归约，仅在 AIV（AI Vector）核上运行
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TPipe 统一在入口申请、以指针传入各模板（design/Kernel.md §1 入口代码）
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        // tilingKey=2：Empty 模板（EMPTY_A/EMPTY_R 共用，kernel 内按 usedCoreNum 区分）
        // ——真实计算链（arch35/lp_norm_reduce_empty.h：EMPTY_A 全核早退 /
        // EMPTY_R Duplicate 固化值 0 → V_MTE3 → CopyOut 循环，据 design/Kernel.md
        // §9.7 + DESIGN-BRANCH-2.md §3–§5；不搬入输入数据——empty 不做 reduce，
        // 只 Duplicate 固化值）。
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(LpNormReduceEmptyTilingData, tilingData, tiling);
        NsLpNormReduce::LpNormReduceEmptyKernel<DTYPE_X> op;
        op.Init(y, &tilingData, &pipe); // 空 tensor 不读 x：仅绑输出 y（Kernel.md §1 入口三形态）
        op.Process();
    } else if constexpr (isGroup) {
        // tilingKey=1：Group 模板（Phase 1 部分归约写 workspace → SyncAll → Phase 2 二次归约）
        // ——真实计算链（arch35/lp_norm_reduce_group.h：Phase1Process → SyncAll →
        // Phase2Process，据 design/Kernel.md §9.8 + DESIGN-BRANCH-1.md §3–§5）。
        // ⚠ 任务类型：group 分支含 SyncAll 硬同步，须 MIX_AIV_1_0（纯 AIV 计算、
        // 1:0 任务比）保证参与核同时驻留（bn3d_training_reduce split-reduce /
        // reduce_mean_variance 同款；该 instantiation 的 SEL 条目已在
        // lp_norm_reduce_tiling_key.h 携带 ASCENDC_TPL_KERNEL_TYPE_SEL(MIX_AIV_1_0)，
        // 此处 default 为 SEL 未覆盖时的兜底）。
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        GET_TILING_DATA_WITH_STRUCT(LpNormReduceTilingData, tilingData, tiling);
        // ⚠ 用户 workspace：950 运行时在 workspace 头部保留 16MB 系统区
        // （RESERVED_WORKSPACE = 16MB，与 host ws[0] 的 sysWorkspaceSize
        // （GetLibApiWorkSpaceSize，16MB）对应——ffts 跨核同步 mailbox 即在该区），
        // GetUserWorkspace 换算到用户区起始（DESIGN-BRANCH-1.md §8「框架分配的
        // 独立用户区」；bn3d_training_reduce / gn_training_reduce 同款）——直绑
        // 原始地址会踩 SyncAll 信箱。
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        NsLpNormReduce::LpNormReduceGroupKernel<DTYPE_X> op;
        op.InitGroup(x, y, userWorkspace, &tilingData, &pipe);
        op.ProcessGroup();
    } else {
        // tilingKey=0：Base 模板（aLoop 外层 × rChunk 内层 + Phase A/B）——真实计算链
        // （arch35/lp_norm_reduce_base.h：CopyIn → PreElewise → 二分缓存树归约 →
        // PostElewise → CopyOut，据 design/Kernel.md §9 + DESIGN-BRANCH-0.md §3–§5）
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(LpNormReduceTilingData, tilingData, tiling);
        NsLpNormReduce::LpNormReduceBaseKernel<DTYPE_X> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
    // base / empty 无用户 workspace（base：DESIGN-BRANCH-0.md §8 kernel 侧不绑定
    // workspace；empty：DESIGN-BRANCH-2.md §8 用户 workspace 无（usrSize=0），kernel
    // 全程只触 GM y（EMPTY_R）或零访问（EMPTY_A））——入口签名仍含 workspace GM
    // 地址，该两路不消费；group 路径经 GetUserWorkspace 消费（见上）。
    (void)workspace;
}
