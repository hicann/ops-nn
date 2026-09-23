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
// LpNormReduce_package/op_kernel/arch35/lp_norm_reduce_tiling_key.h
// =============================================================================
//
// ROLE: LpNormReduce TPL（模板化 TilingKey）声明 —— reduction 范式双 bool 轴。
//   据 docs/LpNormReduce/design/TilingKey.md §1：
//   - 声明序 isGroup 占 bit0、isEmptyTensor 占 bit1（BOOL 各 1 bit）；
//   - GET_TPL_TILING_KEY 实际位编码：base=0、group=1、empty=2；
//   - (isGroup=1, isEmptyTensor=1) 互斥不实例化（tilingKey=3 为设计性空位）；
//   - dtype 不进 key（框架按输入名 x 以 DTYPE_X 编译期实例化 fp16/fp32/bf16 三档）、
//     p / axisNum / keepdim / epsilon 等纯参数差异不拆 key。
//   模板参数与 op_kernel/lp_norm_reduce_apt.cpp 的 kernel 入口签名
//   template <bool isGroup, bool isEmptyTensor> 一一对应、顺序一致。
//
// =============================================================================

#ifndef LP_NORM_REDUCE_TILING_KEY_H_
#define LP_NORM_REDUCE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

// ---------------------------------------------------------------------------
// ASCENDC_TPL_ARGS_DECL — 声明编译期模板参数（host 侧供 GET_TPL_TILING_KEY 编码，
// kernel 侧经 ASCENDC_TPL_PRE 预处理导出 @@ 标记供 codegen 实例化）
// ---------------------------------------------------------------------------
ASCENDC_TPL_ARGS_DECL(LpNormReduce, ASCENDC_TPL_BOOL_DECL(isGroup, 0, 1), // 0=base/empty, 1=group（A×R 2D 分核）
                      ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1) // 0=非空, 1=空 tensor（EMPTY_A/EMPTY_R 共用）
);

// ---------------------------------------------------------------------------
// ASCENDC_TPL_SEL — 实例化组合表（3 组合 × 2 档 dtype = 6 份 binary，见 TilingKey.md）
// SEL 块顺序（base→empty→group）只影响 codegen 实例化遍历顺序，runtime 永远按
// tilingKey 位编码值直接选符号。
//
// 每条 SEL 携带 ASCENDC_TPL_KERNEL_TYPE_SEL（kernel 任务类型，atvoss reduce 框架
// reduce_tiling_key_decl.h 同款机制）：group（tilingKey=1）为 SyncAll 硬同步分支，
// 须 MIX_AIV_1_0（纯 AIV 计算、1:0 任务比——保证 SyncAll 参与核同时驻留，
// bn3d_training_reduce split-reduce / reduce_mean_variance / atvoss reduce
// *_GROUP SEL 同款）；base / empty 无核间依赖保持 AIV_ONLY。
// ---------------------------------------------------------------------------
ASCENDC_TPL_SEL(
    // tilingKey=0：base（isGroup=0, isEmptyTensor=0）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(isGroup, 0),
                         ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
    // tilingKey=2：empty（isGroup=0, isEmptyTensor=1；EMPTY_A/EMPTY_R 共用同一 binary）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(isGroup, 0),
                         ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
    // tilingKey=1：group（isGroup=1, isEmptyTensor=0）——SyncAll 硬同步 → MIX_AIV_1_0
    // （isGroup=1 与 isEmptyTensor=1 互斥：(1,1) 无 SEL 块，tilingKey=3 为设计性空位）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(isGroup, 1),
                         ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif // LP_NORM_REDUCE_TILING_KEY_H_
