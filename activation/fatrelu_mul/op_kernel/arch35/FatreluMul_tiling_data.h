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
// fatrelu_mul_package/op_kernel/arch35/FatreluMul_tiling_data.h
// =============================================================================
//
// ROLE: FatreluMul 全局 TilingData 结构体定义（host / kernel 共享同构布局）。
//   双路径（small-tail 行打包 / big-tail 行分段，见 docs/fatrelu_mul/design/TilingKey.md）
//   共用同一个非模板化 TilingData 结构体：两路径仅 CopyIn / CopyOut / 行循环组织
//   不同，行模型展开量、多核切分量与 UB 切分量完全同构，分支差异由「路径不使用
//   的字段置 0」承载，不设分支私有结构体（docs/fatrelu_mul/design/TilingData.md）。
//   host 侧（fatrelu_mul_tiling_arch35.cpp TilingFunc）填充；kernel 侧
//   （fatrelu_mul_apt.cpp）经 GET_TILING_DATA_WITH_STRUCT(FatreluMulTilingData,
//   td, tiling) 消费（docs/fatrelu_mul/design/DevView.md「工程目录结构」）。
//
// CONTENTS:
//   - kMaxInputSlots = 2（x、threshold）/ kMaxOutputSlots = 1（y）输入输出槽位常量
//   - FatreluMulTilingData：batch_size / half_dim / need_core_num / rows_former /
//     rows_tail_core / tile_elems / rows_per_group —— 与
//     docs/fatrelu_mul/design/TilingData.md 逐字段一致，不增不删
//     （字段单位见各字段注释：元素数 / 核数；无 kPhysNodes 共享常量，见「范式特有约束」）
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FatreluMul   — TilingData 结构名前缀（FatreluMulTilingData）
//   文件名前缀    : FatreluMul_  — DevView.md「命名规范」（TilingData.md 落位约定）
//   UPPER        : FATRELUMUL   — （本文件未直接使用）
//
// =============================================================================

// FatreluMul TilingData —— 行模型：x 展平为 (batch_size, 2*half_dim) 连续矩阵，y 为 (batch_size, half_dim)
// host 侧（FatreluMul_tiling_arch35.cpp TilingFunc）填充，kernel 侧（FatreluMul_kernel.h）消费，两侧同构布局
#pragma once
#include <cstdint>

// 输入 / 输出槽位数：用于 kernel 入口 GM 指针数组绑定（ins/outs）与注册声明
// threshold 为单元素标量 Tensor，不占 UB buffer 槽位（GM 直读，见「范式特有约束」）
constexpr int64_t kMaxInputSlots = 2;  // 输入：x、threshold
constexpr int64_t kMaxOutputSlots = 1; // 输出：y

struct FatreluMulTilingData {
    // —— 行模型展开量（Host Tiling「输入预处理」步骤算出）——
    int64_t batch_size; // M = numel(x) / lastDim(x)：行数（= y 行数；其余维乘积，含 batch=0 空 Tensor 场景）
    int64_t half_dim; // d = lastDim(x) / 2：gate / up 半区长度（= y 末维；末维奇数已在 aclnn 层拒绝，d ≥ 0 整数）

    // —— 多核切分（行域，Host Tiling「多核切分」步骤算出）——
    int64_t need_core_num; // 参与计算的核数（= SetBlockDim 值；空 Tensor 置 1）
    int64_t rows_former;   // 每核基础行数 = batch_size / need_core_num（整除部分；空 Tensor 置 0）
    int64_t rows_tail_core; // 尾核数 = batch_size % need_core_num（前 rows_tail_core 个核各多处理 1 行；整除时为 0）

    // —— UB 切分（Host Tiling「UB 切分」步骤算出）——
    int64_t tile_elems; // 单 tile 元素容量：按 UB 预算 ÷ per-dtype 每元素 buffer 深度反推，按 256B 对齐
                        // （对齐因子 = 256 / sizeof(T) 元素；dtype 相关，判界 d vs tileElems 决定 tilingKey）
    int64_t rows_per_group; // small-tail 行打包每组的行数 = tile_elems / half_dim 向下取整（≥ 1）；
                            // 仅 tilingKey=0 路径使用；big-tail 路径与空 Tensor 短路场景置 0
};
