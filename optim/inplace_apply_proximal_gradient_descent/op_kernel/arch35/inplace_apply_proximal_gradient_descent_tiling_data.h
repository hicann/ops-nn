/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * inplace_apply_proximal_gradient_descent_package/op_kernel/arch35/inplace_apply_proximal_gradient_descent_tiling_data.h
 * =============================================================================
 * Role: DESIGN §7 的公共 TilingData 汇总结构体。六个计算变体（三个 dtype
 *       外层 binary × runtime key 0/1）共享同一结构体，Host 侧 typed
 *       `GetTilingData<T>()`（§9.7）与 Kernel 侧
 *       `GET_TILING_DATA_WITH_STRUCT`（§10.1）共用的同一 POD 类型。
 *
 * Contents:
 *   - InplaceApplyProximalGradientDescentTilingData：§7 表列的 10 个字段（元素数 / 核数）。
 *   - static_assert<std::is_standard_layout_v>：standard-layout 布局保证。
 *   - static_assert<std::is_trivially_copyable_v>：trivially-copyable 序列化保证。
 *
 * 字段规则（DESIGN §7）：只携带六个计算变体共同的线性切分量；不包含 dtype、key、
 * buffer-mode、elemBytes、computeType 等类型/模式私有字段。dtype 由 `DTYPE_VAR`
 * 注入，`BUFFER_MODE` 由 §6 模板选择表达，均不在 TilingData 中重复存储。
 * =============================================================================
 */

#ifndef INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_DATA_H_
#define INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_DATA_H_

#include <cstdint>
#include <type_traits>

struct InplaceApplyProximalGradientDescentTilingData {
    int64_t dim0 = 0;                // 展平后的 var/delta/output 有效元素总数
    int32_t usedCoreNum = 0;         // 实际启动的 AIV 核数；空 Tensor 为 1 并立即短路
    int32_t reserved = 0;            // 固定为 0，保持后续 int64_t 字段自然对齐
    int64_t blockFactor = 0;         // 非尾核处理的基础元素数
    int64_t blockTail = 0;           // 最后一个有效核处理的元素数
    int64_t ubFactor = 0;            // 每个完整 UB 块的有效元素数，按 256B 向下对齐
    int64_t ubLoopOfFormerBlock = 0; // 非尾核 UB tile 循环数
    int64_t ubTailOfFormerBlock = 0; // 非尾核最后一个 UB tile 的有效元素数
    int64_t ubLoopOfTailBlock = 0;   // 尾核 UB tile 循环数
    int64_t ubTailOfTailBlock = 0;   // 尾核最后一个 UB tile 的有效元素数
};

static_assert(std::is_standard_layout_v<InplaceApplyProximalGradientDescentTilingData>,
              "InplaceApplyProximalGradientDescentTilingData must be standard-layout POD");
static_assert(std::is_trivially_copyable_v<InplaceApplyProximalGradientDescentTilingData>,
              "InplaceApplyProximalGradientDescentTilingData must be trivially copyable POD");

#endif
