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
 * \file thnn_fused_gru_cell.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_THNN_FUSED_GRU_CELL_H_
#define OP_API_INC_LEVEL0_THNN_FUSED_GRU_CELL_H_

#include <array>

#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

namespace l0op {

// 门控列数：input_gates / hidden_gates = 3H（门序 r, z, n）；storage = 5H（rg/zg/ng/hn/残余 5 段）
constexpr int64_t GATES_PER_INPUT = 3;
constexpr int64_t GATES_PER_STORAGE = 5;

// 共享输出 shape 推导：hy.shape = hx.shape = (B, H)、storage.shape = (B, 5H)。
// L0 分配与 L2 CheckShape 共用同一 helper，返回 false 表示 hx 为空或非 rank-2
inline bool ThnnFusedGruCellOutShape(const aclTensor* hx, op::Shape& hyShape, op::Shape& storageShape)
{
    if (hx == nullptr || hx->GetViewShape().GetDimNum() != 2U) {
        return false;
    }
    const int64_t batch = hx->GetViewShape().GetDim(0);
    const int64_t hidden = hx->GetViewShape().GetDim(1);
    hyShape.SetDimNum(0);
    hyShape.AppendDim(batch);
    hyShape.AppendDim(hidden);
    storageShape.SetDimNum(0);
    storageShape.AppendDim(batch);
    storageShape.AppendDim(GATES_PER_STORAGE * hidden);
    return true;
}

// 加入 AiCore launch list；可选 bias 空指针原样转发（缺省 ≡ 全零）。
// 返回内部分配的连续输出 {hy, storage}，失败返回空数组
std::array<const aclTensor*, 2> ThnnFusedGruCell(const aclTensor* inputGates, const aclTensor* hiddenGates,
                                                 const aclTensor* hx, const aclTensor* inputBias,
                                                 const aclTensor* hiddenBias, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_THNN_FUSED_GRU_CELL_H_
