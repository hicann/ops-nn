/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRU_BLOCK_CELL_GRAD_INFER_COMMON_H_
#define GRU_BLOCK_CELL_GRAD_INFER_COMMON_H_
#include "graph/types.h"
#include "op_common/log/log.h"
namespace ops {
namespace gru_block_cell_grad {
constexpr size_t kNumInputs = 10;
constexpr size_t kNumOutputs = 4;

// Input order: x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h
constexpr size_t kInX = 0;
constexpr size_t kInHPrev = 1;
constexpr size_t kInWRu = 2;
constexpr size_t kInWC = 3;
constexpr size_t kInBRu = 4;
constexpr size_t kInBC = 5;
constexpr size_t kInR = 6;
constexpr size_t kInU = 7;
constexpr size_t kInC = 8;
constexpr size_t kInDH = 9;

// Output order: d_x, d_h_prev, d_c_bar, d_r_bar_u_bar
constexpr size_t kOutDX = 0;
constexpr size_t kOutDHPrev = 1;
constexpr size_t kOutDCBar = 2;
constexpr size_t kOutDRub = 3;

// Expected ranks: 8 matrix inputs rank 2, b_ru/b_c rank 1; all 4 outputs rank 2.
constexpr size_t kInputRank[kNumInputs] = {2, 2, 2, 2, 1, 1, 2, 2, 2, 2};
constexpr size_t kOutputRank = 2;

constexpr const char* kInputNames[kNumInputs] = {"x", "h_prev", "w_ru", "w_c", "b_ru", "b_c", "r", "u", "c", "d_h"};
constexpr const char* kOutputNames[kNumOutputs] = {"d_x", "d_h_prev", "d_c_bar", "d_r_bar_u_bar"};

template <typename ContextT>
bool InputTensorDescsAreLegal(ContextT* context, const char* node)
{
    for (size_t i = 0; i < kNumInputs; ++i) {
        const auto* desc = context->GetInputDesc(i);
        if (desc == nullptr) {
            OP_LOGE(node, "input %s has no tensor desc", kInputNames[i]);
            return false;
        }
        if (desc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE(node, "input %s dtype must be float32", kInputNames[i]);
            return false;
        }
        if (desc->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE(node, "input %s format must be ND", kInputNames[i]);
            return false;
        }
    }
    return true;
}

template <typename ContextT>
bool TensorDescsAreLegal(ContextT* context, const char* node)
{
    if (!InputTensorDescsAreLegal(context, node)) {
        return false;
    }
    for (size_t i = 0; i < kNumOutputs; ++i) {
        const auto* desc = context->GetOutputDesc(i);
        if (desc == nullptr) {
            continue; // not instantiated: dtype comes from InferDataType
        }
        const ge::DataType dtype = desc->GetDataType();
        if (dtype != ge::DT_UNDEFINED && dtype != ge::DT_FLOAT) {
            OP_LOGE(node, "output %s dtype must be float32", kOutputNames[i]);
            return false;
        }
        if (dtype != ge::DT_UNDEFINED && desc->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE(node, "output %s format must be ND", kOutputNames[i]);
            return false;
        }
    }
    return true;
}

} // namespace gru_block_cell_grad
} // namespace ops
#endif
