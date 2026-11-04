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
 * \file mat_mul_infershape.cpp
 * \brief MatMul infershape: 复用公共batch matmul推导(1D-8D/batch广播/bias N校验)
 */
#include "common/op_host/matmul_common_infershape.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t MATMUL_BIAS_IDX = 2;
static ge::graphStatus InferShapeForMatMul(gert::InferShapeContext* context)
{
    // MatMul支持1D/2D输入(1D由公共库升维处理);拒绝3D+(那是bmm的场景)
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    if (shape_x1 == nullptr || shape_x2 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (shape_x1->GetDimNum() > 2 || shape_x2->GetDimNum() > 2) {
        OP_LOGE(context->GetNodeName(), "inputs of matmul must be 1D or 2D, use batch_mat_mul for higher dims");
        return ge::GRAPH_FAILED;
    }
    return Ops::NN::InferShapeForBatchMatMul(context, 0, MATMUL_BIAS_IDX);
}
} // namespace

IMPL_OP(MatMul).InferShape(InferShapeForMatMul);
} // namespace ops
