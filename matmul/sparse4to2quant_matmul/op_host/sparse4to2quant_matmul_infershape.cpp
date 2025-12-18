/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse4to2quant_matmul_infershape.cpp
 * \brief
 */
#include "common/op_host/matmul_common_infershape.h"

namespace {
constexpr size_t ND_DIM_NUM = 2UL;
constexpr int64_t UNKNOWN_DIM_NUM = -2L;
constexpr int64_t SPARSE_ATOMIC_SIZE = 8L;
static ge::graphStatus InferShapeForSparse4to2QuantMatmul(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
        CUBE_INNER_ERR_REPORT("Sparse4to2QuantMatmul", "Context of infershape is null"), return ge::GRAPH_FAILED);
    auto xShape = context->GetInputShape(0);
    auto weightShape = context->GetInputShape(1);
    auto yShape = context->GetOutputShape(0);
    auto opName = context->GetNodeName();

    OP_CHECK_IF(
        xShape == nullptr || weightShape == nullptr || yShape == nullptr,
        CUBE_INNER_ERR_REPORT(opName, "[InferShape] shape is null"), return ge::GRAPH_FAILED);

    if (Ops::NN::CheckIsUnknownDimNum(*xShape) || Ops::NN::CheckIsUnknownDimNum(*weightShape)) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(
        opName, "Input xShape: %s, weightShape: %s", Ops::Base::ToString(*xShape).c_str(),
        Ops::Base::ToString(*weightShape).c_str());

    if (xShape->GetDimNum() != ND_DIM_NUM || weightShape->GetDimNum() != ND_DIM_NUM) {
        CUBE_INNER_ERR_REPORT(
            opName, "Input x and sparseWeight shape only support two dims shape, actually is %zu and %zu.",
            xShape->GetDimNum(), weightShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // only support false true for x and sparseWeight
    int64_t mDim = xShape->GetDim(0);
    int64_t nDim = weightShape->GetDim(0);

    int64_t xKDim = xShape->GetDim(1);
    int64_t weightKDim = weightShape->GetDim(1);

    int64_t xKDimAlign = (xKDim + SPARSE_ATOMIC_SIZE - 1) / SPARSE_ATOMIC_SIZE * SPARSE_ATOMIC_SIZE;

    OP_CHECK_IF(
        xKDimAlign != weightKDim + weightKDim,
        CUBE_INNER_ERR_REPORT(
            opName, "Input x k dim (8-aligned) must be double sparseWeight k dim , but x is %ld, sparseWeight is %ld.",
            xKDimAlign, weightKDim),
        return ge::GRAPH_FAILED);

    yShape->SetDimNum(0);
    yShape->AppendDim(mDim);
    yShape->AppendDim(nDim);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sparse4to2QuantMatmul).InferShape(InferShapeForSparse4to2QuantMatmul);

} // namespace