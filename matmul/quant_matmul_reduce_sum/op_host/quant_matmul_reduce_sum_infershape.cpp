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
 * \file quant_matmul_reduce_sum_infershape.cpp
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"

using namespace gert;
namespace {
const size_t X1_INDEX = 0;
const size_t X2_INDEX = 1;
const size_t OUTPUT_DIM = 2;
const size_t X1_DIM_NUM = 3;

static ge::graphStatus InferShapeQuantMatmulReduceSum(InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("QuantMatmulReduceSum", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeX1 = context->GetInputShape(X1_INDEX);
    auto shapeX2 = context->GetInputShape(X2_INDEX);
    auto shapeOut = context->GetOutputShape(0);
    if (shapeX1 == nullptr || shapeX2 == nullptr || shapeOut == nullptr) {
        OP_LOGE(opName, "[InferShape] shape or attrs is null");
        return ge::GRAPH_FAILED;
    }

    // x1 and x2
    int64_t dimX1 = shapeX1->GetDimNum();
    int64_t dimX2 = shapeX2->GetDimNum();
    if (dimX1 != X1_DIM_NUM || dimX2 != X1_DIM_NUM) {
        OP_LOGE(opName, "[InferShape] x1_dim(%ld) and x2_dim(%ld) is not 3.", dimX1, dimX2);
        return ge::GRAPH_FAILED;
    }
    int64_t bX1 = shapeX1->GetDim(0);
    int64_t mX1 = shapeX1->GetDim(1);
    int64_t kX1 = shapeX1->GetDim(2);
    int64_t bX2 = shapeX2->GetDim(0);
    int64_t kX2 = shapeX2->GetDim(1);
    int64_t nX2 = shapeX2->GetDim(2);
    if (bX1 != bX2) {
        OP_LOGE(opName, "[InferShape] B_x1(%ld) must equal to B_x2(%ld).", bX1, bX2);
        return ge::GRAPH_FAILED;
    }
    if (kX1 != kX2) {
        OP_LOGE(opName, "[InferShape] K_x1(%ld) must equal to K_x2(%ld).", kX1, kX2);
        return ge::GRAPH_FAILED;
    }

    // output shape
    shapeOut->SetDimNum(OUTPUT_DIM);
    shapeOut->SetDim(0, mX1);
    shapeOut->SetDim(1, nX2);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantMatmulReduceSum(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace ops {
IMPL_OP_INFERSHAPE(QuantMatmulReduceSum)
    .InferShape(InferShapeQuantMatmulReduceSum)
    .InferDataType(InferDataTypeQuantMatmulReduceSum);
}