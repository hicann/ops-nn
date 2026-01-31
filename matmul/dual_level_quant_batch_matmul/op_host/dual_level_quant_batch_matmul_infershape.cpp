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
 * \file dual_level_quant_batch_matmul_infershape.cpp
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

namespace {
constexpr size_t FIRST_DIM = 0;
constexpr size_t SECOND_DIM = 1;
constexpr size_t SHAPE_SIZE = 2UL;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
const char* NODE_NAME = "DualLevelQuantBatchMatmul";

bool CheckIsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

static ge::graphStatus ShapeCheckAndInfer(
    const gert::Shape* x1Shape, const gert::Shape* x2Shape, gert::Shape* yShape,
    const bool transposeX1, const bool transposeX2)
{
    size_t numDimA = x1Shape->GetDimNum();
    size_t numDimB = x2Shape->GetDimNum();
    OP_CHECK_IF(
        numDimA != SHAPE_SIZE || numDimB != SHAPE_SIZE,
        OP_LOGE(
            NODE_NAME, "The shape of x1 and x2 must be 2, which are %s and %s",
            Ops::Base::ToString(*x1Shape).c_str(), Ops::Base::ToString(*x2Shape).c_str()),
        return ge::GRAPH_FAILED);
    size_t mIdx = transposeX1 ? SECOND_DIM : FIRST_DIM;
    size_t kaIdx = transposeX1 ? FIRST_DIM : SECOND_DIM;
    size_t kbIdx = transposeX2 ? SECOND_DIM : FIRST_DIM;
    size_t nIdx = transposeX2 ? FIRST_DIM : SECOND_DIM;
    int64_t x1M = x1Shape->GetDim(mIdx);
    int64_t x1K = x1Shape->GetDim(kaIdx);
    int64_t x2K = x2Shape->GetDim(kbIdx);
    int64_t x2N = x2Shape->GetDim(nIdx);

    OP_CHECK_IF(
        x1K != x2K && x1K > 0 && x2K > 0, OP_LOGE(NODE_NAME, "Ka[%ld] != Kb[%ld]", x1K, x2K),
        return ge::GRAPH_FAILED);
    size_t outDimNum = std::max(numDimA, numDimB);
    yShape->SetDimNum(outDimNum);
    yShape->SetDim(FIRST_DIM, x1M);
    yShape->SetDim(SECOND_DIM, x2N);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForDualLevelQuantBatchMatmul(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShapeForDualLevelQuantBatchMatmul begin");
    auto x1Shape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x2Shape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* transposeX1 = attrs->GetAttrPointer<bool>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeX1);
    const bool* transposeX2 = attrs->GetAttrPointer<bool>(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeX2);
    if (CheckIsUnknownDimNum(*x1Shape) || CheckIsUnknownDimNum(*x2Shape)) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(
        context->GetNodeName(),
        "x1_shape: %s, x2_shape: %s, transpose_x1: %d, transpose_x2: %d",
        Ops::Base::ToString(*x1Shape).c_str(), Ops::Base::ToString(*x2Shape).c_str(), *transposeX1, *transposeX2);

    OP_CHECK_IF(
        ShapeCheckAndInfer(x1Shape, x2Shape, yShape, *transposeX1, *transposeX2) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The Shape Check failed "), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(DualLevelQuantBatchMatmul).InferShape(InferShapeForDualLevelQuantBatchMatmul);
}