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
 * \file mx_to_block_mx_quant_infershape.cpp
 * \brief InferShape and InferDataType implementation for MxToBlockMxQuant operator.
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {

constexpr size_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t BLOCK_SIZE = 32;
constexpr size_t MIN_DIM_NUM = 2;
constexpr size_t MAX_DIM_NUM = 3;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN};

graphStatus InferShapeForMxToBlockMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForMxToBlockMxQuant");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* mxscaleShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* scale1Shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1Shape);
    gert::Shape* scale2Shape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2Shape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set outputs to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scale1Shape);
        Ops::Base::SetUnknownRank(*scale2Shape);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape->GetDimNum() < MIN_DIM_NUM || xShape->GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [2, 3].", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // Output y has the same shape as input x
    *yShape = *xShape;

    // Output scale1 has the same shape as input mxscale
    *scale1Shape = *mxscaleShape;

    // scale2
    int64_t lastDim = xShape->GetDim(xShape->GetDimNum() - DIGIT_ONE);
    int64_t secondLastDim = xShape->GetDim(xShape->GetDimNum() - DIGIT_TWO);
    int64_t scale2Dim3 = ((Ops::Base::CeilDiv(secondLastDim, BLOCK_SIZE) + DIGIT_ONE) / DIGIT_TWO) * DIGIT_TWO /
                         DIGIT_TWO;
    int64_t scale2Dim2 = lastDim;

    scale2Shape->SetDimNum(0);
    if (xShape->GetDimNum() == DIGIT_THREE) {
        scale2Shape->AppendDim(xShape->GetDim(0));
    }
    scale2Shape->AppendDim(scale2Dim3);
    scale2Shape->AppendDim(scale2Dim2);
    scale2Shape->AppendDim(DIGIT_TWO);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForMxToBlockMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForMxToBlockMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForMxToBlockMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(
            context->GetNodeName(),
            "dst_type is illegal, only supports 35(FLOAT8_E5M2) or 36(FLOAT8_E4M3FN), but got %ld(%s), please check.",
            *dstDtype, ge::TypeUtils::DataTypeToAscendString(outDtype).GetString()),
        return ge::GRAPH_FAILED);
    // y dtype is determined by dst_type attribute
    context->SetOutputDataType(0, outDtype);
    // scale1 and scale2 are always FLOAT8_E8M0
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(2, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForMxToBlockMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MxToBlockMxQuant)
    .InferShape(InferShapeForMxToBlockMxQuant)
    .InferDataType(InferDataTypeForMxToBlockMxQuant);
} // namespace ops
