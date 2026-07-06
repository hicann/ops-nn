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
 * \file anti_mx_quant_infershape.cpp
 * \brief InferShape and InferDataType implementation for AntiMxQuant operator.
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr size_t INDEX_ATTR_AXIS = 0;
constexpr size_t INDEX_ATTR_DST_TYPE = 1;
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;
constexpr size_t MIN_MXSCALE_DIM_NUM = 2;
constexpr size_t MAX_MXSCALE_DIM_NUM = 8;
constexpr int64_t BLOCK_SIZE = 32;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};

graphStatus InferShapeForAntiMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForAntiMxQuant");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* mxscaleShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    OP_CHECK_IF(xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [1, 7].", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    // Output y has the same shape as input x
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForAntiMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForAntiMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAntiMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* dstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(
            context->GetNodeName(),
            "dst_type is illegal, only supports 1(FLOAT16), 27(BFLOAT16) or 0(FLOAT32), but got %d(%s), please check.",
            *dstDtype, ge::TypeUtils::DataTypeToAscendString(outDtype).GetString()),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAntiMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AntiMxQuant).InferShape(InferShapeForAntiMxQuant).InferDataType(InferDataTypeForAntiMxQuant);
} // namespace ops
