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
 * \file dynamic_mx_quant_infershape.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr size_t INDEX_ATTR_AXIS = 0;
constexpr size_t INDEX_ATTR_DST_TYPE = 2;
constexpr size_t INDEX_ATTR_BLOCK_SIZE = 3;
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;
static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

graphStatus InferShapeForDynamicMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForDynamicMxQuant");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* scaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    OP_CHECK_IF(
        xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [1, 7].", xShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scaleShape);
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* axis = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    size_t dim = (*axis >= 0) ? static_cast<size_t>(*axis) : static_cast<size_t>(*axis + xShape->GetDimNum());
    const int32_t* blockSize = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);

    int64_t dimSize = 0;
    if (xShape->GetDim(dim) == UNKNOWN_DIM_VALUE_) {
        dimSize = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize = Ops::Base::CeilDiv(xShape->GetDim(dim), static_cast<int64_t>(*blockSize));
        dimSize = (dimSize + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape = *xShape;
    scaleShape->SetDim(dim, dimSize);
    scaleShape->AppendDim(ALIGN_NUM);

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, mxscale shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*scaleShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForDynamicMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForDynamicMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForDynamicMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* dstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(
            context->GetNodeName(),
            "dst_type is illegal, only supports 40(FLOAT4_E2M1), 41(FLOAT4_E1M2), "
            "36(FLOAT8_E4M3FN) or 35(FLOAT8_E5M2). but got %d(%s)please check.",
            *dstDtype, ge::TypeUtils::DataTypeToAscendString(outDtype).GetString()),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, outDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForDynamicMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicMxQuant)
    .InferShape(InferShapeForDynamicMxQuant)
    .InferDataType(InferDataTypeForDynamicMxQuant);
} // namespace ops
