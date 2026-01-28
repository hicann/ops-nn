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
 * \file dynamic_mx_quant_with_dual_axis_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"
#include "../../../foreach/foreach_utils/op_host/common_dtype.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_ATTR_DST_TYPE = 1;
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t Y1_OUTTYPE_INDEX = 0;
constexpr int64_t SCALE1_OUTTYPE_INDEX = 1;
constexpr int64_t Y2_OUTTYPE_INDEX = 2;
constexpr int64_t SCALE2_OUTTYPE_INDEX = 3;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

graphStatus InferShapeForDynamicMxQuantWithDualAxis(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForDynamicMxQuantWithDualAxis");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape1 = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape1);

    gert::Shape* scaleShape1 = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape1);

    gert::Shape* yShape2 = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape2);

    gert::Shape* scaleShape2 = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape2);

    std::string errMsg = optiling::ConcatString("Input x rank[", xShape->GetDimNum(), "] should be in [2, 7].");
    OP_CHECK_IF(
        xShape->GetDimNum() < 2 || xShape->GetDimNum() > MAX_DIM_NUM,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "%s", errMsg.c_str()), return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape1);
        Ops::Base::SetUnknownRank(*scaleShape1);
        Ops::Base::SetUnknownRank(*yShape2);
        Ops::Base::SetUnknownRank(*scaleShape2);
        return ge::GRAPH_SUCCESS;
    }

    *yShape1 = *xShape;
    *yShape2 = *xShape;

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    size_t dim1 = static_cast<size_t>(xShape->GetDimNum() - 1);

    int64_t dimSize1 = 0;
    if (xShape->GetDim(dim1) == UNKNOWN_DIM_VALUE_) {
        dimSize1 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize1 = Ops::Base::CeilDiv(xShape->GetDim(dim1), BLOCK_SIZE);
        dimSize1 = (dimSize1 + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape1 = *xShape;
    scaleShape1->SetDim(dim1, dimSize1);
    scaleShape1->AppendDim(ALIGN_NUM);

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, mxscale1 shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*scaleShape1).c_str());

    size_t dim2 = static_cast<size_t>(xShape->GetDimNum() - 2);
    int64_t dimSize2 = 0;
    if (xShape->GetDim(dim2) == UNKNOWN_DIM_VALUE_) {
        dimSize2 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize2 = Ops::Base::CeilDiv(xShape->GetDim(dim2), BLOCK_SIZE);
        dimSize2 = (dimSize2 + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape2 = *xShape;
    scaleShape2->SetDim(dim2, dimSize2);
    scaleShape2->AppendDim(ALIGN_NUM);

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, mxscale2 shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*scaleShape1).c_str());

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForDynamicMxQuantWithDualAxis");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForDynamicMxQuantWithDualAxis(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForDynamicMxQuantWithDualAxis");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* dstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);

    std::string errMsg = optiling::ConcatString(
        "dst_type is illegal, only supports 40(FLOAT4_E2M1), 41(FLOAT4_E1M2), "
        "36(FLOAT8_E4M3FN) or 35(FLOAT8_E5M2). but got ",
        *dstDtype, "(", ge::TypeUtils::DataTypeToAscendString(outDtype).GetString(), ")", " please check.");

    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "%s", errMsg.c_str()), return ge::GRAPH_FAILED);
    context->SetOutputDataType(Y1_OUTTYPE_INDEX, outDtype);
    context->SetOutputDataType(SCALE1_OUTTYPE_INDEX, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(Y2_OUTTYPE_INDEX, outDtype);
    context->SetOutputDataType(SCALE2_OUTTYPE_INDEX, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForDynamicMxQuantWithDualAxis");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicMxQuantWithDualAxis)
    .InferShape(InferShapeForDynamicMxQuantWithDualAxis)
    .InferDataType(InferDataTypeForDynamicMxQuantWithDualAxis);
} // namespace ops
