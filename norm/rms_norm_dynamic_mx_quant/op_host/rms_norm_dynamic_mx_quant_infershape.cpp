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
 * \file rms_norm_dynamic_mx_quant_infershape.cpp
 * \brief RmsNormDynamicMxQuant InferShape
 */

#include <sstream>
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
    size_t dim_num = shape.GetDimNum();
    if (dim_num > 0) {
        for (size_t i = 0; i < dim_num - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(dim_num - 1);
    }
    oss << "]";
    return oss.str();
}

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr size_t INDEX_ATTR_DST_TYPE = 3;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t IDX_ONE = 0;
constexpr size_t IDX_TWO = 1;
constexpr size_t IDX_THREE = 2;
constexpr size_t OUTPUT_RSTD_IDX = 4;
static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

graphStatus InferShapeForRmsNormDynamicMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForRmsNormDynamicMxQuant");

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* mxScaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShape);

    OP_CHECK_IF(
        xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [1, 8].", xShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const bool* outputRstdPtr = attrsPtr->GetAttrPointer<bool>(OUTPUT_RSTD_IDX);
    bool hasOutputRstd = (outputRstdPtr != nullptr) ? *outputRstdPtr : false;
    gert::Shape* rstdShape = context->GetOutputShape(2);
    if (hasOutputRstd) {
        OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, mxScale, rstd shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*mxScaleShape);
        if (hasOutputRstd) {
            Ops::Base::SetUnknownRank(*rstdShape);
        }
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;
    size_t dim = xShape->GetDimNum() - 1;
    int64_t blockSz = MX_BLOCK_SIZE;
    int64_t dimSize = 0;
    if (xShape->GetDim(dim) == UNKNOWN_DIM_VALUE_) {
        dimSize = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize = Ops::Base::CeilDiv(xShape->GetDim(dim), blockSz);
        dimSize = (dimSize + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *mxScaleShape = *xShape;
    mxScaleShape->SetDim(dim, dimSize);
    mxScaleShape->AppendDim(ALIGN_NUM);

    if (hasOutputRstd) {
        *rstdShape = *xShape;
        for (size_t i = dim; i < rstdShape->GetDimNum(); i++) {
            rstdShape->SetDim(i, 1);
        }
    }

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, y shape is %s, mxscale shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*yShape).c_str(), Shape2String(*mxScaleShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForRmsNormDynamicMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForRmsNormDynamicMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForRmsNormDynamicMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
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
    context->SetOutputDataType(IDX_ONE, outDtype);
    context->SetOutputDataType(IDX_TWO, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(IDX_THREE, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForRmsNormDynamicMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormDynamicMxQuant)
    .InferShape(InferShapeForRmsNormDynamicMxQuant)
    .InferDataType(InferDataTypeForRmsNormDynamicMxQuant);
} // namespace ops
