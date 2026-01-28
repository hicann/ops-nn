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
 * \file dynamic_dual_level_mx_quant_infershape.cpp
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
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t Y_OUTTYPE_INDEX = 0;
constexpr int64_t LEVEL0_SCALE_OUTTYPE_INDEX = 1;
constexpr int64_t LEVEL1_SCALE_OUTTYPE_INDEX = 2;
constexpr size_t INDEX_ATTR_BLOCK0_SIZE = 1;
constexpr size_t INDEX_ATTR_BLOCK1_SIZE = 2;

graphStatus InferShapeForDynamicDualLevelMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForDynamicDualLevelMxQuant");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* level0ScaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, level0ScaleShape);

    gert::Shape* level1ScaleShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, level1ScaleShape);

    std::string errMsg = optiling::ConcatString("Input x rank[", xShape->GetDimNum(), "] should be in [1, 7].");
    OP_CHECK_IF(
        xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "%s", errMsg.c_str()), return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*level0ScaleShape);
        Ops::Base::SetUnknownRank(*level1ScaleShape);
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* level0BlockSize = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_BLOCK0_SIZE);
    const int32_t* level1BlockSize = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_BLOCK1_SIZE);
    size_t dim1 = static_cast<size_t>(xShape->GetDimNum() - 1);

    int64_t dimSize1 = 0;
    if (xShape->GetDim(dim1) == UNKNOWN_DIM_VALUE_) {
        dimSize1 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize1 = Ops::Base::CeilDiv(xShape->GetDim(dim1), static_cast<int64_t>(*level0BlockSize));
    }

    *level0ScaleShape = *xShape;
    level0ScaleShape->SetDim(dim1, dimSize1);

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, level0Scale shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*level0ScaleShape).c_str());

    int64_t dimSize2 = 0;
    if (xShape->GetDim(dim1) == UNKNOWN_DIM_VALUE_) {
        dimSize2 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize2 = Ops::Base::CeilDiv(xShape->GetDim(dim1), static_cast<int64_t>(*level1BlockSize));
        dimSize2 = (dimSize2 + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *level1ScaleShape = *xShape;
    level1ScaleShape->SetDim(dim1, dimSize2);
    level1ScaleShape->AppendDim(ALIGN_NUM);

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, level1Scale shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*level1ScaleShape).c_str());

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForDynamicDualLevelMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForDynamicDualLevelMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForDynamicDualLevelMxQuant");
    context->SetOutputDataType(Y_OUTTYPE_INDEX, ge::DT_FLOAT4_E2M1);
    context->SetOutputDataType(LEVEL0_SCALE_OUTTYPE_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(LEVEL1_SCALE_OUTTYPE_INDEX, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForDynamicDualLevelMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicDualLevelMxQuant)
    .InferShape(InferShapeForDynamicDualLevelMxQuant)
    .InferDataType(InferDataTypeForDynamicDualLevelMxQuant);
} // namespace ops
