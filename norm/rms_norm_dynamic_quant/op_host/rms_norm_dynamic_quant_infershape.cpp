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
 * \file rms_norm_dynamic_quant_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int X_IDX = 0;
static constexpr int GAMMA_IDX = 1;
static constexpr int SMOOTH_IDX = 2;
static constexpr int BETA_IDX = 3;

static constexpr int Y_IDX = 0;
static constexpr int OUT_SCALE_IDX = 1;
static constexpr int ATTR_INDEX_OF_DST_TYPE = 1;

using namespace ge;
using namespace Ops::Base;

namespace ops {
static bool InferReduceShape(const gert::Shape* xShape, const gert::Shape* gammaShape, gert::Shape* reduceShape)
{
    size_t gammaDimNum = gammaShape->GetDimNum();
    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum < gammaDimNum || gammaDimNum != 1) {
        return false;
    }
    reduceShape->SetDimNum(xDimNum - gammaDimNum);
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        reduceShape->SetDim(i, xShape->GetDim(i));
        OP_LOGI("InferShape4RmsNormDynamicQuant InferReduceShape", "reduceShape[%zu] = [%zu]", i,
                reduceShape->GetDim(i));
    }
    return true;
}

static ge::graphStatus InferShape4RmsNormDynamicQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4RmsNormDynamicQuant");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* outScaleShape = context->GetOutputShape(OUT_SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScaleShape);

    *yShape = *xShape;

    // unknown rank
    if (IsUnknownRank(*xShape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*outScaleShape);
        OP_LOGI(context, "End to do InferShape4RmsNormDynamicQuant with unknown rank.");
        return GRAPH_SUCCESS;
    }

    auto ret = InferReduceShape(xShape, gammaShape, outScaleShape);
    if (!ret) {
        std::string shapeDimMsg = std::to_string(gammaShape->GetDimNum()) + " and " +
                                  std::to_string(xShape->GetDimNum());
        std::string reasonMsg = "The shape dim of input gamma cannot be greater than that of input x";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "gamma and x", shapeDimMsg.c_str(),
                                                  reasonMsg.c_str());
        return GRAPH_FAILED;
    }
    OP_LOGI(context, "End to do InferShape4RmsNormDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4RmsNormDynamicQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormDynamicQuant");
    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            yDtype = static_cast<ge::DataType>(dstDtype);
        }
    }
    context->SetOutputDataType(Y_IDX, yDtype);
    context->SetOutputDataType(OUT_SCALE_IDX, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4RmsNormDynamicQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormDynamicQuant)
    .InferShape(InferShape4RmsNormDynamicQuant)
    .InferDataType(InferDtype4RmsNormDynamicQuant);
} // namespace ops
